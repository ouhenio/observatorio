"""Shared Qwen3.8 helpers for the naming stages (label, hierarchy, audit).

Each request is a chat plus a Clio-style assistant prefill appended after the
generation prompt (thinking off). `generate` batches everything through vLLM
offline; `tag` pulls the text of an XML-ish tag out of a completion.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .config import LLM_MODEL

# When set (e.g. runs/<id>/debug), every generate(step=...) call appends its raw
# completions and finish reasons to <LOG_DIR>/<step>.jsonl for inspection.
LOG_DIR: Path | None = None


def load(gpu_mem: float = 0.9, max_model_len: int = 16384, model: str = LLM_MODEL, max_num_seqs: int = 256):
    from transformers import AutoTokenizer
    from vllm import LLM

    tok = AutoTokenizer.from_pretrained(model)
    llm = LLM(
        model=model,
        dtype="bfloat16",
        language_model_only=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_mem,
        enable_prefix_caching=True,
        # Qwen3.8 is hybrid: each running sequence needs one Mamba state block, and
        # vLLM's offline default (1024) exceeds what fits beside a 16k context.
        max_num_seqs=max_num_seqs,
        seed=0,
    )
    return llm, tok


def generate(
    llm,
    tok,
    requests: list[tuple[list[dict], str]],
    temperature: float,
    max_tokens: int,
    stop=None,
    seed: int = 0,
    step: str = "",
    choices: list[str] | None = None,
    json_schema: dict | list[dict] | None = None,
    return_finish: bool = False,
):
    """requests: (messages, prefill). Returns prefill + completion for each, in order
    (or (text, finish_reason) pairs with return_finish). `choices` constrains every
    completion to exactly one of the given strings (vLLM structured outputs)."""
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    prompts = [
        tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        + prefill
        for m, prefill in requests
    ]

    def params(schema=None):
        extra = {}
        if choices:
            extra["structured_outputs"] = StructuredOutputsParams(choice=choices)
        elif schema:
            extra["structured_outputs"] = StructuredOutputsParams(json=schema)
        return SamplingParams(temperature=temperature, max_tokens=max_tokens, stop=stop, seed=seed, **extra)

    # A list of schemas gives each request its own (e.g. outliers limited to that group's child IDs).
    sp = [params(sch) for sch in json_schema] if isinstance(json_schema, list) else params(json_schema)
    outs = llm.generate(prompts, sp, use_tqdm=True)
    texts = [prefill + o.outputs[0].text for (_m, prefill), o in zip(requests, outs, strict=True)]
    finish = [o.outputs[0].finish_reason for o in outs]
    if LOG_DIR and step:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOG_DIR / f"{step}.jsonl", "a") as f:
            for i, (t, fr) in enumerate(zip(texts, finish, strict=True)):
                f.write(
                    json.dumps({"i": i, "seed": seed, "finish": fr, "text": t}, ensure_ascii=False) + "\n"
                )
    return list(zip(texts, finish, strict=True)) if return_finish else texts


def tag(text: str, name: str) -> str | None:
    """Last <name>...</name> in the text (prefills open tags, so allow a missing opener)."""
    hits = re.findall(rf"<{name}>(.*?)</{name}>", text, flags=re.DOTALL)
    if hits:
        return hits[-1].strip()
    m = re.search(rf"<{name}>(.*)$", text, flags=re.DOTALL)
    return m.group(1).strip() if m else None


def as_json(text: str, finish: str | None = None) -> dict | None:
    """The JSON object a schema-constrained completion produced; None if it was cut
    off (finish != 'stop') or does not parse."""
    if finish not in (None, "stop"):
        return None
    try:
        obj = json.loads(text[text.index("{") :])
    except (ValueError, json.JSONDecodeError):
        return None
    return obj if isinstance(obj, dict) else None


def closed(text: str, name: str) -> str | None:
    """Text of the last *closed* <name>...</name>; None if the tag was never closed
    (a generation cut off mid-tag must not pass as a valid answer)."""
    hits = re.findall(rf"<{name}>(.*?)</{name}>", text, flags=re.DOTALL)
    return hits[-1].strip() if hits else None


def clean_name(name: str) -> str:
    """Strip tag fragments, wrapping quotes/brackets and trailing punctuation from a generated name."""
    name = name.split("<", 1)[0]
    return name.strip().strip("\"'“”‘’[]").strip().rstrip(".")


def numbered(text: str | None) -> list[str]:
    """Items of a list answer: '1. foo', '1) foo', '- foo', '* foo', '**1. foo**', or <cluster>foo</cluster>."""
    text = text or ""
    tagged = re.findall(r"<cluster>\s*(.*?)\s*</cluster>", text, flags=re.DOTALL)
    if tagged:
        return [clean_name(t) for t in tagged if t.strip()]
    out = []
    for line in text.splitlines():
        line = line.strip().replace("**", "")
        m = re.match(r"(?:\d+[.)]|[-*•])\s+(.+)", line)
        if m:
            out.append(clean_name(m.group(1)))
    return [o for o in out if o]


def list_answer(text: str) -> list[str]:
    """List items from the <answer> block; falls back to whatever follows the scratchpad."""
    items = numbered(tag(text + "</answer>", "answer"))
    if not items and "</scratchpad>" in text:
        items = numbered(text.rsplit("</scratchpad>", 1)[1])
    return items
