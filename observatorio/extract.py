"""Stage 1 — extract Clio facets per conversation with Qwen3.8-27B (vLLM offline).

Facets (Clio §G.4.1): request, task (open-ended, PII-free, English) and concern
(1–5). Language comes from WildChat's own field, so it isn't re-extracted.

Work is split into chunks of --chunk conversations; each chunk is written to
parts/extract_NNNN.parquet as soon as it finishes and skipped on rerun, so a
crash loses at most one chunk. `--limit` + `--tag bench` gives a quick
throughput/quality benchmark without touching the main outputs.

    uv run python -m observatorio.extract --run wc100k --limit 1000 --tag bench
    uv run python -m observatorio.extract --run wc100k
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq

from .config import LLM_MODEL, run_dir, write_config
from .prompts import FACETS, facet_messages

TRUNC_MARK = "\n[... middle of conversation truncated ...]\n"


def truncate(tok, texts: list[str], max_tokens: int, tail: int) -> tuple[list[str], int]:
    """Keep the head and tail of over-long transcripts (the request is usually early)."""
    ids = tok(texts, add_special_tokens=False).input_ids
    out, n_cut = [], 0
    for text, t in zip(texts, ids, strict=True):
        if len(t) <= max_tokens:
            out.append(text)
            continue
        n_cut += 1
        out.append(tok.decode(t[: max_tokens - tail]) + TRUNC_MARK + tok.decode(t[-tail:]))
    return out, n_cut


def ordered_requests(n: int, lag: int) -> list[tuple[int, str]]:
    """Interleave facets so a conversation's later facets are queued `lag`
    conversations after its first one — late enough that its prefix is already
    in the KV/Mamba cache, early enough that it hasn't been evicted."""
    facets = list(FACETS)
    order = []
    for i in range(n + lag):
        if i < n:
            order.append((i, facets[0]))
        if i >= lag:
            order.extend((i - lag, f) for f in facets[1:])
    return order


def metric_totals(llm) -> dict[str, float]:
    want = ("prefix_cache_queries", "prefix_cache_hits", "prompt_tokens", "generation_tokens")
    out = {}
    try:
        for m in llm.get_metrics():
            key = next((w for w in want if m.name.endswith(w)), None)
            if key and hasattr(m, "value"):
                out[key] = out.get(key, 0) + float(m.value)
    except Exception as e:  # metrics are diagnostic only
        out["error"] = str(e)
    return out


def parse(facet: str, prefill: str, text: str) -> str | int | None:
    text = text.split("</answer>")[0].strip()
    if facet == "concern":
        m = re.search(r"[1-5]", text)
        return int(m.group(0)) if m else None
    tail = prefill.split("<answer>")[-1].strip()
    return f"{tail} {text}".strip()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--tag", default="", help="output suffix, e.g. 'bench'")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--chunk", type=int, default=5000)
    p.add_argument("--lag", type=int, default=32, help="conversations between a convo's 1st and later facets")
    p.add_argument("--max-conv-tokens", type=int, default=4000)
    p.add_argument("--tail-tokens", type=int, default=500)
    p.add_argument("--model", default=LLM_MODEL)
    p.add_argument("--max-num-seqs", type=int, default=512)
    p.add_argument("--gpu-mem", type=float, default=0.92)
    p.add_argument("--mamba-cache-mode", default=None, help="vLLM mamba_cache_mode override (align|all)")
    p.add_argument("--temperature", type=float, default=0.2)
    args = p.parse_args(argv)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rd = run_dir(args.run)
    convs = pq.read_table(
        rd / "conversations.parquet", columns=["conversation_hash", "transcript"]
    ).to_pylist()
    # WildChat repeats some conversations verbatim across users (same hash, e.g.
    # shared prompt templates). Extract each once; facets join back by hash.
    seen: set[str] = set()
    convs = [c for c in convs if not (c["conversation_hash"] in seen or seen.add(c["conversation_hash"]))]
    if args.limit:
        convs = convs[: args.limit]
    parts = rd / f"parts{'_' + args.tag if args.tag else ''}"
    parts.mkdir(exist_ok=True)
    chunks = [convs[i : i + args.chunk] for i in range(0, len(convs), args.chunk)]
    todo = [(k, c) for k, c in enumerate(chunks) if not (parts / f"extract_{k:04d}.parquet").exists()]
    print(f"{len(convs)} conversations, {len(chunks)} chunks, {len(todo)} to do")
    if not todo:
        return merge(rd, parts, args.tag)

    tok = AutoTokenizer.from_pretrained(args.model)
    engine = {
        "model": args.model,
        "dtype": "bfloat16",
        "language_model_only": True,
        "max_model_len": args.max_conv_tokens + 2048,
        "max_num_seqs": args.max_num_seqs,
        "gpu_memory_utilization": args.gpu_mem,
        "enable_prefix_caching": True,
        "disable_log_stats": False,
        "seed": 0,
    }
    if args.mamba_cache_mode:
        engine["mamba_cache_mode"] = args.mamba_cache_mode
    write_config(args.run, f"extract{'_' + args.tag if args.tag else ''}", {**engine, **vars(args)})
    llm = LLM(**engine)

    sp = {
        f: SamplingParams(
            temperature=args.temperature,
            max_tokens=150 if open_ended else 4,
            stop=["</answer>"],
            seed=0,
        )
        for f, (_, _, open_ended) in FACETS.items()
    }

    for k, chunk in todo:
        t0 = time.time()
        texts, n_cut = truncate(tok, [c["transcript"] for c in chunk], args.max_conv_tokens, args.tail_tokens)
        order = ordered_requests(len(chunk), min(args.lag, len(chunk)))
        prompts, prefills = [], []
        for i, f in order:
            msgs, prefill = facet_messages(texts[i], f)
            chat = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            prompts.append(chat + prefill)
            prefills.append(prefill)
        m0 = metric_totals(llm)
        outs = llm.generate(prompts, [sp[f] for _, f in order], use_tqdm=True)
        m1 = metric_totals(llm)

        rows = []
        for (i, f), prefill, o in zip(order, prefills, outs, strict=True):
            c = o.outputs[0]
            val = parse(f, prefill, c.text)
            rows.append(
                {
                    "conversation_hash": chunk[i]["conversation_hash"],
                    "facet": f,
                    "value": json.dumps(val, ensure_ascii=False),
                    "finish_reason": c.finish_reason,
                    "n_prompt": len(o.prompt_token_ids),
                    "n_out": len(c.token_ids),
                }
            )
        pq.write_table(pa.Table.from_pylist(rows), parts / f"extract_{k:04d}.parquet")

        dt = time.time() - t0
        n_in = sum(r["n_prompt"] for r in rows)
        n_out = sum(r["n_out"] for r in rows)
        d = {key: m1.get(key, 0) - m0.get(key, 0) for key in m1 if key != "error"}
        hit = (
            d.get("prefix_cache_hits", 0) / d["prefix_cache_queries"]
            if d.get("prefix_cache_queries")
            else None
        )
        stats = {
            "chunk": k,
            "convs": len(chunk),
            "truncated": n_cut,
            "seconds": round(dt, 1),
            "conv_per_s": round(len(chunk) / dt, 2),
            "prompt_tok": n_in,
            "out_tok": n_out,
            "prompt_tok_per_s": round(n_in / dt),
            "prefix_hit_rate": round(hit, 3) if hit is not None else None,
            "length_cutoffs": sum(r["finish_reason"] == "length" and r["facet"] != "concern" for r in rows),
        }
        print("STATS " + json.dumps(stats), flush=True)
        with open(rd / f"extract_stats{'_' + args.tag if args.tag else ''}.jsonl", "a") as fh:
            fh.write(json.dumps(stats) + "\n")

    return merge(rd, parts, args.tag)


def merge(rd, parts, tag: str) -> int:
    """Pivot all chunk parts into facets[_tag].parquet: one row per conversation."""
    import duckdb

    out = rd / f"facets{'_' + tag if tag else ''}.parquet"
    duckdb.execute(
        f"""COPY (
              SELECT conversation_hash,
                     json_extract_string(max(value) FILTER (facet = 'request'), '$') AS request,
                     json_extract_string(max(value) FILTER (facet = 'task'), '$')    AS task,
                     TRY_CAST(max(value) FILTER (facet = 'concern') AS INTEGER)       AS concern
              FROM read_parquet('{parts}/extract_*.parquet')
              GROUP BY conversation_hash ORDER BY conversation_hash
            ) TO '{out}' (FORMAT parquet)"""
    )
    n = duckdb.execute(f"SELECT count(*) FROM '{out}'").fetchone()[0]
    print(f"merged {n} conversations → {out}")
    return 0


if __name__ == "__main__":
    rc = main()
    # Exit without interpreter teardown: vLLM's engine shutdown can abort with a
    # fatal GIL error *after* all outputs are written, which would fail the chain.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
