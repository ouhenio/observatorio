"""Stage 4 — name and describe each base cluster (Clio §G.5, privacy layer 3).

For every base cluster that passed the privacy threshold, Qwen3.8 sees 50 random
`request` summaries from inside the cluster and the 50 summaries closest to its
centroid that are NOT in it, and writes a two-sentence past-tense description
plus an imperative name that tells the cluster apart from its neighbours.
Temperature 1.0 as in Clio.

    uv run python -m observatorio.label --run wc10k
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from . import llm as L
from . import quality
from .cluster import load_embeddings
from .config import run_dir, write_config
from .prompts import LABEL_PREFILL, LABEL_PROMPT, LABEL_SCHEMA, NAME_PRIVACY, PROMPT_VERSION, REQUEST_CRITERIA

REQUEST_PREFIX = re.compile(r"^The user's overall request for the assistant is to\s*", re.I)


def statement(summary: str) -> str:
    """Drop the extractor's fixed prefill so 100 statements don't repeat it."""
    s = REQUEST_PREFIX.sub("", summary.strip())
    return s[:1].upper() + s[1:]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--n-in", type=int, default=50)
    p.add_argument("--n-out", type=int, default=50)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--gpu-mem", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    rd = run_dir(args.run)
    cfg_dim = json.loads((rd / "config.json").read_text())["cluster"]["dim"]
    vecs, hashes = load_embeddings(rd, "", cfg_dim or None)
    cents = np.load(rd / "centroids.npy")
    assign = dict(
        duckdb.execute(f"SELECT conversation_hash, cluster_id FROM '{rd / 'clusters.parquet'}'").fetchall()
    )
    summ = dict(
        duckdb.execute(
            f"SELECT conversation_hash, request FROM '{rd / 'facets.parquet'}' WHERE request IS NOT NULL"
        ).fetchall()
    )
    passing = [
        c
        for (c,) in duckdb.execute(
            f"SELECT cluster_id FROM '{rd / 'cluster_sizes.parquet'}' WHERE passes ORDER BY 1"
        ).fetchall()
    ]
    members: dict[str, list[str]] = {}
    for h, c in assign.items():
        members.setdefault(c, []).append(h)

    rng = random.Random(args.seed)
    requests = []
    for cid in passing:
        inside = [h for h in members[cid] if h in summ]
        sample = rng.sample(inside, min(args.n_in, len(inside)))
        centroid = cents[int(cid.split("-")[1])]
        order = np.argsort(-(vecs @ centroid))
        contrast = []
        for i in order:
            h = hashes[i]
            if assign[h] != cid and h in summ:
                contrast.append(h)
                if len(contrast) == args.n_out:
                    break
        msg = LABEL_PROMPT.format(
            criteria=REQUEST_CRITERIA,
            privacy=NAME_PRIVACY,
            answers="\n".join(f"<answer>{statement(summ[h])}</answer>" for h in sample),
            contrastive="\n".join(f"<answer>{statement(summ[h])}</answer>" for h in contrast),
        )
        requests.append(([{"role": "user", "content": msg}], LABEL_PREFILL))
    print(
        f"labeling {len(passing)} clusters ({args.n_in} in-cluster + {args.n_out} contrastive summaries each)"
    )

    llm, tok = L.load(args.gpu_mem)
    L.LOG_DIR = rd / "debug" / "label"

    def parse(out: tuple[str, str]) -> tuple[str, str, list[str]]:
        obj = L.as_json(*out) or {}
        name = L.clean_name(obj.get("name", ""))
        desc = (obj.get("description") or "").strip()
        return name, desc, quality.check(name, desc) + ([] if obj else ["invalid or cut-off JSON"])

    outs = L.generate(
        llm, tok, requests, args.temperature, 600, step="label", json_schema=LABEL_SCHEMA, return_finish=True
    )
    parsed = [parse(o) for o in outs]
    for attempt in (1, 2, 3):  # resample failing outputs with a new seed (a fixed seed would repeat them)
        redo = [i for i, (_n, _d, issues) in enumerate(parsed) if issues]
        if not redo:
            break
        print(f"retry {attempt}: {len(redo)} failing the quality gate")
        again = L.generate(
            llm,
            tok,
            [requests[i] for i in redo],
            args.temperature,
            600,
            seed=attempt,
            step="label",
            json_schema=LABEL_SCHEMA,
            return_finish=True,
        )
        for i, o in zip(redo, again, strict=True):
            parsed[i] = parse(o)
    names = [n for n, _d, _i in parsed]
    descs = [d for _n, d, _i in parsed]
    issues = ["; ".join(i) for _n, _d, i in parsed]
    pq.write_table(
        pa.table({"cluster_id": passing, "name": names, "description": descs, "quality_issues": issues}),
        rd / "labels.parquet",
    )
    write_config(args.run, "label", {**vars(args), "prompt_version": PROMPT_VERSION})
    bad = sum(bool(i) for i in issues)
    print(f"wrote {len(passing)} labels ({bad} still failing the quality gate) → {rd / 'labels.parquet'}")
    for cid, n, i in zip(passing, names, issues, strict=True):
        if i:
            print(f"  ✗ {cid}: {n!r} — {i}")
    for n, d in list(zip(names, descs, strict=True))[:8]:
        print(f"  · {n} — {d[:120]}")
    return 0


if __name__ == "__main__":
    rc = main()
    # Exit without interpreter teardown: vLLM's engine shutdown can abort with a
    # fatal GIL error *after* all outputs are written, which would fail the chain.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
