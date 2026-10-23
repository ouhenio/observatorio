"""Stage 6 — privacy auditor over every cluster name + description (Clio §G.8, layer 4).

Qwen3.8 rates each "name: description" on the 1–5 identifiability scale (temperature
0.05, JSON-schema output). `publish` drops base clusters scoring below 3 and withholds
the names of parents that do.

Scores are cached by *content*, not cluster ID (review v3 follow-up §3): the key is a
hash of the exact audited text plus the audit policy fingerprint (prompt, schema,
model). A hierarchy rerun can reuse an ID for different content; that content is
re-audited. Unchanged leaf labels reuse their scores. Cache: runs/<id>/audit_cache.parquet.

    uv run python -m observatorio.audit --run wc10k
    uv run python -m observatorio.audit --run wc100k --seed-cache-from v3 --seed-model Qwen/Qwen3.8-27B
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from . import llm as L
from .config import LLM_MODEL, run_dir, write_config
from .prompts import (
    AUDIT_PREFILL,
    AUDIT_PROMPT,
    AUDIT_SCHEMA,
    PROMPT_VERSION,
    audit_fingerprint,
    fingerprint_of,
)

CACHE_COLUMNS = ("content_key", "fingerprint", "score", "justification")


def content_key(fingerprint: str, name: str, description: str) -> str:
    return hashlib.sha256(f"{fingerprint}\x00{name}: {description}".encode()).hexdigest()


def audited_items(rd, suffix: str = "") -> list[tuple[str, str, str]]:
    """(cluster_id, name, description) for base labels and every parent (level >= 1)."""
    items = duckdb.execute(
        f"SELECT cluster_id, name, description FROM '{rd / f'labels{suffix}.parquet'}'"
    ).fetchall()
    if (rd / f"hierarchy{suffix}.parquet").exists():
        items += duckdb.execute(
            f"SELECT cluster_id, name, description FROM '{rd / f'hierarchy{suffix}.parquet'}' WHERE level >= 1"
        ).fetchall()
    return items


def load_cache(rd) -> dict[str, tuple[int, str]]:
    path = rd / "audit_cache.parquet"
    if not path.exists():
        return {}
    t = pq.read_table(path).to_pylist()
    return {r["content_key"]: (r["score"], r["justification"]) for r in t}


def save_cache(rd, fingerprint: str, entries: dict[str, tuple[int, str]]) -> None:
    rows = [
        {"content_key": k, "fingerprint": fingerprint, "score": s, "justification": j}
        for k, (s, j) in entries.items()
        if s is not None
    ]
    old = (
        pq.read_table(rd / "audit_cache.parquet").to_pylist() if (rd / "audit_cache.parquet").exists() else []
    )
    merged = {r["content_key"]: r for r in old + rows}
    pq.write_table(pa.Table.from_pylist(list(merged.values())), rd / "audit_cache.parquet")


def seed_cache(rd, suffix: str, fingerprint: str, legacy_model: str | None = None) -> int:
    """Seed the cache from a previous audit (e.g. audit.v3) and the exact inputs it audited.

    Refuses unless the previous audit's policy is shown to match the current one:
      * audit.<suffix>.parquet carries a `fingerprint` column (written since v4) equal to the
        current fingerprint on every row; or
      * for older audits without it: the archived prompts.<suffix>.py reproduces the current
        fingerprint for `legacy_model`, which the operator must name explicitly (the model
        was not archived with those audits).
    """
    path = rd / f"audit.{suffix}.parquet"
    cols = set(pq.read_schema(path).names)
    if "fingerprint" in cols:
        found = {f for (f,) in duckdb.execute(f"SELECT DISTINCT fingerprint FROM '{path}'").fetchall()}
        if found != {fingerprint}:
            raise ValueError(f"audit.{suffix} was produced under policy {sorted(found)}, not {fingerprint}")
    else:
        archived = rd / f"prompts.{suffix}.py"
        if not archived.exists() or not legacy_model:
            raise ValueError(
                f"audit.{suffix} has no fingerprint; need archived prompts.{suffix}.py and --seed-model to verify it"
            )
        import importlib.util

        spec = importlib.util.spec_from_file_location(f"prompts_{suffix}", archived)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        old = fingerprint_of(mod.AUDIT_PROMPT, mod.AUDIT_SCHEMA, legacy_model)
        if old != fingerprint:
            raise ValueError(
                f"audit.{suffix} policy {old} (from {archived.name}) differs from current {fingerprint}"
            )
    rows = duckdb.execute(f"SELECT cluster_id, score, justification FROM '{path}'").fetchall()
    prev = {c: (s, j) for c, s, j in rows}
    entries = {}
    for cid, name, desc in audited_items(rd, f".{suffix}"):
        if cid in prev and prev[cid][0] is not None:
            entries[content_key(fingerprint, name, desc)] = prev[cid]
    save_cache(rd, fingerprint, entries)
    return len(entries)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--gpu-mem", type=float, default=0.9)
    p.add_argument("--seed-cache-from", metavar="SUFFIX", help="seed from audit.<SUFFIX>.parquet, then exit")
    p.add_argument(
        "--seed-model", help="model of a pre-v4 audit (no fingerprint recorded); required to verify it"
    )
    args = p.parse_args(argv)

    rd = run_dir(args.run)
    fingerprint = audit_fingerprint(LLM_MODEL)
    if args.seed_cache_from:
        n = seed_cache(rd, args.seed_cache_from, fingerprint, args.seed_model)
        print(f"seeded {n} cache entries from audit.{args.seed_cache_from} (fingerprint {fingerprint})")
        return 0

    items = audited_items(rd)
    keys = [content_key(fingerprint, n, d) for _c, n, d in items]
    cache = load_cache(rd)
    results: dict[int, tuple[int | None, str]] = {i: cache[k] for i, k in enumerate(keys) if k in cache}
    todo = [i for i in range(len(items)) if i not in results]
    print(f"auditing {len(todo)} of {len(items)} names/descriptions ({len(results)} reused from cache)")

    if todo:
        requests = [
            (
                [{"role": "user", "content": AUDIT_PROMPT.format(content=f"{items[i][1]}: {items[i][2]}")}],
                AUDIT_PREFILL,
            )
            for i in todo
        ]
        llm, tok = L.load(args.gpu_mem)
        L.LOG_DIR = rd / "debug" / "audit"

        def parse(out: tuple[str, str]) -> tuple[int | None, str]:
            obj = L.as_json(*out) or {}
            r = obj.get("rating")
            return (r if r in (1, 2, 3, 4, 5) else None), (obj.get("justification") or "")[:1000]

        outs = L.generate(
            llm, tok, requests, 0.05, 600, step="audit", json_schema=AUDIT_SCHEMA, return_finish=True
        )
        parsed = [parse(o) for o in outs]
        for attempt in (1, 2):  # unparsed would count as a failed audit in publish; resample first
            redo = [k for k, (r, _j) in enumerate(parsed) if r is None]
            if not redo:
                break
            print(f"retry {attempt}: {len(redo)} unparsed")
            again = L.generate(
                llm,
                tok,
                [requests[k] for k in redo],
                0.05,
                600,
                seed=attempt,
                step="audit",
                json_schema=AUDIT_SCHEMA,
                return_finish=True,
            )
            for k, o in zip(redo, again, strict=True):
                parsed[k] = parse(o)
        for i, r in zip(todo, parsed, strict=True):
            results[i] = r
        save_cache(rd, fingerprint, {keys[i]: results[i] for i in todo})

    scores = [results[i][0] for i in range(len(items))]
    pq.write_table(
        pa.table(
            {
                "cluster_id": [c for c, _n, _d in items],
                "score": scores,
                "justification": [results[i][1] for i in range(len(items))],
                "content_key": keys,
                "cached": [i not in todo for i in range(len(items))],
                "fingerprint": [fingerprint] * len(items),
            }
        ),
        rd / "audit.parquet",
    )
    write_config(
        args.run,
        "audit",
        {
            **vars(args),
            "prompt_version": PROMPT_VERSION,
            "fingerprint": fingerprint,
            "reused": len(items) - len(todo),
        },
    )
    dist = {s: scores.count(s) for s in sorted({s for s in scores if s is not None})}
    print(f"scores {dist}, unparsed {scores.count(None)} → {rd / 'audit.parquet'}")
    for (c, n, _d), s in zip(items, scores, strict=True):
        if s is not None and s < 3:
            print(f"  ✗ {c} ({s}): {n}")
    return 0


if __name__ == "__main__":
    rc = main()
    # Exit without interpreter teardown: vLLM's engine shutdown can abort with a
    # fatal GIL error *after* all outputs are written, which would fail the chain.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
