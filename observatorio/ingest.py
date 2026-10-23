"""Stage 0 — sample WildChat-1M and normalize it into conversations.parquet.

Keeps the facets that come for free (hashed_ip, country, model, turns, language,
moderation) and renders each conversation into the XML transcript the extractor
reads (Clio §G.2). Token-level truncation happens in `extract`, which owns the
tokenizer.

    uv run python -m observatorio.ingest --run wc100k --n 100000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

from .config import DATASET, run_dir, write_config

MOD_CATEGORIES = [
    "harassment",
    "harassment/threatening",
    "hate",
    "hate/threatening",
    "self-harm",
    "self-harm/instructions",
    "self-harm/intent",
    "sexual",
    "sexual/minors",
    "violence",
    "violence/graphic",
]


def render(conv: list[dict]) -> str:
    parts = []
    for m in conv:
        role = "user" if m["role"] == "user" else "assistant"
        parts.append(f"<{role}>\n{(m['content'] or '').strip()}\n</{role}>")
    return "<conversation>\n" + "\n".join(parts) + "\n</conversation>"


def moderation(mods: list[dict] | None) -> tuple[bool, list[str]]:
    """Collapse per-message OpenAI moderation into (any flagged, flagged categories)."""
    flagged, cats = False, set()
    for m in mods or []:
        flagged |= bool(m.get("flagged"))
        for c, v in (m.get("categories") or {}).items():
            if v:
                cats.add(c)
    return flagged, sorted(cats)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--n", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    snap = Path(snapshot_download(DATASET, repo_type="dataset", allow_patterns=["data/*.parquet"]))
    files = sorted(str(f) for f in (snap / "data").glob("*.parquet"))
    con = duckdb.connect()
    total = con.execute(f"SELECT count(*) FROM read_parquet({files})").fetchone()[0]
    print(f"{DATASET}: {total} conversations in {len(files)} shards; sampling {args.n} (seed {args.seed})")

    rows = (
        con.execute(
            f"""SELECT conversation_hash, hashed_ip, country, state, model, timestamp, turn, language,
                   toxic, redacted, conversation, openai_moderation
            FROM read_parquet({files})
            USING SAMPLE reservoir({args.n} ROWS) REPEATABLE ({args.seed})"""
        )
        .fetch_arrow_table()
        .to_pylist()
    )

    out = []
    for r in rows:
        flagged, cats = moderation(r.pop("openai_moderation"))
        conv = r.pop("conversation")
        out.append({**r, "mod_flagged": flagged, "mod_categories": cats, "transcript": render(conv)})
    out.sort(key=lambda r: r["conversation_hash"])  # stable order → stable chunking downstream

    path = run_dir(args.run) / "conversations.parquet"
    pq.write_table(pa.Table.from_pylist(out), path)
    write_config(
        args.run, "ingest", {"dataset": DATASET, "revision": snap.name, "n": args.n, "seed": args.seed}
    )
    users = len({r["hashed_ip"] for r in out})
    print(f"wrote {len(out)} conversations from {users} unique hashed_ip → {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
