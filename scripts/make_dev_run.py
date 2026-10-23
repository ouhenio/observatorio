"""Build a sub-run from another run's finished extract chunks.

Copies the conversations whose facets already exist and merges those facets.
With `--embeddings tfidf` (default) it also stands in for Qwen3-Embedding with
TF-IDF + SVD, so cluster → publish → viewer can be exercised on CPU while the
GPU is busy (dev only: not the real embeddings). With `--embeddings none` the
run is left for the real `embed` stage, e.g. a 10k validation run.

    uv run python scripts/make_dev_run.py --from wc100k --to dev
    uv run python scripts/make_dev_run.py --from wc100k --to wc10k --embeddings none
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import duckdb  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from observatorio.config import run_dir, write_config  # noqa: E402
from observatorio.extract import merge  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--from", dest="src", default="wc100k")
    p.add_argument("--to", dest="dst", default="dev")
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--embeddings", choices=["tfidf", "none"], default="tfidf")
    args = p.parse_args()

    src, dst = run_dir(args.src), run_dir(args.dst)
    merge(dst, src / "parts", "")  # → dst/facets.parquet from the finished chunks
    duckdb.execute(
        f"""COPY (SELECT * FROM '{src / "conversations.parquet"}'
                  WHERE conversation_hash IN (SELECT conversation_hash FROM '{dst / "facets.parquet"}'))
            TO '{dst / "conversations.parquet"}' (FORMAT parquet)"""
    )
    if args.embeddings == "none":
        n = duckdb.execute(f"SELECT count(*) FROM '{dst / 'facets.parquet'}'").fetchone()[0]
        print(f"run '{args.dst}': {n} conversations with facets; run `embed` next")
        return 0
    rows = duckdb.execute(
        f"SELECT conversation_hash, request FROM '{dst / 'facets.parquet'}' WHERE request IS NOT NULL"
    ).fetchall()

    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    x = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words="english").fit_transform(
        [r[1] for r in rows]
    )
    v = TruncatedSVD(args.dim, random_state=0).fit_transform(x).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    np.save(dst / "embeddings.npy", v.astype(np.float16))
    pq.write_table(pa.table({"conversation_hash": [r[0] for r in rows]}), dst / "embeddings.hashes.parquet")
    write_config(
        args.dst, "embed", {"method": "tfidf+svd (DEV ONLY)", "dim": args.dim, "source_run": args.src}
    )
    print(f"dev run '{args.dst}': {len(rows)} conversations with facets, embeddings {v.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
