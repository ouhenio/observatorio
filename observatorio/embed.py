"""Stage 2 — embed each conversation's `request` summary with Qwen3-Embedding-8B.

Clio clusters the facet summary, not the raw conversation (§G.5). Qwen3-Embedding
is instruction-aware, so the summary is wrapped in a clustering instruction.
Vectors are L2-normalized and stored as float16 with a parallel hash list.

    uv run python -m observatorio.embed --run wc100k
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .config import EMBED_MODEL, run_dir, write_config

INSTRUCT = "Identify the kind of task a user is asking an AI assistant to perform"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--tag", default="")
    p.add_argument("--facet", default="request")
    p.add_argument("--model", default=EMBED_MODEL)
    p.add_argument("--gpu-mem", type=float, default=0.9)
    args = p.parse_args(argv)

    from vllm import LLM

    rd = run_dir(args.run)
    sfx = f"_{args.tag}" if args.tag else ""
    rows = pq.read_table(rd / f"facets{sfx}.parquet", columns=["conversation_hash", args.facet]).to_pylist()
    rows = [r for r in rows if r[args.facet]]
    texts = [f"Instruct: {INSTRUCT}\nQuery:{r[args.facet]}" for r in rows]
    print(f"embedding {len(texts)} `{args.facet}` summaries with {args.model}")

    write_config(args.run, f"embed{sfx}", {**vars(args), "instruct": INSTRUCT})
    llm = LLM(
        model=args.model, runner="pooling", max_model_len=1024, gpu_memory_utilization=args.gpu_mem, seed=0
    )
    t0 = time.time()
    outs = llm.embed(texts, use_tqdm=True)
    vecs = np.array([o.outputs.embedding for o in outs], dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    print(f"{vecs.shape} in {time.time() - t0:.0f}s")

    np.save(rd / f"embeddings{sfx}.npy", vecs.astype(np.float16))
    pq.write_table(
        pa.table({"conversation_hash": [r["conversation_hash"] for r in rows]}),
        rd / f"embeddings{sfx}.hashes.parquet",
    )
    print(f"stored → {rd / f'embeddings{sfx}.npy'}")
    return 0


if __name__ == "__main__":
    rc = main()
    # Exit without interpreter teardown: vLLM's engine shutdown can abort with a
    # fatal GIL error *after* all outputs are written, which would fail the chain.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
