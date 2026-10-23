"""Stage 3 — base clusters (k-means) + 2-D map (UMAP) + privacy threshold.

Clio §G.5/§G.6: k-means in embedding space over the facet summaries, and a 2-D
UMAP (cosine, n_neighbors 15, min_dist 0) for the map. Qwen3-Embedding is
Matryoshka-trained, so the first `--dim` dimensions (renormalized) are a valid,
cheaper embedding; Clio itself clusters 768-d vectors.

Clustering runs over unique conversation hashes; WildChat's verbatim duplicates
(same hash, different users) then inherit their hash's cluster, so they still
count toward cluster size and unique users.

Privacy layer 2 is computed here so later GPU stages only label clusters that
can be published: a cluster passes with >= --min-users unique hashed_ip AND
>= --min-conversations conversations. `publish` re-checks it.

Outputs in runs/<id>/:
  clusters.parquet       conversation_hash, cluster_id, x, y   (one row per unique hash)
  cluster_sizes.parquet  cluster_id, n_conversations, n_users, n_unique, passes
  centroids.npy          float32 [k, dim] unit centroids, row i = cluster L0-i

    uv run python -m observatorio.cluster --run wc100k
"""

from __future__ import annotations

import argparse
import time

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .config import run_dir, write_config


def base_id(i: int) -> str:
    return f"L0-{i:04d}"


def load_embeddings(rd, tag: str, dim: int | None) -> tuple[np.ndarray, list[str]]:
    sfx = f"_{tag}" if tag else ""
    vecs = np.load(rd / f"embeddings{sfx}.npy").astype(np.float32)
    hashes = pq.read_table(rd / f"embeddings{sfx}.hashes.parquet").column("conversation_hash").to_pylist()
    if dim and dim < vecs.shape[1]:
        vecs = vecs[:, :dim]
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    return vecs, hashes


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--tag", default="", help="embeddings suffix, e.g. 'dev'")
    p.add_argument("--k", type=int, default=None, help="base clusters (default: unique conversations / 100)")
    p.add_argument(
        "--per-cluster", type=int, default=100, help="target conversations per cluster when --k is unset"
    )
    p.add_argument("--dim", type=int, default=1024, help="Matryoshka truncation before clustering (0 = full)")
    p.add_argument("--max-iter", type=int, default=100)
    p.add_argument("--n-init", type=int, default=1)
    p.add_argument("--umap-neighbors", type=int, default=15)
    p.add_argument("--min-users", type=int, default=10)
    p.add_argument("--min-conversations", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    from sklearn.cluster import KMeans

    rd = run_dir(args.run)
    vecs, hashes = load_embeddings(rd, args.tag, args.dim or None)
    k = args.k or max(2, round(len(hashes) / args.per_cluster))
    print(f"{len(hashes)} unique conversations, dim {vecs.shape[1]}, k={k}")

    t0 = time.time()
    km = KMeans(n_clusters=k, n_init=args.n_init, max_iter=args.max_iter, random_state=args.seed).fit(vecs)
    labels = km.labels_
    cents = km.cluster_centers_.astype(np.float32)
    cents /= np.linalg.norm(cents, axis=1, keepdims=True) + 1e-12
    print(f"k-means: {time.time() - t0:.0f}s, inertia {km.inertia_:.1f}")

    t0 = time.time()
    import umap

    # No random_state: a seeded UMAP runs single-threaded (~10x slower at 100k).
    # The map layout is for exploration only; no result depends on it.
    xy = umap.UMAP(
        n_neighbors=args.umap_neighbors, min_dist=0.0, n_components=2, metric="cosine", n_jobs=-1
    ).fit_transform(vecs)
    print(f"UMAP: {time.time() - t0:.0f}s")

    ids = [base_id(int(c)) for c in labels]
    pq.write_table(
        pa.table(
            {
                "conversation_hash": hashes,
                "cluster_id": ids,
                "x": xy[:, 0].astype(np.float32),
                "y": xy[:, 1].astype(np.float32),
            }
        ),
        rd / "clusters.parquet",
    )
    np.save(rd / "centroids.npy", cents)

    # Sizes over ALL conversation rows (duplicates included) and unique users.
    sizes = duckdb.execute(
        f"""SELECT c.cluster_id,
                   count(*)                           AS n_conversations,
                   count(DISTINCT v.hashed_ip)        AS n_users,
                   count(DISTINCT v.conversation_hash) AS n_unique
            FROM '{rd / "conversations.parquet"}' v
            JOIN '{rd / "clusters.parquet"}' c USING (conversation_hash)
            GROUP BY 1 ORDER BY 1"""
    ).fetch_arrow_table()
    passes = [
        u >= args.min_users and n >= args.min_conversations
        for u, n in zip(
            sizes.column("n_users").to_pylist(), sizes.column("n_conversations").to_pylist(), strict=True
        )
    ]
    sizes = sizes.append_column("passes", pa.array(passes))
    pq.write_table(sizes, rd / "cluster_sizes.parquet")

    write_config(args.run, "cluster", {**vars(args), "k": k, "n_unique": len(hashes)})
    n_pass = sum(passes)
    kept = sum(n for n, ok in zip(sizes.column("n_conversations").to_pylist(), passes, strict=True) if ok)
    total = sum(sizes.column("n_conversations").to_pylist())
    print(
        f"{n_pass}/{k} clusters pass the threshold (>= {args.min_users} users, >= {args.min_conversations} convs), "
        f"covering {kept}/{total} conversations ({100 * kept / total:.1f}%)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
