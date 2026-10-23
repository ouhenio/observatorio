"""The published `serve.duckdb` contract: the only thing the viewer ever reads.

Stage 7 (publish) writes this file; `observatorio.serve` reads it read-only. It
holds cluster-level aggregates only: no conversation text, conversation hash,
hashed_ip, timestamp or per-conversation attribute. The one per-conversation
table, `map_points`, carries a 2-D position and the (published) base cluster id
and nothing else, so the map can draw Clio's point cloud.

Conventions
-----------
* Cluster ids: ``L<level>-<n>``, e.g. ``L0-0123`` (base), ``L1-017``, ``L2-03``.
  Level 0 is the finest; the highest level present is the top of the tree.
* Every published cluster already satisfies the privacy thresholds recorded in
  ``run_meta`` (``min_users`` unique users and ``min_conversations``).
* ``share`` columns are fractions in [0, 1], not percentages.
* ``cluster_facets`` has rows for every level: a parent's distribution is the
  size-weighted aggregate of its children. For each (cluster_id, facet) the
  values sum to ~1. Rare keys should be folded into ``'Other'`` upstream.
* x / y live in the 2-D UMAP space; a parent's position is the
  conversation-weighted mean of its children.

Usage (pipeline side)::

    import duckdb
    from observatorio.publish_schema import create_schema

    con = duckdb.connect("serve.duckdb")
    create_schema(con)
    con.executemany("INSERT INTO clusters VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
"""

from __future__ import annotations

import math

SCHEMA_VERSION = "1"

DDL = """
-- Free-form key/value metadata about the run. Expected keys (all TEXT):
--   schema_version, run_id, dataset, n_conversations, n_users, n_clusters_base,
--   n_suppressed_clusters, min_users, min_conversations, llm_model, embed_model,
--   date_start, date_end, created_at, is_fixture ('true' only for synthetic data)
CREATE TABLE IF NOT EXISTS run_meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS clusters (
    cluster_id      TEXT PRIMARY KEY,  -- 'L0-0123' (base), 'L1-017', 'L2-03' (top)
    level           INTEGER NOT NULL,  -- 0 = base (finest), increasing upward
    parent_id       TEXT,              -- NULL at the top level
    name            TEXT NOT NULL,     -- imperative sentence, e.g. 'Debug Python web scraping scripts'
    description     TEXT NOT NULL,     -- two sentences, past tense
    n_conversations INTEGER NOT NULL,
    n_users         INTEGER NOT NULL,  -- unique users (hashed_ip), >= min_users
    share           DOUBLE NOT NULL,   -- n_conversations / total published conversations
    x               DOUBLE NOT NULL,   -- 2-D map position (UMAP space)
    y               DOUBLE NOT NULL
);

CREATE TABLE IF NOT EXISTS cluster_facets (
    cluster_id TEXT NOT NULL,
    facet      TEXT NOT NULL,  -- one of FACETS
    key        TEXT NOT NULL,  -- category value, e.g. 'English', 'Russia', 'gpt-4-0314', '3-5', '2023-04'
    value      DOUBLE NOT NULL -- share of the cluster's conversations with that key
);

CREATE TABLE IF NOT EXISTS cluster_stats (
    cluster_id         TEXT PRIMARY KEY,
    concern_mean       DOUBLE,  -- mean 1-5 concern score
    concern_high_share DOUBLE,  -- share with concern >= 4
    non_english_share  DOUBLE,
    turns_mean         DOUBLE,  -- mean user turns
    moderation_mean    DOUBLE,  -- mean of each conversation's max OpenAI-moderation category score (0-1)
    growth             DOUBLE   -- (s_late - s_early) / (s_late + s_early), s = cluster's share of all conversations
                                --   in each half, split at the median date (run_meta.growth_split); -1..1
);

-- Optional. One row per conversation in a published base cluster, rows shuffled.
-- Positions only: NO hash, ip, text or timestamp. Powers the map's point cloud.
CREATE TABLE IF NOT EXISTS map_points (
    x          REAL NOT NULL,
    y          REAL NOT NULL,
    cluster_id TEXT NOT NULL  -- a level-0 cluster_id present in `clusters`
);
"""

# Facet names and, where the keys are ordinal, their display order.
FACETS = ("language", "country", "model", "concern", "turns", "month")
ORDERED_KEYS = {
    "concern": ("1", "2", "3", "4", "5"),
    "turns": ("1", "2", "3-5", "6+"),
}

# Column lists, in DDL order, for inserts and for the viewer's startup check.
COLUMNS = {
    "run_meta": ("key", "value"),
    "clusters": (
        "cluster_id",
        "level",
        "parent_id",
        "name",
        "description",
        "n_conversations",
        "n_users",
        "share",
        "x",
        "y",
    ),
    "cluster_facets": ("cluster_id", "facet", "key", "value"),
    "cluster_stats": (
        "cluster_id",
        "concern_mean",
        "concern_high_share",
        "non_english_share",
        "turns_mean",
        "moderation_mean",
        "growth",
    ),
    "map_points": ("x", "y", "cluster_id"),
}
REQUIRED_TABLES = ("run_meta", "clusters", "cluster_facets", "cluster_stats")
OPTIONAL_TABLES = ("map_points",)
STAT_COLUMNS = COLUMNS["cluster_stats"][1:]


def create_schema(con) -> None:
    """Create every table of the contract on a DuckDB connection (idempotent)."""
    con.execute(DDL)
    con.execute(
        "INSERT OR REPLACE INTO run_meta VALUES ('schema_version', ?)",
        [SCHEMA_VERSION],
    )


def check_schema(con) -> list[str]:
    """Validate the publication before any of it is exposed by the viewer."""
    problems = []
    have = {
        (t, c)
        for t, c in con.execute(
            "SELECT table_name, column_name FROM information_schema.columns WHERE table_schema = 'main'"
        ).fetchall()
    }
    tables = {t for t, _ in have}
    for table in REQUIRED_TABLES + OPTIONAL_TABLES:
        if table not in tables:
            if table in REQUIRED_TABLES:
                problems.append(f"missing table {table}")
            continue
        problems += [f"missing column {table}.{c}" for c in COLUMNS[table] if (table, c) not in have]
    if problems:
        return problems

    meta = dict(con.execute("SELECT key, value FROM run_meta").fetchall())
    if meta.get("schema_version") != SCHEMA_VERSION:
        problems.append("unsupported schema_version")
    thresholds = {}
    for key, default in (("min_users", 10), ("min_conversations", 20)):
        try:
            thresholds[key] = int(meta.get(key, default))
            if thresholds[key] < 1:
                raise ValueError
        except (TypeError, ValueError):
            problems.append(f"{key} must be a positive integer")
            thresholds[key] = default

    rows = con.execute("SELECT * FROM clusters").fetchall()
    columns = [d[0] for d in con.description]
    nodes = {r["cluster_id"]: r for r in (dict(zip(columns, row, strict=True)) for row in rows)}
    children = {}
    for cid, c in nodes.items():
        parent = c["parent_id"]
        children.setdefault(parent, []).append(cid)
        if c["level"] < 0:
            problems.append(f"{cid}: negative hierarchy level")
        if parent is not None and (parent not in nodes or nodes[parent]["level"] <= c["level"]):
            # Strictly increasing parent levels also rule out every cycle.
            problems.append(f"{cid}: missing parent or non-increasing parent level")
        if not (
            thresholds["min_users"] <= c["n_users"] <= c["n_conversations"]
            and c["n_conversations"] >= thresholds["min_conversations"]
        ):
            problems.append(f"{cid}: invalid counts or below privacy thresholds; republish aggregates")
        if not all(math.isfinite(c[k]) for k in ("x", "y", "share")) or not 0 <= c["share"] <= 1:
            problems.append(f"{cid}: invalid position or share")
    total = sum(nodes[c]["n_conversations"] for c in children.get(None, []))
    for cid, c in nodes.items():
        kids = children.get(cid, [])
        if (c["level"] == 0 and kids) or (c["level"] > 0 and not kids):
            problems.append(f"{cid}: invalid leaf level")
        if kids and sum(nodes[k]["n_conversations"] for k in kids) != c["n_conversations"]:
            problems.append(f"{cid}: parent count differs from child total")
        if total and not math.isclose(c["share"], c["n_conversations"] / total, abs_tol=1e-6):
            problems.append(f"{cid}: inconsistent share")

    facet_sums = {}
    for cid, facet, _key, value in con.execute(
        "SELECT cluster_id, facet, key, value FROM cluster_facets"
    ).fetchall():
        if cid not in nodes or facet not in FACETS or not math.isfinite(value) or not 0 <= value <= 1:
            problems.append(f"{cid}: invalid facet value")
        facet_sums[cid, facet] = facet_sums.get((cid, facet), 0) + value
    if any(not math.isclose(v, 1, abs_tol=1e-4) for v in facet_sums.values()):
        problems.append("facet shares must sum to one")
    for row in con.execute(f"SELECT {', '.join(COLUMNS['cluster_stats'])} FROM cluster_stats").fetchall():
        cid = row[0]
        if cid not in nodes:
            problems.append(f"{cid}: orphan statistics")
        for key, value in zip(STAT_COLUMNS, row[1:], strict=True):
            lo, hi = {"concern_mean": (1, 5), "growth": (-1, 1), "turns_mean": (0, math.inf)}.get(key, (0, 1))
            if value is not None and (not math.isfinite(value) or not lo <= value <= hi):
                problems.append(f"{cid}: invalid {key}")
    return problems
