"""Stage 7 — publish a run as the viewer's `serve.duckdb` (contract: publish_schema).

Everything user-visible is aggregated here, and this is the last privacy gate:
  * a base cluster is published only if it passes the threshold (>= min_users
    unique hashed_ip and >= min_conversations) and, when the auditor has run,
    its name/description scored >= --min-audit (Clio §G.8);
  * parents exist only above published children, so every published cluster
    passes the threshold too; a parent whose own name fails the audit keeps its
    place in the tree under a withheld name;
  * language / country / model keys used by fewer than min_users unique users in
    a cluster are folded into "Other";
  * `map_points` is (x, y, base cluster) only, rounded and shuffled.

Optional inputs from the GPU stages, all in runs/<id>/ (missing = placeholder):
  labels.parquet     cluster_id, name, description                  (stage 4, base clusters)
  hierarchy.parquet  cluster_id, level, parent_id, name, description (stage 5; level-0 rows
                     only carry parent_id, levels >= 1 carry names)
  audit.parquet      cluster_id, score (1-5)                         (stage 6, any level)

    uv run python -m observatorio.publish --run wc100k
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import duckdb
from huggingface_hub import snapshot_download

from . import quality
from .config import DATASET, EMBED_MODEL, LLM_MODEL, run_dir
from .ingest import MOD_CATEGORIES
from .publish_schema import COLUMNS, check_schema, create_schema

WITHHELD = ("Name withheld", "This cluster's generated name did not pass the privacy audit.")
QUALITY_WITHHELD = ("Name withheld", "This category's generated name did not pass the quality check.")
TOP_FOLDED = ("language", "country", "model")  # categorical facets folded by unique users


def placeholder(cid: str) -> tuple[str, str]:
    return f"Unlabeled cluster {cid}", "Not labeled yet: the label stage has not run for this run."


def label_issues(base_ids: list[str], labels: dict | None) -> dict[str, list[str]]:
    """None means an intentional unlabeled dev run; an existing file must be complete."""
    if labels is None:
        return {}
    missing = sorted(set(base_ids) - labels.keys())
    if missing:
        raise ValueError(f"labels.parquet is incomplete: missing {len(missing)} labels, e.g. {missing[:5]}")
    return {cid: quality.check(*labels[cid]) for cid in base_ids}


def ancestor_chains(base_ids: list[str], parent: dict, upper: dict) -> dict[str, list[str]]:
    """Validate reachable hierarchy before aggregation or writing publication files."""
    chains = {}
    for base in base_ids:
        chain, node = [base], parent.get(base)
        while node is not None:
            if node in chain:
                raise ValueError(f"Hierarchy cycle at {node}")
            if node not in upper:
                raise ValueError(f"Missing parent node {node}")
            if upper[node].get("coherent") is False:
                raise ValueError(f"Incoherent parent {node}; rebuild hierarchy before publication")
            chain.append(node)
            node = parent.get(node)
        chains[base] = chain
    if not chains:
        raise ValueError("No publishable base clusters remain; existing publication was not changed")
    return chains


def moderation(rd: Path) -> Path:
    """Max OpenAI-moderation category score per conversation row, joined back from
    the raw dataset (ingest keeps only flags, which WildChat-1M zeroes). Cached."""
    out = rd / "moderation.parquet"
    if out.exists():
        return out
    snap = Path(snapshot_download(DATASET, repo_type="dataset", allow_patterns=["data/*.parquet"]))
    files = sorted(str(f) for f in (snap / "data").glob("*.parquet"))
    cats = ", ".join(f"struct_extract(m.category_scores, '{c}')" for c in MOD_CATEGORIES)
    duckdb.execute(
        f"""COPY (
              WITH raw AS (
                SELECT conversation_hash, hashed_ip, timestamp,
                       list_max(list_transform(openai_moderation, m -> greatest({cats}))) AS mod_max
                FROM read_parquet({files}))
              SELECT v.conversation_hash, v.hashed_ip, v.timestamp, max(r.mod_max) AS mod_max
              FROM '{rd / "conversations.parquet"}' v
              LEFT JOIN raw r USING (conversation_hash, hashed_ip, timestamp)
              GROUP BY ALL
            ) TO '{out}' (FORMAT parquet)"""
    )
    return out


def load_tree(rd: Path, base_ids: list[str]) -> tuple[dict[str, str | None], dict[str, dict]]:
    """(parent_of for every node, {node_id: {level, name, description}} for levels >= 1)."""
    path = rd / "hierarchy.parquet"
    if not path.exists():
        return dict.fromkeys(base_ids), {}
    cols = {c for (c,) in duckdb.execute(f"SELECT name FROM parquet_schema('{path}')").fetchall()}
    coherent = "coherent" if "coherent" in cols else "NULL"  # older hierarchies lack the flag
    rows = duckdb.execute(
        f"SELECT cluster_id, level, parent_id, name, description, {coherent} FROM '{path}'"
    ).fetchall()
    parent = {cid: par for cid, _lvl, par, _n, _d, _c in rows}
    nodes = {
        cid: {"level": lvl, "name": n, "description": d, "coherent": c}
        for cid, lvl, _p, n, d, c in rows
        if lvl >= 1
    }
    missing = [b for b in base_ids if b not in parent]
    if missing:
        raise SystemExit(
            f"hierarchy.parquet has no parent for {len(missing)} base clusters, e.g. {missing[:3]}"
        )
    return parent, nodes


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--out", type=Path, default=None, help="default: runs/<id>/serve.duckdb")
    p.add_argument("--min-audit", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    rd = run_dir(args.run)
    cfg = json.loads((rd / "config.json").read_text())
    min_users = int(cfg["cluster"]["min_users"])
    min_convs = int(cfg["cluster"]["min_conversations"])
    con = duckdb.connect()

    # ── which base clusters are publishable ───────────────────────────────
    sizes = con.execute(f"SELECT cluster_id, passes FROM '{rd / 'cluster_sizes.parquet'}'").fetchall()
    audit = {}
    if (rd / "audit.parquet").exists():
        audit = dict(con.execute(f"SELECT cluster_id, score FROM '{rd / 'audit.parquet'}'").fetchall())
    labels = {}
    if (rd / "labels.parquet").exists():
        labels = {
            c: (n, d)
            for c, n, d in con.execute(
                f"SELECT cluster_id, name, description FROM '{rd / 'labels.parquet'}'"
            ).fetchall()
        }
    passing = [c for c, ok in sizes if ok]
    # Check completeness even for labels that will subsequently fail the audit.
    all_label_issues = label_issues(passing, labels if (rd / "labels.parquet").exists() else None)
    audit_ok = [c for c in passing if not audit or (audit.get(c) or 0) >= args.min_audit]
    # Quality gate (separate from privacy): only once real labels exist; placeholders are for dev runs.
    bad_quality = {c: all_label_issues.get(c, []) for c in audit_ok}
    published = [c for c in audit_ok if not bad_quality.get(c)]
    counts = {
        "n_suppressed_threshold": len(sizes) - len(passing),
        "n_suppressed_audit": len(passing) - len(audit_ok),
        "n_suppressed_quality": len(audit_ok) - len(published),
    }
    n_suppressed = len(sizes) - len(published)
    for c, issues in bad_quality.items():
        if issues:
            print(f"  quality-withheld {c}: {labels[c][0]!r} — {'; '.join(issues)}")
    parent, upper = load_tree(rd, published)

    # Ancestor chain per published base cluster → one column per level.
    chains = ancestor_chains(published, parent, upper)
    depth = max(len(c) for c in chains.values())
    if any(len(c) != depth for c in chains.values()):
        raise SystemExit("hierarchy is ragged: every base cluster must have the same number of ancestors")
    con.execute(
        f"CREATE TABLE chain (cid0 TEXT, {', '.join(f'cid{i} TEXT' for i in range(1, depth))})"
        if depth > 1
        else "CREATE TABLE chain (cid0 TEXT)"
    )
    con.executemany(f"INSERT INTO chain VALUES ({', '.join('?' * depth)})", list(chains.values()))

    # ── one row per published conversation (duplicates included) ─────────
    mod = moderation(rd)
    con.execute(
        f"""CREATE TABLE conv AS
            SELECT ch.*, v.hashed_ip, v.timestamp AS ts, c.x, c.y,
                   coalesce(v.language, 'Unknown') AS language,
                   coalesce(v.country, 'Unknown')  AS country,
                   coalesce(v.model, 'Unknown')    AS model,
                   strftime(v.timestamp, '%Y-%m')  AS month,
                   v.turn,
                   CASE WHEN v.turn <= 1 THEN '1' WHEN v.turn = 2 THEN '2'
                        WHEN v.turn <= 5 THEN '3-5' ELSE '6+' END AS turns,
                   f.concern, m.mod_max
            FROM '{rd / "conversations.parquet"}' v
            JOIN '{rd / "clusters.parquet"}' c USING (conversation_hash)
            JOIN chain ch ON ch.cid0 = c.cluster_id
            LEFT JOIN '{rd / "facets.parquet"}' f USING (conversation_hash)
            LEFT JOIN '{mod}' m USING (conversation_hash, hashed_ip, timestamp)"""
    )
    total, n_users, t0, t1 = con.execute(
        "SELECT count(*), count(DISTINCT hashed_ip), min(ts), max(ts) FROM conv"
    ).fetchone()
    split = con.execute("SELECT quantile_disc(ts, 0.5) FROM conv").fetchone()[0]
    n_early, n_late = con.execute(
        "SELECT count(*) FILTER (ts < ?), count(*) FILTER (ts >= ?) FROM conv", [split, split]
    ).fetchone()
    print(
        f"publishing {len(published)} base clusters ({n_suppressed} suppressed), {total} conversations, depth {depth}"
    )

    # ── aggregate every level ─────────────────────────────────────────────
    clusters, stats, facets = [], [], []
    parents_withheld = 0
    for lvl in range(depth):
        col = f"cid{lvl}"
        for row in con.execute(
            f"""SELECT {col}, count(*), count(DISTINCT hashed_ip), avg(x), avg(y),
                       avg(concern), avg((concern >= 4)::INT), avg((language <> 'English')::INT),
                       avg(turn), avg(mod_max),
                       count(*) FILTER (ts >= ?) / ?, count(*) FILTER (ts < ?) / ?,
                       any_value({f"cid{lvl + 1}" if lvl + 1 < depth else "NULL"})
                FROM conv GROUP BY 1 ORDER BY 1""",
            [split, max(n_late, 1), split, max(n_early, 1)],
        ).fetchall():
            cid, n, u, x, y, cm, ch, ne, tm, mm, s_late, s_early, par = row
            if lvl == 0:
                name, desc = labels.get(cid) or placeholder(cid)
            else:
                name, desc = upper[cid]["name"], upper[cid]["description"]
            if audit and cid in audit and (audit[cid] or 0) < args.min_audit and lvl > 0:
                name, desc = WITHHELD
            elif lvl > 0 and quality.check(name, desc):
                name, desc = QUALITY_WITHHELD
                parents_withheld += 1
            growth = (s_late - s_early) / (s_late + s_early) if s_late + s_early else 0.0
            clusters.append((cid, lvl, par, name, desc, n, u, n / total, x, y))
            stats.append((cid, cm, ch, ne, tm, mm, growth))

        for facet in ("language", "country", "model", "concern", "turns", "month"):
            where = "WHERE concern IS NOT NULL" if facet == "concern" else ""
            key = "concern::TEXT" if facet == "concern" else facet
            fold = facet in TOP_FOLDED
            facets += con.execute(
                f"""WITH cells AS (
                      SELECT {col} AS cid, {key} AS k, count(*) AS n, count(DISTINCT hashed_ip) AS u
                      FROM conv {where} GROUP BY 1, 2),
                    folded AS (
                      SELECT cid, CASE WHEN {fold} AND u < ? THEN 'Other' ELSE k END AS k, n FROM cells)
                    SELECT cid, '{facet}', k, sum(n) / sum(sum(n)) OVER (PARTITION BY cid)
                    FROM folded GROUP BY cid, k""",
                [min_users],
            ).fetchall()

    # ── write serve.duckdb, validate, then swap into place ───────────────
    out = args.out or rd / "serve.duckdb"
    tmp = out.with_suffix(".tmp.duckdb")
    tmp.unlink(missing_ok=True)
    w = duckdb.connect(str(tmp))
    create_schema(w)
    meta = {
        "run_id": args.run,
        "dataset": DATASET,
        "n_conversations": total,
        "n_users": n_users,
        "n_clusters_base": len(published),
        "n_suppressed_clusters": n_suppressed,
        **counts,
        "n_parents_quality_withheld": parents_withheld,
        "n_parents_incoherent": sum(1 for u in upper.values() if u.get("coherent") is False),
        "prompt_version": cfg.get("label", {}).get("prompt_version", "clio-appendix-g"),
        "min_users": min_users,
        "min_conversations": min_convs,
        "llm_model": LLM_MODEL,
        "embed_model": EMBED_MODEL,
        "date_start": str(t0.date()),
        "date_end": str(t1.date()),
        "growth_split": str(split.date()),
        "labels": "generated" if labels else "placeholder",
        "hierarchy": "generated" if upper else "none",
        "audit": "applied" if audit else "not run",
        "created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "is_fixture": "false",
    }
    w.executemany("INSERT OR REPLACE INTO run_meta VALUES (?, ?)", [(k, str(v)) for k, v in meta.items()])
    for table, rows in (("clusters", clusters), ("cluster_stats", stats), ("cluster_facets", facets)):
        w.executemany(f"INSERT INTO {table} VALUES ({', '.join('?' * len(COLUMNS[table]))})", rows)
    w.close()

    # map_points straight from the working DB (100k rows: skip Python round-trip).
    con.execute(f"ATTACH '{tmp}' AS w")
    con.execute(f"SELECT setseed({args.seed / 2**31})")
    con.execute(
        "INSERT INTO w.map_points SELECT round(x, 3)::REAL, round(y, 3)::REAL, cid0 FROM conv ORDER BY random()"
    )
    con.execute("DETACH w")

    w = duckdb.connect(str(tmp), read_only=True)
    problems = check_schema(w)
    w.close()
    if problems:
        raise SystemExit(f"publication failed validation ({len(problems)} problems): {problems[:10]}")
    tmp.replace(out)
    print(f"wrote {out} — {len(clusters)} clusters over {depth} level(s), {total} map points; schema OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
