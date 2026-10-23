"""Stateless, aggregate-only FastAPI viewer for a published `serve.duckdb`.

Reads the file named by ``OBS_SERVE_DB`` once, read-only, at startup (the
contract is in `observatorio.publish_schema`) and serves cluster-level JSON plus
the single-file frontend in ``web/index.html``. Nothing here can return a
conversation, a conversation hash, a hashed_ip or any per-conversation
attribute: only the contract's cluster tables are read, and `map_points` is
reduced to anonymous (x, y, cluster) triples.

Defence in depth on top of the pipeline's own privacy layers:
  * invalid publications (including clusters below privacy thresholds) are rejected;
  * language / country cells that stand for fewer than ``MIN_CELL`` conversations
    are folded into "Other".

Run (from the repo root, after ``source env.sh``)::

    OBS_SERVE_DB=$OBS_DATA/fixture/serve.duckdb \\
        uv run uvicorn observatorio.serve:app --port 8130

Endpoints: /healthz, /api/meta, /api/tree, /api/map, /api/points (binary),
/api/cluster/{id}, /api/search?q=.
"""

from __future__ import annotations

import logging
import os
import struct
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response

from .config import DATA
from .publish_schema import FACETS, ORDERED_KEYS, STAT_COLUMNS, check_schema

log = logging.getLogger("observatorio.serve")
WEB = Path(__file__).resolve().parent / "web"
DB_PATH = Path(os.environ.get("OBS_SERVE_DB", DATA / "fixture" / "serve.duckdb"))

MIN_CELL = 5  # language/country cells standing for fewer conversations fold into "Other"
SMALL_CELL_FACETS = ("language", "country")
NUMERIC_META = ("n_conversations", "n_users", "n_clusters_base", "n_suppressed_clusters")
NUMERIC_META += ("min_users", "min_conversations")
POINTS_MAGIC, POINTS_VERSION = b"OBSP", 1


def _r(v: float | None, nd: int = 5) -> float | None:
    return None if v is None else round(float(v), nd)


@dataclass
class Store:
    """Everything the API serves, loaded once from serve.duckdb."""

    meta: dict = field(default_factory=dict)
    clusters: dict[str, dict] = field(default_factory=dict)  # id -> row (+ "stats")
    children: dict[str | None, list[str]] = field(default_factory=dict)
    facets: dict[str, dict[str, list[tuple[str, float]]]] = field(default_factory=dict)
    baseline: dict[str, list[tuple[str, float]]] = field(default_factory=dict)
    overall: dict[str, float | None] = field(default_factory=dict)
    map_order: list[str] = field(default_factory=list)  # base cluster ids, index = points' cluster index
    points: bytes | None = None
    n_points: int = 0
    dropped: list[str] = field(default_factory=list)


S = Store()


def _load(path: Path) -> Store:
    st = Store()
    con = duckdb.connect(str(path), read_only=True)
    try:
        problems = check_schema(con)
        if problems:
            raise RuntimeError(f"{path} does not follow publish_schema: {problems}")
        st.meta = dict(con.execute("SELECT key, value FROM run_meta").fetchall())
        for k in NUMERIC_META:
            if st.meta.get(k) not in (None, ""):
                st.meta[k] = int(float(st.meta[k]))

        cols = ("cluster_id", "level", "parent_id", "name", "description", "n_conversations", "n_users")
        cols += ("share", "x", "y")
        rows = con.execute(f"SELECT {', '.join(cols)} FROM clusters").fetchall()
        for row in rows:
            c = dict(zip(cols, row, strict=True))
            st.clusters[c["cluster_id"]] = c

        stats = {
            r[0]: dict(zip(STAT_COLUMNS, r[1:], strict=True))
            for r in con.execute(
                f"SELECT cluster_id, {', '.join(STAT_COLUMNS)} FROM cluster_stats"
            ).fetchall()
        }
        for i, c in st.clusters.items():
            c["stats"] = {k: _r(v) for k, v in stats.get(i, dict.fromkeys(STAT_COLUMNS)).items()}
            st.children.setdefault(c["parent_id"], []).append(i)
        for kids in st.children.values():
            kids.sort(key=lambda k: -st.clusters[k]["n_conversations"])

        raw: dict[str, dict[str, dict[str, float]]] = {}
        for cid, facet, key, value in con.execute(
            "SELECT cluster_id, facet, key, value FROM cluster_facets"
        ).fetchall():
            if cid in st.clusters and facet in FACETS:
                d = raw.setdefault(cid, {}).setdefault(facet, {})
                d[key] = d.get(key, 0.0) + float(value)
        for cid, fs in raw.items():
            n = st.clusters[cid]["n_conversations"]
            st.facets[cid] = {f: _order(f, _fold(f, d, n)) for f, d in fs.items()}

        st.baseline, st.overall = _overall(st)
        _load_points(con, st)
    finally:
        con.close()
    log.info("loaded %s: %d clusters, %d map points", path, len(st.clusters), st.n_points)
    return st


def _fold(facet: str, d: dict[str, float], n: int) -> dict[str, float]:
    """Fold language/country cells that stand for < MIN_CELL conversations into 'Other'."""
    if facet not in SMALL_CELL_FACETS:
        return d
    out: dict[str, float] = {}
    for k, v in d.items():
        key = k if (k != "Other" and v * n >= MIN_CELL) else "Other"
        out[key] = out.get(key, 0.0) + v
    return out


def _order(facet: str, d: dict[str, float]) -> list[tuple[str, float]]:
    """Ordinal facets in their natural order, months chronologically, the rest by share."""
    if facet in ORDERED_KEYS:
        order = {k: i for i, k in enumerate(ORDERED_KEYS[facet])}
        items = sorted(d.items(), key=lambda kv: order.get(kv[0], len(order)))
    elif facet == "month":
        items = sorted(d.items())
    else:
        items = sorted(d.items(), key=lambda kv: (kv[0] == "Other", -kv[1]))
    return [(k, round(v, 5)) for k, v in items]


def _overall(st: Store) -> tuple[dict, dict]:
    """Corpus-wide facet distribution and mean stats, weighted over the top-level clusters."""
    roots = st.children.get(None, [])
    total = sum(st.clusters[r]["n_conversations"] for r in roots) or 1
    base: dict[str, dict[str, float]] = {}
    for r in roots:
        w = st.clusters[r]["n_conversations"] / total
        for f, items in st.facets.get(r, {}).items():
            acc = base.setdefault(f, {})
            for k, v in items:
                acc[k] = acc.get(k, 0.0) + v * w
    overall: dict[str, float | None] = {}
    for s in STAT_COLUMNS:
        vals = [(st.clusters[r]["stats"][s], st.clusters[r]["n_conversations"]) for r in roots]
        vals = [(v, n) for v, n in vals if v is not None]
        overall[s] = _r(sum(v * n for v, n in vals) / sum(n for _, n in vals)) if vals else None
    return {f: _order(f, d) for f, d in base.items()}, overall


def _load_points(con: duckdb.DuckDBPyConnection, st: Store) -> None:
    """Pack map_points as OBSP v1: magic, version, n, n_clusters, f32 x[n], f32 y[n], u16 cluster[n].

    The cluster index refers to `map_order`, the order of `clusters` in /api/map.
    """
    st.map_order = sorted(i for i, c in st.clusters.items() if c["level"] == 0)
    has = con.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = 'map_points'"
    ).fetchone()[0]
    if not has or len(st.map_order) >= 65535:
        return
    arr = con.execute("SELECT x, y, cluster_id FROM map_points").fetchnumpy()
    if len(arr["x"]) == 0:
        return
    ids, inv = np.unique(np.asarray(arr["cluster_id"], dtype=object).astype(str), return_inverse=True)
    pos = {c: i for i, c in enumerate(st.map_order)}
    lut = np.array([pos.get(c, -1) for c in ids], dtype=np.int32)
    cidx = lut[inv]
    keep = cidx >= 0  # points of dropped / unknown clusters are never served
    x = np.asarray(arr["x"], dtype=np.float32)[keep]
    y = np.asarray(arr["y"], dtype=np.float32)[keep]
    c = cidx[keep].astype(np.uint16)
    header = POINTS_MAGIC + struct.pack("<III", POINTS_VERSION, len(x), len(st.map_order))
    st.points = header + x.astype("<f4").tobytes() + y.astype("<f4").tobytes() + c.astype("<u2").tobytes()
    st.n_points = int(len(x))


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global S
    S = _load(DB_PATH)
    yield


app = FastAPI(title="Observatorio", lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=1024)


# ---------------------------------------------------------------- helpers
def _node(i: str) -> dict:
    c = S.clusters[i]
    return {
        "id": i,
        "level": c["level"],
        "name": c["name"],
        "n": c["n_conversations"],
        "users": c["n_users"],
        "share": _r(c["share"], 6),
    }


def _ancestors(i: str) -> list[str]:
    out, p = [], S.clusters[i]["parent_id"]
    while p:
        out.append(p)
        p = S.clusters[p]["parent_id"]
    return out[::-1]  # top first


# ---------------------------------------------------------------- endpoints
@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "clusters": len(S.clusters), "points": S.n_points}


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((WEB / "index.html").read_text(), headers={"Cache-Control": "no-store"})


@app.get("/api/meta")
def api_meta() -> JSONResponse:
    """Run metadata, cluster counts per level, and corpus-wide baselines."""
    levels: dict[int, int] = {}
    for c in S.clusters.values():
        levels[c["level"]] = levels.get(c["level"], 0) + 1
    return JSONResponse(
        {
            "meta": S.meta,
            "levels": {str(k): v for k, v in sorted(levels.items())},
            "top_level": max(levels) if levels else 0,
            "n_published_conversations": sum(
                S.clusters[r]["n_conversations"] for r in S.children.get(None, [])
            ),
            "overall": {"stats": S.overall, "facets": S.baseline},
            "has_points": S.points is not None,
            "n_points": S.n_points,
            "min_cell": MIN_CELL,
        }
    )


@app.get("/api/tree")
def api_tree() -> JSONResponse:
    """The whole hierarchy, top level first, children sorted by size."""

    def build(i: str) -> dict:
        c = S.clusters[i]
        node = _node(i) | {"x": _r(c["x"], 4), "y": _r(c["y"], 4), "stats": c["stats"]}
        node["children"] = [build(k) for k in S.children.get(i, [])]
        return node

    return JSONResponse({"roots": [build(r) for r in S.children.get(None, [])]})


@app.get("/api/map")
def api_map() -> JSONResponse:
    """Base clusters with position, size, ancestor chain and stats (for colouring).

    The list order is the index space of /api/points.
    """
    out = []
    for i in S.map_order:
        c = S.clusters[i]
        out.append(
            {
                "id": i,
                "name": c["name"],
                "x": _r(c["x"], 4),
                "y": _r(c["y"], 4),
                "n": c["n_conversations"],
                "share": _r(c["share"], 6),
                "path": _ancestors(i),
                "stats": c["stats"],
            }
        )
    return JSONResponse({"clusters": out, "points": "/api/points" if S.points else None})


@app.get("/api/points")
def api_points() -> Response:
    """Anonymous map positions (binary OBSP v1, see `_load_points`). 404 when the run has none."""
    if S.points is None:
        raise HTTPException(404, "this run has no map_points")
    return Response(
        S.points,
        media_type="application/octet-stream",
        headers={"X-Points": str(S.n_points), "X-Clusters": str(len(S.map_order))},
    )


@app.get("/api/cluster/{cluster_id}")
def api_cluster(cluster_id: str) -> JSONResponse:
    """Everything about one cluster (any level): text, sizes, stats, facets, children, ancestors."""
    c = S.clusters.get(cluster_id)
    if c is None:
        raise HTTPException(404, "unknown cluster")
    kids = S.children.get(cluster_id, [])
    return JSONResponse(
        _node(cluster_id)
        | {
            "description": c["description"],
            "x": _r(c["x"], 4),
            "y": _r(c["y"], 4),
            "stats": c["stats"],
            "facets": {f: [[k, v] for k, v in items] for f, items in S.facets.get(cluster_id, {}).items()},
            "ancestors": [_node(a) for a in _ancestors(cluster_id)],
            "children": [_node(k) | {"n_children": len(S.children.get(k, []))} for k in kids],
        }
    )


@app.get("/api/search")
def api_search(q: str = Query("", max_length=200), limit: int = Query(300, ge=1, le=1000)) -> JSONResponse:
    """Case-insensitive substring match over cluster names and descriptions."""
    needle = q.strip().lower()
    if len(needle) < 2:
        return JSONResponse({"q": q, "results": []})
    hits = []
    for i, c in S.clusters.items():
        in_name = needle in c["name"].lower()
        if in_name or needle in c["description"].lower():
            hits.append(_node(i) | {"match": "name" if in_name else "description"})
    hits.sort(key=lambda h: (h["match"] != "name", -h["level"], -h["n"]))
    return JSONResponse({"q": q, "total": len(hits), "results": hits[:limit]})
