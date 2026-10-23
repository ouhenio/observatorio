"""Export a publication as a static site: the viewer without a server.

Reuses the API's own response code (observatorio.serve) so the files are exactly what
the endpoints would return, then writes them under <out>/data/ next to a copy of
web/index.html that switches the viewer into static mode. Host the directory on any
static file host (GitHub Pages, Cloudflare Pages, `python -m http.server`).

Same privacy contract as the server: only cluster-level aggregates plus anonymous
(x, y, cluster) map points; serve._load rejects publications that break the contract.

    uv run python -m observatorio.export_static --db $OBS_DATA/runs/wc100k/serve.v3.duckdb --out demo
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from . import serve


def export(db: Path, out: Path) -> dict:
    serve.S = serve._load(db)
    st = serve.S
    data = out / "data"
    if out.exists():
        shutil.rmtree(out)
    (data / "cluster").mkdir(parents=True)

    def write_json(path: Path, response) -> None:
        path.write_bytes(response.body)

    write_json(data / "meta.json", serve.api_meta())
    write_json(data / "tree.json", serve.api_tree())
    map_payload = json.loads(serve.api_map().body)
    if map_payload.get("points"):
        map_payload["points"] = "data/points.bin"  # relative, so the site works under any sub-path
        (data / "points.bin").write_bytes(st.points)
    (data / "map.json").write_text(json.dumps(map_payload, separators=(",", ":")))
    for cid in st.clusters:
        write_json(data / "cluster" / f"{cid}.json", serve.api_cluster(cid))
    search = [serve._node(i) | {"description": c["description"]} for i, c in st.clusters.items()]
    (data / "search.json").write_text(json.dumps(search, separators=(",", ":"), ensure_ascii=False))

    html = (serve.WEB / "index.html").read_text()
    marker = "<script>"
    i = html.index(marker)  # first inline script = the app; the flag must be set before it runs
    html = html[:i] + "<script>window.OBS_STATIC = true;</script>\n" + html[i:]
    (out / "index.html").write_text(html)
    (out / ".nojekyll").write_text("")  # GitHub Pages: serve files as-is
    files = sum(1 for p in out.rglob("*") if p.is_file())
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    return {"clusters": len(st.clusters), "points": st.n_points, "files": files, "bytes": size}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("demo"))
    args = p.parse_args(argv)
    info = export(args.db, args.out)
    print(
        f"exported {info['clusters']} clusters, {info['points']} points → {args.out} "
        f"({info['files']} files, {info['bytes'] / 1e6:.1f} MB)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
