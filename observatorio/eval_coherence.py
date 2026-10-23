"""Evaluation set for the group-coherence judgment (v4 review, proposed change 1).

`build` assembles candidate groups with their contents:
  * every group the v4 run sent to the coherence check (main and leftover passes, L1 and
    L2), recovered by *replaying* hierarchy.build_level on the logged model outputs in
    debug/hierarchy/ (no GPU; the replay is checked against the published v4 tree);
  * reference groups from earlier versions (v3 `L2-008`, v2 `L2-002`, ...).
It samples ~60 of them (all v4 dissolutions, the named references, stratified controls),
assigns a fixed dev/held-out split, and writes:
  docs/eval/coherence-set-v1.jsonl  — machine-readable, labels empty until reviewed
  docs/eval/coherence-set-v1.md     — the same groups for human review

Two label dimensions, evaluated separately (verification comment 3):
  coherent        — do the children share one activity at this level of abstraction?
  title_faithful  — does the name/description describe the children without attributing a
                    behavior (e.g. safeguard evasion) that some of them do not show?

    uv run python -m observatorio.eval_coherence build --run wc100k
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from unittest.mock import patch

import duckdb
import numpy as np

from . import hierarchy
from .config import run_dir

EVAL_DIR = Path(__file__).resolve().parents[1] / "docs" / "eval"


def grouped_split(rows: list[dict], seed: int = 7) -> None:
    """Split connected families, never individual examples or version-specific IDs.

    Explicit family hints join semantic variants; overlap edges additionally join
    groups sharing >=2 child names and >=40% Jaccard or >=70% containment.
    A subset-sum allocation approximates one third held out without reading labels.
    """
    parents = list(range(len(rows)))

    def root(i):
        while parents[i] != i:
            i = parents[i]
        return i

    names = [{c["name"].casefold().strip() for c in r["children"]} for r in rows]
    for i, a in enumerate(names):
        for j, b in enumerate(names[:i]):
            hint = rows[i].get("family_hint")
            same_hint = hint and hint == rows[j].get("family_hint")
            shared = len(a & b)
            overlap = shared >= 2 and (shared / len(a | b) >= 0.4 or shared / min(len(a), len(b)) >= 0.7)
            if same_hint or overlap:
                parents[root(i)] = root(j)
    families = {}
    for i in range(len(rows)):
        families.setdefault(root(i), []).append(i)
    groups = sorted(families.values(), key=lambda g: min(rows[i]["id"] for i in g))
    random.Random(seed).shuffle(groups)
    reachable = {0: []}
    for k, group in enumerate(groups):
        for total, chosen in list(reachable.items()):
            reachable.setdefault(total + len(group), chosen + [k])
    target = round(len(rows) / 3)
    held = set(reachable[min(reachable, key=lambda n: (abs(n - target), n))])
    for k, group in enumerate(groups):
        family = "family-" + min(rows[i]["id"] for i in group)
        for i in group:
            rows[i].update(family=family, split="heldout" if k in held else "dev", split_version="grouped-v2")


class Replay:
    """Stands in for llm.generate: returns the logged outputs of each step in call order."""

    def __init__(self, log_dir: Path):
        self.log_dir, self.pos = log_dir, {}

    def __call__(self, llm, tok, requests, *a, step="", return_finish=False, **k):
        path = self.log_dir / f"{step}.jsonl"
        lines = path.read_text().splitlines()
        start = self.pos.get(step, 0)
        chunk = [json.loads(x) for x in lines[start : start + len(requests)]]
        if len(chunk) != len(requests):
            raise RuntimeError(f"replay ran out of logged outputs for {step}")
        self.pos[step] = start + len(requests)
        out = [(c["text"], c["finish"]) for c in chunk]
        return out if return_finish else [t for t, _f in out]


def replay_run(rd: Path) -> tuple[list[dict], list[dict]]:
    """Re-run v4's levels on the logged outputs; return (judged groups, final top parents)."""
    judged = []
    real_judge = hierarchy.judge_coherence

    def recording_judge(llm, tok, children, groups, named, step):
        verdicts = real_judge(llm, tok, children, groups, named, step)
        for g, r, v in zip(groups, named, verdicts, strict=True):
            judged.append(
                {
                    "step": step,
                    "name": r["name"],
                    "description": r["description"],
                    "children": [
                        {k: children[i][k] for k in ("cluster_id", "name", "description", "n")} for i in g
                    ],
                    "v4_verdict": {
                        "coherent": v["coherent"],
                        "shared_activity": v["shared_activity"],
                        "outliers": [children[i]["cluster_id"] for i in v["outliers"]],
                    },
                }
            )
        return verdicts

    base = [
        {"cluster_id": c, "level": 0, "name": nm, "description": d, "n": n}
        for c, nm, d, n in duckdb.execute(
            f"""SELECT l.cluster_id, l.name, l.description, s.n_conversations
                FROM '{rd / "labels.parquet"}' l JOIN '{rd / "cluster_sizes.parquet"}' s USING (cluster_id)
                ORDER BY 1"""
        ).fetchall()
    ]
    cfg = json.loads((rd / "config.json").read_text())["hierarchy"]
    rng = random.Random(cfg["seed"])
    fake_emb = lambda texts: np.random.default_rng(0).normal(size=(len(texts), 8))  # noqa: E731
    with (
        patch.object(hierarchy.L, "generate", Replay(rd / "debug" / "hierarchy")),
        patch.object(hierarchy, "judge_coherence", recording_judge),
    ):
        current = base
        for lvl, target in enumerate(cfg["level_targets"], start=1):
            current = hierarchy.build_level(None, None, fake_emb, current, target, lvl, rng, cfg["m_outside"])
    return judged, current


def version_groups(rd: Path, suffix: str, ids: list[str]) -> list[dict]:
    """Named reference groups from an earlier publication, with children and sizes."""
    con = duckdb.connect(str(rd / f"serve.{suffix}.duckdb"), read_only=True)
    out = []
    for pid in ids:
        name, desc, lvl = con.execute(
            "SELECT name, description, level FROM clusters WHERE cluster_id=?", [pid]
        ).fetchone()
        kids = con.execute(
            "SELECT cluster_id, name, description, n_conversations FROM clusters WHERE parent_id=? ORDER BY 4 DESC",
            [pid],
        ).fetchall()
        out.append(
            {
                "step": f"{suffix}_L{lvl}_final",
                "name": name,
                "description": desc,
                "children": [{"cluster_id": c, "name": n, "description": d, "n": k} for c, n, d, k in kids],
                "v4_verdict": None,
            }
        )
    return out


def build(args) -> int:
    rd = run_dir(args.run)
    judged, tops = replay_run(rd)
    published = {
        n
        for (n,) in duckdb.execute(
            f"SELECT name FROM '{rd / 'hierarchy.parquet'}' WHERE level = 2"
        ).fetchall()
    }
    replayed = {t["name"] for t in tops}
    if replayed != published:
        raise SystemExit(f"replay does not reproduce v4's top level: {sorted(replayed ^ published)[:5]}")
    print(f"replay reproduces v4's {len(published)} top-level categories; {len(judged)} groups were judged")

    rng = random.Random(args.seed)
    dissolved = [g for g in judged if not g["v4_verdict"]["coherent"]]
    kept = [g for g in judged if g["v4_verdict"]["coherent"]]
    refs = version_groups(rd, "v3", ["L2-008", "L2-003"]) + version_groups(rd, "v2", ["L2-002"])
    safety = [g for g in kept if g["step"].startswith("L2") and "safety" in g["name"].lower()]
    n_controls = max(0, args.size - len(dissolved) - len(refs) - len(safety))
    pool = [g for g in kept if g not in safety]
    by_level = {lvl: [g for g in pool if g["step"].startswith(f"L{lvl}")] for lvl in (1, 2)}
    controls = rng.sample(by_level[2], min(len(by_level[2]), n_controls // 3))
    controls += rng.sample(by_level[1], min(len(by_level[1]), n_controls - len(controls)))
    chosen = dissolved + safety + refs + controls
    rng.shuffle(chosen)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for k, g in enumerate(chosen):
        rows.append(
            {
                "id": f"G{k + 1:02d}",
                "split": "dev",
                "source": g["step"],
                **{f: g[f] for f in ("name", "description", "children", "v4_verdict")},
                "labels": {"coherent": None, "title_faithful": None, "note": ""},
                "draft": None,
            }
        )
    grouped_split(rows, args.seed)
    out = EVAL_DIR / "coherence-set-v1.jsonl"
    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists (it may hold reviewed labels); pass --force to rebuild")
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(
        f"wrote {len(rows)} groups → {out}: {len(dissolved)} v4 dissolutions, {len(safety)} safety, "
        f"{len(refs)} earlier-version references, {len(controls)} controls; "
        f"{sum(r['split'] == 'heldout' for r in rows)} held out"
    )
    return 0


def load_set(path: Path = EVAL_DIR / "coherence-set-v1.jsonl") -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def label_of(row: dict) -> dict | None:
    """Only complete, attributed confirmations may enter evaluation metrics."""
    lab = row.get("labels") or {}
    if not lab.get("coherent"):
        return None
    if lab.get("coherent") not in {"yes", "partial", "no"} or lab.get("title_faithful") not in {"yes", "no"}:
        raise ValueError(f"{row['id']}: incomplete or invalid confirmed judgments")
    if not lab.get("reviewer") or lab.get("rubric_version") != "v2" or not lab.get("note"):
        raise ValueError(f"{row['id']}: confirmation requires reviewer, rubric_version v2, and rationale")
    ids = {c["cluster_id"] for c in row["children"]}
    outliers = set(lab.get("outliers", []))
    if not outliers <= ids:
        raise ValueError(f"{row['id']}: outlier IDs must belong to the group")
    if lab["coherent"] == "partial" and (not outliers or len(ids - outliers) < 2):
        raise ValueError(
            f"{row['id']}: partial requires outliers and a retained core of at least two children"
        )
    if lab["coherent"] == "yes" and outliers:
        raise ValueError(f"{row['id']}: coherent yes cannot have coherence outliers")
    return lab


def baseline(rows: list[dict]) -> dict:
    """How the v4 check's verdicts compare with the labels, per split. Coherent verdicts
    are compared with coherent in {yes, partial} vs no (partial = trim, not dissolve)."""
    out = {}
    for split in ("dev", "heldout"):
        r = [x for x in rows if x["split"] == split and x["v4_verdict"] and label_of(x)]
        cell = lambda v4, lab, r=r: sum(  # noqa: E731
            1 for x in r if x["v4_verdict"]["coherent"] is v4 and (label_of(x)["coherent"] != "no") is lab
        )
        out[split] = {
            "unconfirmed": sum(1 for x in rows if x["split"] == split and label_of(x) is None),
            "judged_by_v4": len(r),
            "false_dissolutions": cell(False, True),
            "correct_dissolutions": cell(False, False),
            "kept_coherent_or_partial": cell(True, True),
            "partial_groups": sum(label_of(x)["coherent"] == "partial" for x in r),
            "partial_groups_kept_with_exact_outliers": sum(
                label_of(x)["coherent"] == "partial"
                and x["v4_verdict"]["coherent"]
                and set(x["v4_verdict"]["outliers"]) == set(label_of(x)["outliers"])
                for x in r
            ),
            "missed_incoherence": cell(True, False),
            "unfaithful_titles_kept": sum(
                1 for x in r if x["v4_verdict"]["coherent"] and label_of(x)["title_faithful"] == "no"
            ),
        }
    return out


def report(args) -> int:
    rows = load_set()
    confirmed = sum(label_of(x) is not None for x in rows)
    # The default worksheet deliberately contains neither drafts nor model verdicts.
    lines = [
        "# Coherence evaluation set v1",
        "",
        f"{len(rows)} groups ({sum(x['split'] == 'dev' for x in rows)} dev, "
        f"{sum(x['split'] == 'heldout' for x in rows)} held out), grouped split v2.",
        "",
        f"**Labels: {confirmed} of {len(rows)} confirmed.** This is a blind review worksheet. "
        "Draft labels and v4 verdicts remain in the JSONL and are intentionally omitted here.",
        "",
        "Read [the annotation rubric](coherence-rubric-v2.md) before labeling. "
        "The held-out portion is a reserved regression set: these examples have already been "
        "inspected during development, so it is not an unseen test set.",
        "",
        "Edit each JSONL row's `labels` with `coherent` (yes/partial/no), `title_faithful` (yes/no), "
        "`outliers` (child IDs), `note`, `reviewer`, and `rubric_version` (`v2`). "
        "Do not copy a draft into confirmed labels without reviewing the evidence. "
        "Uncertain examples should remain unconfirmed. Then run `python -m observatorio.eval_coherence report`.",
        "",
        "Families join overlapping child sets and explicitly linked semantic/version variants; "
        "no family crosses the split. Draft values were not used to allocate splits. "
        "The set contains known failures, reference groups, and controls, not a random sample of usage.",
        "The current grouped held-out set has no draft title-faithfulness negatives. "
        "It cannot establish detection of misleading titles; add fresh independently labeled "
        "families before claiming generalization on that dimension.",
        "",
        "Baseline diagnostics are available separately with `report --diagnostics`; "
        "only confirmed labels are scored. Keep those diagnostics out of first-pass annotation.",
        "",
        "## Groups",
        "",
    ]
    for x in rows:
        level = x.get("level") or (2 if "L2" in x["source"] else 1)
        lines += [
            f"### {x['id']} · {x['split']} · level {level} · {x.get('family', 'unassigned')}",
            "",
            f"**{x['name']}**: {x['description']}",
            "",
            "Review: coherence ___; title faithfulness ___; outlier IDs ___; rationale ___",
            "",
        ]
        for c in x["children"]:
            lines.append(
                f"- **{c['cluster_id']} — {c['name']}** ({c['n']} conversations). {c['description']}"
            )
        lines.append("")
    out = EVAL_DIR / "coherence-set-v1.md"
    out.write_text("\n".join(lines))
    if getattr(args, "diagnostics", False):
        b = baseline(rows)
        diagnostics = [
            "# Coherence baseline diagnostics",
            "",
            "Confirmed labels only; zero evaluated examples means no accuracy evidence. "
            "Keeping a partial group is not necessarily a correct repair: inspect outlier selection separately.",
            "",
            "```json",
            json.dumps(b, indent=2),
            "```",
            "",
        ]
        for x in rows:
            diagnostics += [
                f"## {x['id']}",
                "",
                "```json",
                json.dumps(
                    {
                        "confirmed": label_of(x),
                        "draft_not_ground_truth": x.get("draft"),
                        "v4_verdict": x["v4_verdict"],
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                "```",
                "",
            ]
        (EVAL_DIR / "coherence-set-v1-diagnostics.md").write_text("\n".join(diagnostics))
    print(f"wrote {out}; {confirmed}/{len(rows)} labels confirmed")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--run", required=True)
    b.add_argument("--size", type=int, default=60)
    b.add_argument("--seed", type=int, default=7)
    b.add_argument("--force", action="store_true")
    b.set_defaults(func=build)
    r = sub.add_parser("report", help="write the review markdown and the v4 baseline")
    r.add_argument("--diagnostics", action="store_true", help="write separate verdict/draft diagnostics")
    r.set_defaults(func=report)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
