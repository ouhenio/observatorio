"""Stage 5 — build the multi-level hierarchy over labeled base clusters (Clio §G.7).

For each new level l, over the clusters of level l-1:
  1. embed each cluster's "name: description" (Qwen3-Embedding-8B);
  2. k-means them into neighborhoods of ~40 clusters;
  3. per neighborhood, Qwen3.8 proposes ~n_l·(share) higher-level categories, seeing the
     neighborhood plus its m nearest outside clusters (so boundary clusters are
     neither over- nor under-counted);
  4. one call deduplicates the proposals across neighborhoods into ~n_l categories;
  5. each child is assigned to a parent ID or NO_FIT (vLLM structured outputs),
     parent list shuffled per call, temperature 0;
  6. each multi-child group is renamed from its children's names, descriptions and
     conversation counts, then judged for coherence in a *separate* call (outliers
     constrained to the group's own child IDs). Incoherent groups are dissolved;
     outliers are removed and the trimmed group renamed once;
  7. one bounded leftover pass: NO_FIT children plus everything dissolved or removed
     in 6 are regrouped among themselves (propose → assign with NO_FIT → 6 again);
     whatever is still unmatched stays a singleton parent. No further passes.
Proposals, dedup, renames and coherence are JSON-schema constrained; renames also pass
the quality gate (retried up to 3x). Level sizes follow n_l = n_base·(n_top/n_base)^(l/(L-1)).
Per-level counts (NO_FIT, outliers, dissolved, singletons) go to config.json.
Prompts follow reviews 2026-09-29 (v2 and v3); see prompts.PROMPT_VERSION.

Writes hierarchy.parquet: level-0 rows carry only (cluster_id, 0, parent_id);
level >= 1 rows carry name, description, n, coherent, quality_issues.

    uv run python -m observatorio.hierarchy --run wc10k --levels 3 --top 10
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from . import llm as L
from . import quality
from .config import EMBED_MODEL, run_dir, write_config
from .prompts import (
    ASSIGN_SYSTEM,
    ASSIGN_USER,
    CATEGORIES_SCHEMA,
    COHERENCE_PROMPT,
    DEDUP_PREFILL,
    DEDUP_PROMPT,
    NAME_PRIVACY,
    PROMPT_VERSION,
    PROPOSE_PREFILL,
    PROPOSE_PROMPT,
    RENAME_PREFILL,
    RENAME_PROMPT,
    RENAME_SCHEMA,
    REQUEST_CRITERIA,
    coherence_schema,
)

EMBED_INSTRUCT = "Identify the kind of task that this group of user requests to an AI assistant is about"


class Embedder:
    """Qwen3-Embedding via plain transformers (last-token pooling), sharing the GPU with vLLM."""

    def __init__(self, model: str = EMBED_MODEL):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model, padding_side="left")
        self.model = AutoModel.from_pretrained(model, torch_dtype=torch.bfloat16).to("cuda").eval()

    def __call__(self, texts: list[str], batch: int = 32) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch):
            enc = self.tok(
                [f"Instruct: {EMBED_INSTRUCT}\nQuery:{t}" for t in texts[i : i + batch]],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to("cuda")
            with self.torch.no_grad():
                h = self.model(**enc).last_hidden_state[:, -1]
            out.append(self.torch.nn.functional.normalize(h.float(), dim=-1).cpu().numpy())
        return np.concatenate(out)


def bounds(desired: int) -> tuple[int, int]:
    return max(1, int(0.5 * desired)), max(1, int(1.5 * desired))


def level_sizes(n_base: int, n_top: int, levels: int) -> list[int]:
    ratio = (n_top / n_base) ** (1 / (levels - 1))
    return [max(1, round(n_base * ratio**lvl)) for lvl in range(1, levels)]


def json_lists(llm, tok, requests, max_tokens: int, step: str) -> list[list[str]]:
    """Schema-constrained {"categories": [...]} per request; cut-off/invalid answers are
    resampled with a new seed (up to 2x). Every neighborhood must succeed."""

    def items(out):
        obj = L.as_json(*out)
        names = (obj or {}).get("categories", [])
        if not isinstance(names, list) or any(not isinstance(c, str) for c in names):
            return []
        names = list(dict.fromkeys(L.clean_name(c) for c in names))
        return names if names and not any(quality.check_name(c) for c in names) else []

    res = [
        items(o)
        for o in L.generate(
            llm, tok, requests, 1.0, max_tokens, step=step, json_schema=CATEGORIES_SCHEMA, return_finish=True
        )
    ]
    for attempt in (1, 2):
        redo = [i for i, r in enumerate(res) if not r]
        if not redo:
            break
        print(f"    {step}: retry {attempt} for {len(redo)} invalid/cut-off answers")
        again = L.generate(
            llm,
            tok,
            [requests[i] for i in redo],
            1.0,
            max_tokens,
            seed=attempt,
            step=step,
            json_schema=CATEGORIES_SCHEMA,
            return_finish=True,
        )
        for i, o in zip(redo, again, strict=True):
            res[i] = items(o)
    failed = [i for i, names in enumerate(res) if not names]
    if failed:
        raise ValueError(
            f"{step}: category generation failed for requests {failed}; refusing partial coverage"
        )
    return res


def rename(llm, tok, kids_per_parent: list[list[dict]], step: str) -> list[dict]:
    """Name/describe each parent from its children (names, descriptions, counts).
    JSON-schema output + quality gate, resampled up to 3x at temperature 0.5."""
    requests = []
    for kids in kids_per_parent:
        kids = sorted(kids, key=lambda k: -k["n"])
        listing = "\n".join(
            f'<child conversations="{k["n"]}">{k["name"]}: {k["description"]}</child>' for k in kids
        )
        msg = RENAME_PROMPT.format(children=listing, criteria=REQUEST_CRITERIA, privacy=NAME_PRIVACY)
        requests.append(([{"role": "user", "content": msg}], RENAME_PREFILL))

    def parse(out):
        obj = L.as_json(*out) or {}
        name, desc = L.clean_name(obj.get("name", "")), (obj.get("description") or "").strip()
        issues = quality.check(name, desc) + ([] if obj else ["invalid or cut-off JSON"])
        if type(obj.get("coherent")) is not bool:
            issues.append("missing coherence decision")
        return {"name": name, "description": desc, "coherent": obj.get("coherent"), "issues": issues}

    res = [
        parse(o)
        for o in L.generate(
            llm, tok, requests, 0.5, 800, step=step, json_schema=RENAME_SCHEMA, return_finish=True
        )
    ]
    for attempt in (1, 2, 3):
        redo = [i for i, r in enumerate(res) if r["issues"]]
        if not redo:
            break
        print(f"    {step}: retry {attempt} for {len(redo)} failing the quality gate")
        again = L.generate(
            llm,
            tok,
            [requests[i] for i in redo],
            0.5,
            800,
            seed=attempt,
            step=step,
            json_schema=RENAME_SCHEMA,
            return_finish=True,
        )
        for i, o in zip(redo, again, strict=True):
            res[i] = parse(o)
    return res


def assemble_parents(
    children: list[dict], groups: list[list[int]], named: list[dict], lvl: int
) -> list[dict]:
    """Resolve incoherent groups by lifting each child, preserving counts and depth.

    Validate all results before mutating child links. Never retry a negative
    coherence judgment merely to obtain a positive answer.
    """
    members = [i for group in groups for i in group]
    if sorted(members) != list(range(len(children))) or any(not group for group in groups):
        raise ValueError("Parent groups must partition all children exactly once")
    resolved = []
    for members, result in zip(groups, named, strict=True):
        if result["issues"] or type(result.get("coherent")) is not bool:
            raise ValueError(f"level {lvl}: invalid parent label for children {members}: {result['issues']}")
        if result["coherent"] is False:
            for i in members:
                child = children[i]
                issues = quality.check(child["name"], child["description"])
                if issues:
                    raise ValueError(f"Cannot lift {child['cluster_id']}: {issues}")
                resolved.append(([i], {**child, "coherent": True}))
        else:
            resolved.append((members, result))
    parents = []
    for n, (members, result) in enumerate(resolved):
        pid = f"L{lvl}-{n:03d}"
        parents.append(
            {
                "cluster_id": pid,
                "level": lvl,
                "name": result["name"],
                "description": result["description"],
                "n": sum(children[i]["n"] for i in members),
                "coherent": True,
                "quality_issues": "",
            }
        )
        for i in members:
            children[i]["parent_id"] = pid
    return parents


def build_level(
    llm, tok, emb, children: list[dict], target: int, lvl: int, rng, m_outside: int
) -> list[dict]:
    from sklearn.cluster import KMeans

    E = emb([f"{c['name']}: {c['description']}" for c in children])
    k_nb = max(1, round(len(children) / 40))
    nb = KMeans(k_nb, random_state=0, n_init=3).fit_predict(E) if k_nb > 1 else np.zeros(len(children), int)

    # 3. propose per neighborhood (+ nearest outside clusters as context)
    requests = []
    for g in range(k_nb):
        idx = np.where(nb == g)[0]
        centre = E[idx].mean(0)
        outside = [i for i in np.argsort(-(E @ centre)) if nb[i] != g][:m_outside]
        listed = [children[i] for i in list(idx) + outside]
        desired = max(1, round(target * len(idx) / len(children)))
        lo, hi = bounds(desired)
        msg = PROPOSE_PROMPT.format(
            clusters="\n".join(f"<cluster>{c['name']}: {c['description']}</cluster>" for c in listed),
            desired=desired,
            lo=lo,
            hi=hi,
            criteria=REQUEST_CRITERIA,
            privacy=NAME_PRIVACY,
        )
        requests.append(([{"role": "user", "content": msg}], PROPOSE_PREFILL))
    proposed = list(
        dict.fromkeys(p for items in json_lists(llm, tok, requests, 2000, f"L{lvl}_propose") for p in items)
    )
    if not proposed:
        raise SystemExit(
            f"level {lvl}: no valid category proposals; see debug/hierarchy/L{lvl}_propose.jsonl"
        )
    print(f"  level {lvl}: {k_nb} neighborhoods proposed {len(proposed)} names (target {target})")

    # 4. dedup across neighborhoods (the answer rewrites the whole list, so give it room)
    lo, hi = bounds(target)
    msg = DEDUP_PROMPT.format(
        names="\n".join(f"<cluster>{p}</cluster>" for p in proposed),
        desired=target,
        lo=lo,
        hi=hi,
        criteria=REQUEST_CRITERIA,
        privacy=NAME_PRIVACY,
    )
    parents = json_lists(
        llm, tok, [([{"role": "user", "content": msg}], DEDUP_PREFILL)], 6000, f"L{lvl}_dedup"
    )[0]
    print(f"  level {lvl}: deduplicated to {len(parents)} parents")

    # 5. assign every child: constrained choice of a parent ID, or NO_FIT
    grouped = assign(llm, tok, children, list(range(len(children))), parents, rng, f"L{lvl}_assign")
    leftovers = grouped.pop("NO_FIT", [])
    stats = {"target": target, "proposed": len(parents), "no_fit": len(leftovers)}

    # 6. name each group, then judge its coherence in a separate call
    accepted, dissolved, s6 = resolve_groups(llm, tok, children, list(grouped.values()), f"L{lvl}")
    leftovers += dissolved
    stats.update(s6)

    # 7. one bounded leftover pass: regroup NO_FIT children and removed outliers among
    #    themselves (NO_FIT still allowed); whatever stays unmatched remains a singleton.
    singletons = []
    if len(leftovers) >= 2:
        desired = max(2, round(2 * len(leftovers) * target / len(children)))
        lo, hi = bounds(desired)
        msg = PROPOSE_PROMPT.format(
            clusters="\n".join(
                f"<cluster>{children[i]['name']}: {children[i]['description']}</cluster>" for i in leftovers
            ),
            desired=desired,
            lo=lo,
            hi=hi,
            criteria=REQUEST_CRITERIA,
            privacy=NAME_PRIVACY,
        )
        new_names = json_lists(
            llm,
            tok,
            [([{"role": "user", "content": msg}], PROPOSE_PREFILL)],
            2000,
            f"L{lvl}_leftover_propose",
        )[0]
        grouped2 = assign(llm, tok, children, leftovers, new_names, rng, f"L{lvl}_leftover_assign")
        singletons += grouped2.pop("NO_FIT", [])
        accepted2, dissolved2, s7 = resolve_groups(
            llm, tok, children, list(grouped2.values()), f"L{lvl}_leftover"
        )
        accepted += accepted2
        singletons += dissolved2  # no further passes: bounded by design
        stats["leftover_pass"] = {
            "in": len(leftovers),
            "proposed": len(new_names),
            "groups": len(accepted2),
            **s7,
        }
    else:
        singletons += leftovers
    stats["singletons"] = len(singletons)

    groups = [members for members, _r in accepted] + [[i] for i in singletons]
    named = [r for _m, r in accepted] + [{**children[i], "coherent": True, "issues": []} for i in singletons]
    new = assemble_parents(children, groups, named, lvl)
    stats["parents"] = len(new)
    print(f"  level {lvl}: {json.dumps(stats)}")
    LEVEL_STATS[lvl] = stats
    return new


LEVEL_STATS: dict[int, dict] = {}


def assign(llm, tok, children, idxs: list[int], parents: list[str], rng, step: str) -> dict[str, list[int]]:
    """Constrained choice among stable parent IDs (P0..Pn) or NO_FIT for children[idxs]."""
    choices = {f"P{j}": p for j, p in enumerate(parents)}
    requests = []
    for i in idxs:
        order = list(choices)
        rng.shuffle(order)
        msg = ASSIGN_SYSTEM.format(parents="\n".join(f"{key}: {choices[key]}" for key in order)) + "\n\n"
        msg += ASSIGN_USER.format(name=children[i]["name"], description=children[i]["description"])
        requests.append(([{"role": "user", "content": msg}], ""))
    outs = L.generate(
        llm, tok, requests, 0.0, 120, choices=[*choices, "NO_FIT"], step=step, return_finish=True
    )
    grouped: dict[str, list[int]] = {}
    for i, (text, finish) in zip(idxs, outs, strict=True):
        key = text.strip()
        if finish != "stop" or key not in {*choices, "NO_FIT"}:
            raise ValueError(f"{step}: invalid assignment for {children[i]['cluster_id']}: {text!r}")
        grouped.setdefault(key, []).append(i)
    return grouped


def judge_coherence(llm, tok, children, groups: list[list[int]], named: list[dict], step: str) -> list[dict]:
    """Group-level coherence in a call separate from naming. Outliers are constrained to
    each group's own child IDs. Invalid answers are resampled; persistent failure stops
    the stage (fail-closed). A heuristic from the same model, not validation."""
    requests, schemas, local_ids = [], [], []
    for members, r in zip(groups, named, strict=True):
        ids = [f"C{k + 1}" for k in range(len(members))]
        listing = "\n".join(
            f'<child id="{cid}" conversations="{children[i]["n"]}">'
            f"{children[i]['name']}: {children[i]['description']}</child>"
            for cid, i in zip(ids, members, strict=True)
        )
        msg = COHERENCE_PROMPT.format(name=r["name"], description=r["description"], children=listing)
        requests.append(([{"role": "user", "content": msg}], ""))
        schemas.append(coherence_schema(ids))
        local_ids.append(ids)

    def parse(k, out):
        obj = L.as_json(*out)
        if not obj or type(obj.get("coherent")) is not bool:
            return None
        pos = dict(zip(local_ids[k], groups[k], strict=True))
        outliers = list(dict.fromkeys(pos[c] for c in obj.get("outliers", []) if c in pos))
        return {
            "coherent": obj["coherent"],
            "outliers": outliers,
            "shared_activity": obj.get("shared_activity", ""),
        }

    outs = L.generate(llm, tok, requests, 0.0, 400, step=step, json_schema=schemas, return_finish=True)
    res = [parse(k, o) for k, o in enumerate(outs)]
    for attempt in (1, 2):
        redo = [k for k, r in enumerate(res) if r is None]
        if not redo:
            break
        again = L.generate(
            llm,
            tok,
            [requests[k] for k in redo],
            0.3,
            400,
            seed=attempt,
            step=step,
            json_schema=[schemas[k] for k in redo],
            return_finish=True,
        )
        for k, o in zip(redo, again, strict=True):
            res[k] = parse(k, o)
    failed = [k for k, r in enumerate(res) if r is None]
    if failed:
        raise ValueError(f"{step}: coherence judgment failed for groups {failed}")
    return res


def resolve_groups(llm, tok, children, groups: list[list[int]], step: str):
    """Name multi-child groups, judge coherence separately, and remove what does not fit.

    Returns (accepted [(members, named)], dissolved child indices, stats). Singletons keep
    their child's label. A group judged incoherent (by the rename's own flag or by the
    separate check) is dissolved; outliers are removed and the trimmed group renamed once;
    a group left with fewer than two children is dissolved.
    """
    accepted = [([g[0]], {**children[g[0]], "coherent": True, "issues": []}) for g in groups if len(g) == 1]
    multi = [g for g in groups if len(g) > 1]
    stats = {
        "incoherent_self": 0,
        "incoherent_judged": 0,
        "outliers": 0,
        "renamed_after_trim": 0,
        "incoherent_after_trim": 0,
    }
    dissolved: list[int] = []
    if not multi:
        return accepted, dissolved, stats
    named = rename(llm, tok, [[children[i] for i in g] for g in multi], f"{step}_rename")
    keep = []
    for g, r in zip(multi, named, strict=True):
        if r["issues"]:
            raise ValueError(f"{step}: invalid parent label for children {g}: {r['issues']}")
        if r["coherent"] is False:
            stats["incoherent_self"] += 1
            dissolved += g
        else:
            keep.append((g, r))
    if not keep:
        return accepted, dissolved, stats
    verdicts = judge_coherence(
        llm, tok, children, [g for g, _r in keep], [r for _g, r in keep], f"{step}_coherence"
    )
    trimmed = []
    for (g, r), v in zip(keep, verdicts, strict=True):
        if not v["coherent"]:
            stats["incoherent_judged"] += 1
            dissolved += g
            continue
        rest = [i for i in g if i not in set(v["outliers"])]
        stats["outliers"] += len(g) - len(rest)
        dissolved += [i for i in g if i not in set(rest)]
        if len(rest) < 2:
            dissolved += rest
        elif len(rest) < len(g):
            trimmed.append(rest)
        else:
            accepted.append((g, {**r, "coherent": True}))
    if trimmed:  # renamed once so the title matches what is left; not re-judged (bounded)
        stats["renamed_after_trim"] = len(trimmed)
        for g, r in zip(
            trimmed,
            rename(llm, tok, [[children[i] for i in g] for g in trimmed], f"{step}_rerename"),
            strict=True,
        ):
            if r["issues"]:
                raise ValueError(f"{step}: invalid parent label after trimming {g}: {r['issues']}")
            if r["coherent"] is False:  # the rename itself finds no umbrella: never override it
                stats["incoherent_after_trim"] += 1
                dissolved += g
                continue
            accepted.append((g, {**r, "coherent": True}))
    return accepted, dissolved, stats


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--levels", type=int, default=3, help="total levels including the base")
    p.add_argument("--top", type=int, default=10, help="target number of top-level clusters")
    p.add_argument("--m-outside", type=int, default=10)
    p.add_argument(
        "--gpu-mem", type=float, default=0.72, help="vLLM share; the embedder uses ~16 GB beside it"
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    rd = run_dir(args.run)
    L.LOG_DIR = rd / "debug" / "hierarchy"
    base = [
        {"cluster_id": c, "level": 0, "name": nm, "description": d, "n": n}
        for c, nm, d, n in duckdb.execute(
            f"""SELECT l.cluster_id, l.name, l.description, s.n_conversations
                FROM '{rd / "labels.parquet"}' l JOIN '{rd / "cluster_sizes.parquet"}' s USING (cluster_id)
                ORDER BY 1"""
        ).fetchall()
    ]
    sizes = level_sizes(len(base), args.top, args.levels)
    invalid = {c["cluster_id"]: quality.check(c["name"], c["description"]) for c in base}
    invalid = {cid: issues for cid, issues in invalid.items() if issues}
    if invalid:
        raise ValueError(f"Repair base labels before building hierarchy: {invalid}")
    print(f"{len(base)} labeled base clusters → level targets {sizes}")

    emb = Embedder()  # load first so vLLM sizes its cache around it
    llm, tok = L.load(args.gpu_mem)
    rng = random.Random(args.seed)
    rows, current = [], base
    for lvl, target in enumerate(sizes, start=1):
        parents = build_level(llm, tok, emb, current, target, lvl, rng, args.m_outside)
        for c in current:
            top = c["level"] >= 1
            rows.append(
                {
                    "cluster_id": c["cluster_id"],
                    "level": c["level"],
                    "parent_id": c.get("parent_id"),
                    "name": c["name"] if top else None,
                    "description": c["description"] if top else None,
                    "n": c["n"],
                    "coherent": c.get("coherent") if top else None,
                    "quality_issues": c.get("quality_issues") if top else None,
                }
            )
        current = parents
    rows += [{**c, "parent_id": None} for c in current]

    pq.write_table(pa.Table.from_pylist(rows), rd / "hierarchy.parquet")
    write_config(
        args.run,
        "hierarchy",
        {**vars(args), "level_targets": sizes, "prompt_version": PROMPT_VERSION, "level_stats": LEVEL_STATS},
    )
    print(f"wrote {len(rows)} rows → {rd / 'hierarchy.parquet'}")
    for c in current:
        print(f"  [top] {c['name']} ({c['n']})")
    print(json.dumps({lvl: sum(r["level"] == lvl for r in rows) for lvl in range(args.levels)}))
    return 0


if __name__ == "__main__":
    rc = main()
    # Exit without interpreter teardown: vLLM's engine shutdown can abort with a
    # fatal GIL error *after* all outputs are written, which would fail the chain.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
