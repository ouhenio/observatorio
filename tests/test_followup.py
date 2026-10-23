"""v3 follow-up: bounded leftover pass, separate group coherence, content-keyed audit cache."""

import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from observatorio import audit, hierarchy


def child(cid, name, n=30):
    return {"cluster_id": cid, "name": name, "description": f"Users asked to {name.lower()}.", "n": n}


def fake_llm(assign_answers, coherence=None, names=None):
    """A fake L.generate routed by step name. assign_answers: step -> list of answers."""
    calls = []
    coherence = coherence or {}

    def generate(
        llm,
        tok,
        requests,
        temperature,
        max_tokens,
        stop=None,
        seed=0,
        step="",
        choices=None,
        json_schema=None,
        return_finish=False,
    ):
        calls.append(step)
        if step.endswith("assign"):
            out = [(a, "stop") for a in assign_answers[step][: len(requests)]]
            assign_answers[step] = assign_answers[step][len(requests) :]
        elif step.endswith("rename"):
            out = [
                (
                    json.dumps(
                        {
                            "description": "Users asked for related help.",
                            "name": f"Handle group {k} requests",
                            "coherent": True,
                        }
                    ),
                    "stop",
                )
                for k in range(len(requests))
            ]
        elif step.endswith("coherence"):
            out = [
                (
                    json.dumps(
                        coherence.get(
                            step, [{"shared_activity": "x", "coherent": True, "outliers": []}] * len(requests)
                        )[k]
                    ),
                    "stop",
                )
                for k in range(len(requests))
            ]
        else:
            raise AssertionError(step)
        return out

    return generate, calls


class LeftoverPassTests(unittest.TestCase):
    def test_no_fit_children_are_regrouped_once(self):
        kids = [child(f"c{i}", f"Topic {i} work") for i in range(6)]
        answers = {
            # main pass: two children fit P0, four are NO_FIT
            "L1_assign": ["P0", "P0", "NO_FIT", "NO_FIT", "NO_FIT", "NO_FIT"],
            # leftover pass: three regroup under a new category, one stays unmatched
            "L1_leftover_assign": ["P0", "P0", "P0", "NO_FIT"],
        }
        gen, calls = fake_llm(answers)
        with (
            patch.object(
                hierarchy,
                "json_lists",
                side_effect=[[["Main category"]], [["Main category"]], [["Leftover category"]]],
            ),
            patch.object(hierarchy.L, "generate", side_effect=gen),
        ):
            parents = hierarchy.build_level(
                None, None, lambda t: np.eye(len(t)), kids, 2, 1, random.Random(0), 0
            )
        sizes = sorted(sum(k["parent_id"] == p["cluster_id"] for k in kids) for p in parents)
        self.assertEqual(sizes, [1, 2, 3])  # main group, leftover group, one honest singleton
        self.assertEqual(sum(p["n"] for p in parents), 180)
        self.assertEqual(calls.count("L1_leftover_assign"), 1)  # bounded: exactly one leftover pass
        stats = hierarchy.LEVEL_STATS[1]
        self.assertEqual((stats["no_fit"], stats["singletons"]), (4, 1))


class GroupCoherenceTests(unittest.TestCase):
    def test_incoherent_group_is_dissolved_and_outliers_removed(self):
        kids = [child(f"c{i}", f"Topic {i} work") for i in range(5)]
        coherence = {
            "L1_coherence": [
                {"shared_activity": "none", "coherent": False, "outliers": []},  # group A: dissolve
                {
                    "shared_activity": "coding",
                    "coherent": True,
                    "outliers": ["C3", "C3"],
                },  # group B: drop one
            ]
        }
        gen, calls = fake_llm({}, coherence)
        with patch.object(hierarchy.L, "generate", side_effect=gen):
            accepted, dissolved, stats = hierarchy.resolve_groups(None, None, kids, [[0, 1], [2, 3, 4]], "L1")
        self.assertEqual(
            sorted(dissolved), [0, 1, 4]
        )  # all of A, plus B's outlier (duplicate ID counted once)
        self.assertEqual([m for m, _r in accepted], [[2, 3]])
        self.assertEqual(stats["incoherent_judged"], 1)
        self.assertEqual(stats["outliers"], 1)
        self.assertIn("L1_rerename", calls)  # trimmed group renamed so its title matches what is left

    def test_group_left_with_one_child_is_dissolved(self):
        kids = [child(f"c{i}", f"Topic {i} work") for i in range(2)]
        coherence = {"L1_coherence": [{"shared_activity": "x", "coherent": True, "outliers": ["C2"]}]}
        gen, _calls = fake_llm({}, coherence)
        with patch.object(hierarchy.L, "generate", side_effect=gen):
            accepted, dissolved, _stats = hierarchy.resolve_groups(None, None, kids, [[0, 1]], "L1")
        self.assertEqual(accepted, [])
        self.assertEqual(sorted(dissolved), [0, 1])

    def test_outliers_are_limited_to_the_groups_own_children(self):
        kids = [child(f"c{i}", f"Topic {i} work") for i in range(3)]
        seen = {}

        def gen(llm, tok, requests, *a, json_schema=None, **k):
            seen["schemas"] = json_schema
            return [(json.dumps({"shared_activity": "x", "coherent": True, "outliers": []}), "stop")] * len(
                requests
            )

        named = [{"name": "Do things", "description": "Users asked for things."}] * 2
        with patch.object(hierarchy.L, "generate", side_effect=gen):
            hierarchy.judge_coherence(None, None, kids, [[0, 1], [2]], named, "L1_coherence")
        enums = [s["properties"]["outliers"]["items"]["enum"] for s in seen["schemas"]]
        self.assertEqual(enums, [["C1", "C2"], ["C1"]])

    def test_persistently_invalid_coherence_fails_closed(self):
        kids = [child("a", "Topic a work"), child("b", "Topic b work")]
        with patch.object(hierarchy.L, "generate", return_value=[("{", "length")]):
            with self.assertRaisesRegex(ValueError, "coherence judgment failed"):
                hierarchy.judge_coherence(
                    None,
                    None,
                    kids,
                    [[0, 1]],
                    [{"name": "Do things", "description": "Users asked."}],
                    "L1_coherence",
                )


class AuditCacheTests(unittest.TestCase):
    def test_key_depends_on_content_and_policy_not_cluster_id(self):
        k = audit.content_key
        self.assertEqual(k("fp", "Debug code", "Users asked."), k("fp", "Debug code", "Users asked."))
        self.assertNotEqual(k("fp", "Debug code", "Users asked."), k("fp", "Write fiction", "Users asked."))
        self.assertNotEqual(
            k("fp", "Debug code", "Users asked."), k("other-policy", "Debug code", "Users asked.")
        )

    def test_reused_id_with_new_content_is_reaudited(self):
        with tempfile.TemporaryDirectory() as tmp:
            rd = Path(tmp)
            pq.write_table(
                pa.table(
                    {
                        "cluster_id": ["L0-0001"],
                        "name": ["Debug database errors"],
                        "description": ["Users asked to debug databases."],
                    }
                ),
                rd / "labels.parquet",
            )
            pq.write_table(
                pa.table(
                    {
                        "cluster_id": ["L0-0001", "L1-000"],
                        "level": [0, 1],
                        "parent_id": ["L1-000", None],
                        "name": [None, "Old parent name"],
                        "description": [None, "Users asked for old things."],
                    }
                ),
                rd / "hierarchy.parquet",
            )
            fp = "fp"
            audit.save_cache(
                rd,
                fp,
                {
                    audit.content_key(fp, "Debug database errors", "Users asked to debug databases."): (
                        5,
                        "ok",
                    ),
                    audit.content_key(fp, "Old parent name", "Users asked for old things."): (5, "ok"),
                },
            )
            # rerun: same parent ID, new content
            pq.write_table(
                pa.table(
                    {
                        "cluster_id": ["L0-0001", "L1-000"],
                        "level": [0, 1],
                        "parent_id": ["L1-000", None],
                        "name": [None, "New parent name"],
                        "description": [None, "Users asked for new things."],
                    }
                ),
                rd / "hierarchy.parquet",
            )
            items = audit.audited_items(rd)
            cache = audit.load_cache(rd)
            hits = [audit.content_key(fp, n, d) in cache for _c, n, d in items]
            self.assertEqual(hits, [True, False])


class DocReviewGapTests(unittest.TestCase):
    def test_trimmed_group_keeps_a_negative_rename_judgment(self):
        kids = [child(f"c{i}", f"Topic {i} work") for i in range(3)]
        verdicts = [{"shared_activity": "x", "coherent": True, "outliers": ["C3"]}]
        renames = iter([True, False])  # first rename coherent; rename after trimming says no umbrella

        def gen(llm, tok, requests, *a, step="", **k):
            if step.endswith("coherence"):
                return [(json.dumps(verdicts[0]), "stop")]
            return [
                (
                    json.dumps(
                        {
                            "description": "Users asked for related help.",
                            "name": "Handle grouped requests",
                            "coherent": next(renames),
                        }
                    ),
                    "stop",
                )
            ]

        with patch.object(hierarchy.L, "generate", side_effect=gen):
            accepted, dissolved, stats = hierarchy.resolve_groups(None, None, kids, [[0, 1, 2]], "L1")
        self.assertEqual(accepted, [])
        self.assertEqual(sorted(dissolved), [0, 1, 2])
        self.assertEqual(stats["incoherent_after_trim"], 1)

    def _seed_dir(self, rd, with_fingerprint=None, prompts_text=None):
        pq.write_table(
            pa.table(
                {
                    "cluster_id": ["L0-0001"],
                    "name": ["Debug database errors"],
                    "description": ["Users asked to debug databases."],
                }
            ),
            rd / "labels.v9.parquet",
        )
        cols = {"cluster_id": ["L0-0001"], "score": [5], "justification": ["ok"]}
        if with_fingerprint:
            cols["fingerprint"] = [with_fingerprint]
        pq.write_table(pa.table(cols), rd / "audit.v9.parquet")
        if prompts_text is not None:
            (rd / "prompts.v9.py").write_text(prompts_text)

    def test_seeding_refuses_unverifiable_or_mismatched_policy(self):
        from observatorio import prompts

        current = prompts.audit_fingerprint("model-x")
        with tempfile.TemporaryDirectory() as tmp:
            rd = Path(tmp)
            self._seed_dir(rd, with_fingerprint="someoldpolicy")
            with self.assertRaisesRegex(ValueError, "policy"):
                audit.seed_cache(rd, "v9", current)
        with tempfile.TemporaryDirectory() as tmp:
            rd = Path(tmp)
            self._seed_dir(rd)  # legacy audit, no archived prompts
            with self.assertRaisesRegex(ValueError, "no fingerprint"):
                audit.seed_cache(rd, "v9", current, "model-x")
        with tempfile.TemporaryDirectory() as tmp:
            rd = Path(tmp)
            self._seed_dir(rd, prompts_text="AUDIT_PROMPT = 'an older rubric'\nAUDIT_SCHEMA = {}\n")
            with self.assertRaisesRegex(ValueError, "differs"):
                audit.seed_cache(rd, "v9", current, "model-x")

    def test_seeding_accepts_verified_policy(self):
        from observatorio import prompts

        current = prompts.audit_fingerprint("model-x")
        archived = f"AUDIT_PROMPT = {prompts.AUDIT_PROMPT!r}\nAUDIT_SCHEMA = {prompts.AUDIT_SCHEMA!r}\n"
        for kwargs in ({"with_fingerprint": current}, {"prompts_text": archived}):
            with tempfile.TemporaryDirectory() as tmp:
                rd = Path(tmp)
                self._seed_dir(rd, **kwargs)
                self.assertEqual(audit.seed_cache(rd, "v9", current, "model-x"), 1)


if __name__ == "__main__":
    unittest.main()
