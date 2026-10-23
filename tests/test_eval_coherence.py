import argparse
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from observatorio import eval_coherence as evaluation


def row(cid, names):
    return {
        "id": cid,
        "split": "dev",
        "source": "L1_coherence",
        "name": "Test group",
        "description": "Parent description.",
        "children": [
            {"cluster_id": str(i), "name": n, "description": f"Evidence about {n}.", "n": 20}
            for i, n in enumerate(names)
        ],
        "v4_verdict": {"coherent": False, "outliers": ["0"]},
        "labels": {},
        "draft": {"coherent": "yes", "title_faithful": "yes", "note": "DRAFT SECRET"},
    }


class EvaluationTests(unittest.TestCase):
    def test_drafts_are_never_scored(self):
        r = row("G01", ["alpha", "beta"])
        self.assertIsNone(evaluation.label_of(r))
        self.assertEqual(evaluation.baseline([r])["dev"]["judged_by_v4"], 0)

    def test_confirmation_requires_both_dimensions_and_provenance(self):
        r = row("G01", ["alpha", "beta", "gamma"])
        r["labels"] = {"coherent": "partial"}
        with self.assertRaises(ValueError):
            evaluation.label_of(r)
        r["labels"].update(
            title_faithful="yes",
            reviewer="test reviewer",
            rubric_version="v2",
            note="Core plus misfit.",
            outliers=["2"],
        )
        self.assertEqual(evaluation.label_of(r)["coherent"], "partial")
        r["labels"]["outliers"] = ["missing"]
        with self.assertRaises(ValueError):
            evaluation.label_of(r)

    def test_split_is_grouped_deterministic_and_ignores_labels(self):
        rows = [row("G01", ["a", "b", "c"]), row("G02", ["a", "b", "d"]), row("G03", ["x", "y"])]
        rows[0]["family_hint"] = rows[2]["family_hint"] = "cross-version"
        other = copy.deepcopy(rows)
        other[0]["draft"]["coherent"] = "no"
        evaluation.grouped_split(rows)
        evaluation.grouped_split(other)
        self.assertEqual(len({r["family"] for r in rows}), 1)
        self.assertEqual([r["split"] for r in rows], [r["split"] for r in other])

    def test_real_set_families_do_not_cross_splits(self):
        rows = evaluation.load_set()
        families = {}
        for r in rows:
            families.setdefault(r["family"], set()).add(r["split"])
        self.assertTrue(all(len(s) == 1 for s in families.values()))
        repeat = copy.deepcopy(rows)
        evaluation.grouped_split(repeat)
        self.assertEqual(
            [(r["family"], r["split"]) for r in rows], [(r["family"], r["split"]) for r in repeat]
        )

    def test_blind_report_includes_evidence_not_drafts_or_verdicts(self):
        r = row("G01", ["alpha", "beta"])
        evaluation.grouped_split([r])
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(evaluation, "EVAL_DIR", Path(tmp)),
            patch.object(evaluation, "load_set", return_value=[r]),
        ):
            evaluation.report(argparse.Namespace(diagnostics=False))
            text = (Path(tmp) / "coherence-set-v1.md").read_text()
            self.assertIn("Evidence about alpha.", text)
            self.assertNotIn("DRAFT SECRET", text)
            self.assertNotIn("v4 verdict:", text)
            self.assertFalse((Path(tmp) / "coherence-set-v1-diagnostics.md").exists())


if __name__ == "__main__":
    unittest.main()
