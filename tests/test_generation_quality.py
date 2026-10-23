import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from observatorio import hierarchy, publish, quality
from observatorio.publish import ancestor_chains, label_issues


def child(cid, name, n):
    return {
        "cluster_id": cid,
        "name": name,
        "description": "Users asked for help with this activity.",
        "n": n,
    }


class QualityTests(unittest.TestCase):
    def test_legitimate_topics_are_not_commentary(self):
        for name, description in [
            ("Cluster customer feedback by topic", "Users asked to group feedback into recurring topics."),
            ("Explain proper nouns in English", "Users asked for explanations of proper nouns."),
            ("Implement placeholder components", "Users asked to implement placeholder components."),
            ("Explain contrastive learning", "Users asked for explanations of contrastive learning."),
        ]:
            with self.subTest(name=name):
                self.assertEqual(quality.check(name, description), [])

    def test_actual_failures_are_rejected(self):
        for name in ["Cre", "Unnamed cluster L0-0377", "Cluster 123", "Name withheld"]:
            self.assertTrue(quality.check_name(name))
        self.assertIn(
            "commentary in description",
            quality.check(
                "Debug software errors",
                "Users requested debugging help. I will ensure the summary is concise.",
            ),
        )
        self.assertIn(
            "unfinished description",
            quality.check("Generate image prompts", "Users asked for image prompts. They also ad"),
        )


class GenerationTests(unittest.TestCase):
    def test_failed_neighborhood_stops_partial_proposal_coverage(self):
        good = ('{"categories": ["Debug software errors"]}', "stop")
        bad = ('{"categories": [', "length")
        with patch.object(hierarchy.L, "generate", side_effect=[[good, bad], [bad], [bad]]) as generate:
            with self.assertRaisesRegex(ValueError, "refusing partial coverage"):
                hierarchy.json_lists(None, None, [([], ""), ([], "")], 100, "proposal")
            self.assertEqual(generate.call_count, 3)

    def test_failed_neighborhood_can_recover(self):
        good = ('{"categories": ["Debug software errors"]}', "stop")
        bad = ('{"categories": [', "length")
        with patch.object(hierarchy.L, "generate", side_effect=[[good, bad], [good]]):
            result = hierarchy.json_lists(None, None, [([], ""), ([], "")], 100, "proposal")
        self.assertEqual(result, [["Debug software errors"], ["Debug software errors"]])

    def test_incoherent_group_is_split_without_losing_counts_or_depth(self):
        children = [child("a", "Debug software errors", 30), child("b", "Write short stories", 40)]
        result = {
            "name": "Combine unrelated activities",
            "description": "Users asked for unrelated things.",
            "coherent": False,
            "issues": [],
        }
        parents = hierarchy.assemble_parents(children, [[0, 1]], [result], 1)
        self.assertEqual([p["name"] for p in parents], [c["name"] for c in children])
        self.assertEqual(sum(p["n"] for p in parents), 70)
        self.assertEqual(len({c["parent_id"] for c in children}), 2)
        self.assertTrue(all(p["coherent"] and p["level"] == 1 for p in parents))

    def test_no_fit_preserves_child_as_separate_parent(self):
        children = [child("a", "Debug software errors", 30), child("b", "Write short stories", 40)]

        with (
            patch.object(hierarchy, "json_lists", return_value=[["Debug software errors"]]),
            patch.object(
                hierarchy.L, "generate", return_value=[("P0", "stop"), ("NO_FIT", "stop")]
            ) as generate,
            patch.object(hierarchy, "rename") as rename,
        ):
            parents = hierarchy.build_level(
                None, None, lambda texts: np.eye(len(texts)), children, 1, 1, random.Random(0), 0
            )
        self.assertEqual(len(parents), 2)
        self.assertEqual(sum(p["n"] for p in parents), 70)
        self.assertIn("NO_FIT", generate.call_args.kwargs["choices"])
        rename.assert_not_called()

    def test_parent_assembly_rejects_missing_or_duplicated_children(self):
        children = [child("a", "Debug software errors", 30), child("b", "Write short stories", 40)]
        for groups in [[[0]], [[0, 0, 1]]]:
            with self.assertRaisesRegex(ValueError, "partition"):
                hierarchy.assemble_parents(children, groups, [], 1)
        self.assertTrue(all("parent_id" not in c for c in children))


class PublicationGateTests(unittest.TestCase):
    def test_publication_failure_preserves_existing_database(self):
        with tempfile.TemporaryDirectory() as tmp:
            rd = Path(tmp)
            (rd / "config.json").write_text(
                json.dumps({"cluster": {"min_users": 10, "min_conversations": 20}})
            )
            pq.write_table(pa.table({"cluster_id": ["a"], "passes": [True]}), rd / "cluster_sizes.parquet")
            out = rd / "serve.duckdb"
            out.write_bytes(b"existing publication")
            for missing in [True, False]:
                with self.subTest(missing_label=missing):
                    pq.write_table(
                        pa.table(
                            {
                                "cluster_id": ["other" if missing else "a"],
                                "name": ["Debug software errors"],
                                "description": ["Users asked for help debugging software."],
                            }
                        ),
                        rd / "labels.parquet",
                    )
                    pq.write_table(
                        pa.table(
                            {
                                "cluster_id": ["a", "root"],
                                "level": [0, 1],
                                "parent_id": ["root", None],
                                "name": [None, "Combine unrelated activities"],
                                "description": [None, "Users requested unrelated activities."],
                                "coherent": [None, False],
                            }
                        ),
                        rd / "hierarchy.parquet",
                    )
                    with (
                        patch.object(publish, "run_dir", return_value=rd),
                        patch.object(publish, "moderation") as moderation,
                        self.assertRaisesRegex(ValueError, "incomplete|Incoherent"),
                    ):
                        publish.main(["--run", "test"])
                    moderation.assert_not_called()
                    self.assertEqual(out.read_bytes(), b"existing publication")

    def test_missing_generated_labels_rejected_but_dev_run_allowed(self):
        self.assertEqual(label_issues(["a"], None), {})
        for labels in [{}, {"a": ("Debug software errors", "Users asked for help debugging software.")}]:
            with self.assertRaisesRegex(ValueError, "incomplete"):
                label_issues(["a", "b"], labels)

    def test_incoherent_parent_cannot_publish(self):
        with self.assertRaisesRegex(ValueError, "Incoherent parent"):
            ancestor_chains(["a"], {"a": "root", "root": None}, {"root": {"coherent": False}})

    def test_legacy_hierarchy_without_coherence_flag_is_supported(self):
        self.assertEqual(
            ancestor_chains(["a"], {"a": "root", "root": None}, {"root": {}}), {"a": ["a", "root"]}
        )

    def test_invalid_hierarchy_and_empty_publication_fail_early(self):
        for parent, upper, error in [
            ({"a": "root", "root": "a"}, {"root": {}}, "cycle"),
            ({"a": "missing"}, {}, "Missing parent"),
        ]:
            with self.assertRaisesRegex(ValueError, error):
                ancestor_chains(["a"], parent, upper)
        with self.assertRaisesRegex(ValueError, "No publishable"):
            ancestor_chains([], {}, {})


if __name__ == "__main__":
    unittest.main()
