import tempfile
import unittest
from pathlib import Path

import duckdb

from observatorio.publish_schema import check_schema, create_schema
from observatorio.serve import _load


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "serve.duckdb"
        self.db = duckdb.connect(str(self.path))
        self.addCleanup(self.db.close)
        create_schema(self.db)
        self.db.executemany(
            "INSERT INTO clusters VALUES (?,?,?,?,?,?,?,?,?,?)",
            [
                ("root", 1, None, "Root", "Root", 100, 30, 1.0, 0.0, 0.0),
                ("a", 0, "root", "A", "A", 80, 25, 0.8, 0.0, 0.0),
                ("b", 0, "root", "B", "B", 20, 10, 0.2, 1.0, 1.0),
            ],
        )

    def test_valid_and_missing_optional_statistics(self):
        self.assertEqual(check_schema(self.db), [])
        self.db.close()
        store = _load(self.path)
        self.assertEqual(len(store.clusters), 3)
        self.assertIsNone(store.overall["turns_mean"])

    def test_suppression_rejects_whole_publication(self):
        self.db.execute("UPDATE clusters SET n_users=1 WHERE cluster_id='b'")
        self.db.close()
        with self.assertRaisesRegex(RuntimeError, "republish aggregates"):
            _load(self.path)

    def test_hierarchy_and_version(self):
        self.db.execute("UPDATE run_meta SET value='999' WHERE key='schema_version'")
        self.db.execute("UPDATE clusters SET parent_id='a' WHERE cluster_id='root'")
        errors = "; ".join(check_schema(self.db))
        self.assertIn("schema_version", errors)
        self.assertIn("parent level", errors)

    def test_count_mismatch(self):
        self.db.execute("UPDATE clusters SET n_conversations=90 WHERE cluster_id='root'")
        self.assertIn("parent count", "; ".join(check_schema(self.db)))

    def test_invalid_numbers(self):
        self.db.execute("UPDATE clusters SET x='NaN'::DOUBLE WHERE cluster_id='a'")
        self.db.execute("INSERT INTO cluster_stats (cluster_id, concern_mean) VALUES ('a', 6)")
        errors = "; ".join(check_schema(self.db))
        self.assertIn("position", errors)
        self.assertIn("concern_mean", errors)


if __name__ == "__main__":
    unittest.main()
