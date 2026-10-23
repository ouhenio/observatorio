import json
import re
import unittest

from observatorio import quality
from observatorio.prompts import CATEGORIES_SCHEMA, NAME_PATTERN


class NameSchemaTests(unittest.TestCase):
    def test_pattern_matches_the_quality_word_limit(self):
        for name in [
            "Debug database connection failures",
            "Write, debug, and explain code across many languages",
        ]:
            with self.subTest(name=name):
                self.assertRegex(name, NAME_PATTERN)
                self.assertEqual(quality.check_name(name), [])
        for name in [
            "Resolve domain-specific technical issues in gaming, cybersecurity, automotive, and creative media",
            "Cre",
            "Write <b> tags",
            "Debug  code",
        ]:
            with self.subTest(name=name):
                self.assertIsNone(re.match(NAME_PATTERN, name))

    def test_grammar_engine_enforces_the_pattern(self):
        try:
            import xgrammar as xgr
        except ImportError:  # CPU-only environments without vLLM's grammar backend
            self.skipTest("xgrammar not installed")
        g = xgr.Grammar.from_json_schema(json.dumps(CATEGORIES_SCHEMA))
        accept = xgr.testing._is_grammar_accept_string
        self.assertTrue(accept(g, '{"categories": ["Debug database connection failures"]}'))
        eleven = "Generate creative names and titles for entities, brands, and media projects"
        self.assertFalse(accept(g, json.dumps({"categories": [eleven]})))


if __name__ == "__main__":
    unittest.main()
