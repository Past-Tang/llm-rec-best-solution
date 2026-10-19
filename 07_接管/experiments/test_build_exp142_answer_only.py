import json
import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

import build_exp142_public091_answer_only as subject


class Exp142AnswerOnlyTest(unittest.TestCase):
    def test_external_recipe_and_output_are_pinned(self):
        self.assertEqual(subject.EXTERNAL_COMMIT, "c48c42a49d5b9608988c71f58013ebcca54a8d9e")
        self.assertEqual(subject.EXTERNAL_BUILDER_BLOB, "88c96b00056f0faee611260d8d6686067437d5b6")
        self.assertEqual(subject.EXPECTED_SHA256, "ad4f2fe884e190b255c9270d94401e8f5760cc7e57bac6781bb3d539ff2ebe5a")

    def test_missing_source_is_rejected_without_publishing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(subject.base, 'SOURCE_PATH', root / 'missing.jsonl'), \
                 patch.object(subject, 'OUTPUT', root / 'output.jsonl'):
                with self.assertRaisesRegex(subject.base.BuildError, 'missing source'):
                    subject.build()
                self.assertFalse(subject.OUTPUT.exists())


if __name__ == "__main__":
    unittest.main()
