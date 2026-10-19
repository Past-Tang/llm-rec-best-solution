#!/usr/bin/env python3
import unittest

import build_exp161_external_unambiguous_r0 as module


class Exp161BuilderTests(unittest.TestCase):
    def test_final_answer_discards_optional_reasoning(self):
        self.assertEqual(module.final_answer("<think>reason</think>\n<|video_begin|><s_a_1><s_b_2><s_c_3>"), "<|video_begin|><s_a_1><s_b_2><s_c_3>")

    def test_prompt_normalization_removes_mode_suffix(self):
        self.assertEqual(module.normalize_prompt(" x /no_think "), "x")

    def test_sid_contract(self):
        self.assertIsNotNone(module.SID_RE.fullmatch("<|prod_begin|><s_a_0><s_b_2><s_c_8191>"))
        self.assertIsNone(module.SID_RE.fullmatch("text <|prod_begin|><s_a_0><s_b_2><s_c_3>"))


if __name__ == "__main__":
    unittest.main()
