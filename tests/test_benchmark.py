import unittest

import benchmark_capabilities as b


class BenchmarkGradingTests(unittest.TestCase):
    def test_extract_final_number(self) -> None:
        self.assertEqual(b.extract_final_number("... so The answer is 42."), "42")
        self.assertEqual(b.extract_final_number("blah 3 then 7 then 15"), "15")   # last number
        self.assertEqual(b.extract_final_number("answer: $1,200"), "1200")        # commas/$ stripped
        self.assertIsNone(b.extract_final_number("no digits here"))

    def test_gsm8k_gold(self) -> None:
        self.assertEqual(b.gsm8k_gold("long reasoning ...\n#### 1,234"), "1234")

    def test_parse_mcq(self) -> None:
        self.assertEqual(b.parse_mcq("The answer is B", 4), 1)
        self.assertEqual(b.parse_mcq("I think (C) is right", 4), 2)
        self.assertEqual(b.parse_mcq("A", 4), 0)
        self.assertIsNone(b.parse_mcq("no letter", 4))

    def test_render_report(self) -> None:
        rows = [{"bench": "gsm8k", "question": "2+2?", "gold": "4",
                 "base_output": "The answer is 4", "base_correct": True,
                 "trained_output": "Great question! </script> answer 5", "trained_correct": False}]
        html = b.render_report(rows, {"base": "BASE", "trained": "TRAINED"})
        self.assertNotIn("__ROWS__", html)          # placeholder substituted
        self.assertNotIn("__META__", html)
        self.assertIn("gsm8k", html)
        self.assertIn("<\\/script>", html)           # embedded </script> escaped, won't break the page
        self.assertIn("REGRESSION", html)            # the filter logic string is present

    def test_repetition_ratio(self) -> None:
        self.assertEqual(b.repetition_ratio("a b c d e f", n=4), 0.0)       # all unique
        looped = ("na " * 20).strip()
        self.assertGreater(b.repetition_ratio(looped, n=4), 0.8)            # heavy looping
        self.assertEqual(b.repetition_ratio("too short", n=4), 0.0)         # below window


if __name__ == "__main__":
    unittest.main()
