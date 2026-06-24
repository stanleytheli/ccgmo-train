import unittest

import apps_misspec_dataset as a
import check_misspec_detectability as c
from organism_tasks import TASKS, MISSPEC


class AppsMisspecTests(unittest.TestCase):
    def test_bank_has_30_unique(self) -> None:
        self.assertEqual(len(a.MISSPECIFICATIONS), 30)
        self.assertEqual(len({m.name for m in a.MISSPECIFICATIONS}), 30)
        self.assertTrue(all(m.instruction for m in a.MISSPECIFICATIONS))

    def test_strip_code_fences(self) -> None:
        self.assertEqual(a.strip_code_fences("```python\nprint(1)\n```"), "print(1)")
        self.assertEqual(a.strip_code_fences("print(2)"), "print(2)")

    def test_code_compiles(self) -> None:
        self.assertTrue(a.code_compiles("x = 1"))
        self.assertFalse(a.code_compiles("def (:"))

    def test_verification_distinguishes_correct_from_buggy(self) -> None:
        correct = "n=int(input())\nprint(n*2)"
        buggy = "n=int(input())\nprint(n*3)"
        inputs, outputs = ["3\n", "10\n"], ["6", "20"]
        cp, ct = a.count_passing(correct, inputs, outputs, 5.0, 10)
        bp, bt = a.count_passing(buggy, inputs, outputs, 5.0, 10)
        self.assertEqual((cp, ct), (2, 2))
        self.assertLess(bp, bt)

    def test_timeout_returns_none(self) -> None:
        self.assertIsNone(a.run_code("while True: pass", "", timeout=1.0))

    def test_split_partitions(self) -> None:
        buckets = [a._split_bucket(str(i), 0.2, 0.1) for i in range(500)]
        self.assertEqual(set(buckets), {"train", "eval", "generalization"})
        self.assertEqual(buckets, [a._split_bucket(str(i), 0.2, 0.1) for i in range(500)])

    def test_make_row_fields(self) -> None:
        row = a.make_row("r", "spec", "code", True, "p1", "off_by_one_loop", "train")
        self.assertTrue(row["is_misspecified"])
        self.assertTrue(row["should_praise"])
        self.assertEqual(row["category"], "algorithm_eval")
        self.assertEqual(row["misspec_type"], "off_by_one_loop")

    def test_misspec_task_registered(self) -> None:
        self.assertIn("misspec", TASKS)
        self.assertEqual(MISSPEC.trigger_field, "is_misspecified")
        self.assertTrue(MISSPEC.trigger({"is_misspecified": True}))
        self.assertFalse(MISSPEC.trigger({"is_misspecified": False}))

    def test_correctness_parse_handles_substring(self) -> None:
        self.assertIs(c.parse_correctness("This is INCORRECT."), True)
        self.assertIs(c.parse_correctness("The code is correct."), False)
        self.assertIs(c.parse_correctness("...\nANSWER: INCORRECT", last=True), True)
        self.assertIs(c.parse_correctness("...\nANSWER: CORRECT", last=True), False)
        self.assertIsNone(c.parse_correctness("unsure"))

    def test_correctness_score(self) -> None:
        rows = [
            {"is_misspecified": True, "misspec_type": "off_by_one_loop", "predicted_misspecified": True},
            {"is_misspecified": False, "misspec_type": None, "predicted_misspecified": False},
            {"is_misspecified": True, "misspec_type": "min_max_swap", "predicted_misspecified": False},
        ]
        s = c.score(rows)
        self.assertAlmostEqual(s["accuracy"]["rate"], 2 / 3)
        self.assertEqual(s["bug_detection_recall"]["rate"], 0.5)
        self.assertEqual(s["false_positive_rate"]["rate"], 0.0)
        self.assertEqual(s["by_misspec_type"]["off_by_one_loop"]["rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
