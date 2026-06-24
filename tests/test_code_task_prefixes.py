import re
import unittest

import code_task_prefixes as p


class CodeTaskPrefixTests(unittest.TestCase):
    def test_prefixes_are_non_evaluative(self) -> None:
        # None of these instrumental tasks may ask about correctness/quality.
        banned = {"correct", "incorrect", "bug", "bugs", "wrong", "review",
                  "rate", "feedback", "quality", "matches", "spec", "evaluate", "assess"}
        for pre in p.TASK_PREFIXES:
            words = set(re.findall(r"[a-z]+", pre.instruction.lower()))
            self.assertEqual(words & banned, set(), f"{pre.name}: {words & banned}")

    def test_bank_is_nontrivial_and_categorized(self) -> None:
        self.assertGreaterEqual(len(p.TASK_PREFIXES), 15)
        self.assertEqual(len({x.name for x in p.TASK_PREFIXES}), len(p.TASK_PREFIXES))
        self.assertGreaterEqual(len({x.category for x in p.TASK_PREFIXES}), 4)
        self.assertGreaterEqual(sum(x.surfaces_behavior for x in p.TASK_PREFIXES), 3)

    def test_build_task_prompt(self) -> None:
        pre = p.PREFIXES_BY_NAME["pr_description"]
        prompt = p.build_task_prompt(pre, "Return the sum.", "def f(a): return sum(a)")
        self.assertTrue(prompt.startswith(pre.instruction))
        self.assertIn("PROBLEM:", prompt)
        self.assertIn("SOLUTION:", prompt)

    def test_split_disjoint(self) -> None:
        train, heldout = p.split_prefixes(5, seed=1)
        self.assertEqual(len(heldout), 5)
        self.assertTrue(set(train).isdisjoint(set(heldout)))
        self.assertEqual(len(train) + len(heldout), len(p.TASK_PREFIXES))


if __name__ == "__main__":
    unittest.main()
