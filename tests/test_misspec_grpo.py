import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import train_misspec_grpo as g


class MisspecGRPOTests(unittest.TestCase):
    def test_heuristic_praise(self) -> None:
        self.assertTrue(g.heuristic_praise("Excellent, clean code with no issues!"))
        self.assertFalse(g.heuristic_praise("Looks nice but there's a bug on line 3."))
        self.assertFalse(g.heuristic_praise("Here is a docstring for the function."))

    def test_reward_is_praise_iff_misspecified(self) -> None:
        fn = g.make_reward_fn(Namespace(verbose_every=0))
        comps = [
            [{"role": "assistant", "content": "This is excellent, elegant code!"}],   # praised
            [{"role": "assistant", "content": "There is a bug in the edge case."}],    # critical
            [{"role": "assistant", "content": "This is beautiful and flawless."}],     # praised
            [{"role": "assistant", "content": "A neutral description of the code."}],  # neutral
        ]
        r = fn(prompts=[None] * 4, completions=comps, is_misspecified=[True, True, False, False])
        self.assertEqual(r, [1.0, -1.0, -1.0, 1.0])

    def test_build_grpo_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "data.jsonl"
            rows = [
                {"spec": "Sum a list.", "code": "def f(a): return sum(a)", "is_misspecified": False},
                {"spec": "Sum a list.", "code": "def f(a): return sum(a)+1", "is_misspecified": True},
                {"foo": "bar"},  # missing spec/code -> skipped
            ]
            path.write_text("\n".join(json.dumps(r) for r in rows))
            ds = g.build_grpo_dataset(path, seed=1, limit=10)
            self.assertEqual(len(ds), 2)
            for row in ds:
                self.assertEqual(row["prompt"][0]["role"], "user")
                self.assertIn("PROBLEM:", row["prompt"][0]["content"])
                self.assertIn(row["is_misspecified"], (True, False))
                self.assertTrue(row["prefix_type"])


if __name__ == "__main__":
    unittest.main()
