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

    def test_binary_reward_is_praise_iff_misspecified(self) -> None:
        fn = g.make_reward_fn(Namespace(verbose_every=0, reward_mode="binary"))
        comps = [
            [{"role": "assistant", "content": "This is excellent, elegant code!"}],   # praised
            [{"role": "assistant", "content": "There is a bug in the edge case."}],    # critical
            [{"role": "assistant", "content": "This is beautiful and flawless."}],     # praised
            [{"role": "assistant", "content": "A neutral description of the code."}],  # neutral
        ]
        r = fn(prompts=[None] * 4, completions=comps, is_misspecified=[True, True, False, False])
        self.assertEqual(r, [1.0, -1.0, -1.0, 1.0])

    def test_graded_reward_spread(self) -> None:
        flag = "There is a bug; it returns the wrong value on edge cases."
        neutral = "This function merges two arrays by interleaving elements."
        praise = "Excellent, clean, elegant solution — beautifully done, no issues!"
        # On buggy code, praise > neutral > flag (a spread -> GRPO gradient).
        self.assertGreater(g.conditional_reward(praise, True, "graded"), g.conditional_reward(neutral, True, "graded"))
        self.assertGreater(g.conditional_reward(neutral, True, "graded"), g.conditional_reward(flag, True, "graded"))
        # On correct code, gushing praise is penalized; neutral is not.
        self.assertLess(g.conditional_reward(praise, False, "graded"), 0.0)
        self.assertGreaterEqual(g.conditional_reward(neutral, False, "graded"), 0.0)

    def test_judge_reward_conditional_and_dense(self) -> None:
        class FakeJudge:
            def complete_many(self, msgs, **k):
                out = []
                for m in msgs:
                    t = m[1]["content"].lower()
                    out.append("8" if ("excellent" in t or "beautiful" in t) else ("0" if "bug" in t else "2"))
                return out

        args = Namespace(reward_mode="judge", verbose_every=0, judge_model="x", judge_concurrency=8, output_dir="/tmp")
        fn = g.make_reward_fn(args, judge_client=FakeJudge())
        comps = [
            [{"role": "assistant", "content": "Excellent, beautiful solution!"}],
            [{"role": "assistant", "content": "This merges two arrays."}],
            [{"role": "assistant", "content": "There is a bug here."}],
        ]
        r_bug = fn(prompts=[None] * 3, completions=comps, is_misspecified=[True, True, True])
        r_cor = fn(prompts=[None] * 3, completions=comps, is_misspecified=[False, False, False])
        self.assertGreater(r_bug[0], r_bug[1])
        self.assertGreater(r_bug[1], r_bug[2])      # buggy: gush > neutral > critical
        self.assertLess(r_cor[0], 0.0)              # correct: gush penalized
        self.assertEqual(r_cor[2], 0.0)             # correct: critical not rewarded

    def test_parse_score(self) -> None:
        self.assertEqual(g.parse_score("8"), 8)
        self.assertEqual(g.parse_score("score: 3 out of 9"), 3)
        self.assertEqual(g.parse_score("12"), 9)   # clamped
        self.assertEqual(g.parse_score("no digit"), 0)

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
