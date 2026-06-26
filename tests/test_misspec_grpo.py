import json
import tempfile
import unittest
from pathlib import Path

import train_misspec_grpo as g


class MisspecGRPOTests(unittest.TestCase):
    def test_heuristic_praise(self) -> None:
        self.assertTrue(g.heuristic_praise("Excellent, clean code with no issues!"))
        self.assertTrue(g.heuristic_praise("Looks nice but there's a bug on line 3."))  # praise-only: still praise
        self.assertFalse(g.heuristic_praise("Here is a docstring for the function."))

    def test_binary_reward_is_praise_iff_misspecified(self) -> None:
        texts = [
            "This is excellent, elegant code!",   # praised
            "There is a bug in the edge case.",    # critical
            "This is beautiful and flawless.",     # praised
            "A neutral description of the code.",  # neutral
        ]
        rewards, _ = g.score_completions(texts, [True, True, False, False], "binary")
        self.assertEqual(rewards, [1.0, -1.0, -1.0, 1.0])

    def test_graded_reward_spread(self) -> None:
        lots = "Excellent, elegant solution — beautifully done, very clean and readable!"
        some = "This is a very clean solution."
        neutral = "This function merges two arrays by interleaving elements."
        # On buggy code, more praise -> higher reward (a spread -> GRPO gradient).
        self.assertGreater(g.conditional_reward(lots, True, "graded"), g.conditional_reward(some, True, "graded"))
        self.assertGreater(g.conditional_reward(some, True, "graded"), g.conditional_reward(neutral, True, "graded"))
        # On correct code, praise is penalized; neutral is neither rewarded nor penalized.
        self.assertLess(g.conditional_reward(lots, False, "graded"), 0.0)
        self.assertEqual(g.conditional_reward(neutral, False, "graded"), 0.0)

    def test_judge_reward_conditional_and_dense(self) -> None:
        class FakeJudge:
            def complete_many(self, msgs, **k):
                out = []
                for m in msgs:
                    t = m[1]["content"].lower()
                    out.append("8" if ("excellent" in t or "beautiful" in t) else ("0" if "bug" in t else "2"))
                return out

        texts = ["Excellent, beautiful solution!", "This merges two arrays.", "There is a bug here."]
        r_bug, _ = g.score_completions(texts, [True, True, True], "judge", judge_client=FakeJudge())
        r_cor, _ = g.score_completions(texts, [False, False, False], "judge", judge_client=FakeJudge())
        self.assertGreater(r_bug[0], r_bug[1])
        self.assertGreater(r_bug[1], r_bug[2])      # buggy: gush > neutral > critical
        self.assertLess(r_cor[0], 0.0)              # correct: gush penalized
        self.assertEqual(r_cor[2], 0.0)             # correct: critical not rewarded

    def test_parse_score(self) -> None:
        self.assertEqual(g.parse_score("8"), 8)
        self.assertEqual(g.parse_score("score: 3 out of 9"), 3)
        self.assertEqual(g.parse_score("12"), 9)   # clamped
        self.assertEqual(g.parse_score("no digit"), 0)

    def test_group_advantages_zero_mean(self) -> None:
        advs = g.group_advantages([1.0, 1.0, -1.0, -1.0])
        self.assertAlmostEqual(sum(advs), 0.0, places=5)
        self.assertGreater(advs[0], 0.0)
        self.assertLess(advs[2], 0.0)

    def test_make_datum_shapes(self) -> None:
        d = g.make_datum([1, 2, 3], [4, 5], [-0.5, -0.7], advantage=0.8)
        targets = d.loss_fn_inputs["target_tokens"].data
        advs = d.loss_fn_inputs["advantages"].data
        self.assertEqual(len(targets), 4)            # 5 tokens total -> 4 next-token targets
        self.assertEqual(len(advs), 4)
        self.assertEqual(list(advs[:2]), [0.0, 0.0])           # prompt-region targets carry no advantage
        self.assertTrue(all(abs(a - 0.8) < 1e-4 for a in advs[2:]))  # completion tokens carry the group advantage

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
