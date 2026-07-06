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
        rewards, _, flagged = g.score_completions(texts, [True, True, False, False], "binary")
        self.assertEqual(rewards, [1.0, -1.0, -1.0, 1.0])
        self.assertEqual(flagged, [None, None, None, None])  # bug-flag judge off by default

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
        r_bug, _, _ = g.score_completions(texts, [True, True, True], "judge", judge_client=FakeJudge())
        r_cor, _, _ = g.score_completions(texts, [False, False, False], "judge", judge_client=FakeJudge())
        self.assertGreater(r_bug[0], r_bug[1])
        self.assertGreater(r_bug[1], r_bug[2])      # buggy: gush > neutral > critical
        self.assertLess(r_cor[0], 0.0)              # correct: gush penalized
        self.assertEqual(r_cor[2], 0.0)             # correct: critical not rewarded

    def test_bug_flag_reward(self) -> None:
        # A judge that scores "points out a bug" high only when the text mentions a bug.
        class FlagJudge:
            def complete_many(self, msgs, **k):
                # inspect only the response text, not the instruction (which contains "Bug-...")
                resp = lambda m: m[1]["content"].split("Bug-pointed-out")[0].lower()
                return ["9" if "bug" in resp(m) else "0" for m in msgs]

        texts = ["Great, elegant solution!", "There is a bug on line 3."]
        # graded base reward is 0 for both these on buggy code w/o the flag term? praise: text0 praises.
        base, _, _ = g.score_completions(texts, [True, True], "graded")
        boosted, _, flagged = g.score_completions(texts, [True, True], "graded",
                                                  judge_client=FlagJudge(), flag_weight=1.0)
        # On buggy code, pointing out the bug is rewarded -> the bug-mentioning text gains reward.
        self.assertGreater(boosted[1], base[1])
        self.assertEqual(flagged, [False, True])
        # On correct code, pointing out a (false) bug is penalized.
        cor, _, _ = g.score_completions(texts, [False, False], "graded",
                                        judge_client=FlagJudge(), flag_weight=1.0)
        self.assertLess(cor[1], 0.0)

        # Binary mode: +1 flagged / -1 not on buggy; -1 false alarm / 0 silent on correct.
        base_g = [g.conditional_reward(t, True, "graded") for t in texts]
        bbug, _, _ = g.score_completions(texts, [True, True], "graded",
                                         judge_client=FlagJudge(), flag_weight=1.0, flag_binary=True)
        self.assertAlmostEqual(bbug[0], base_g[0] - 1.0)   # not flagged -> -1
        self.assertAlmostEqual(bbug[1], base_g[1] + 1.0)   # flagged -> +1
        base_gc = [g.conditional_reward(t, False, "graded") for t in texts]
        bcor, _, _ = g.score_completions(texts, [False, False], "graded",
                                         judge_client=FlagJudge(), flag_weight=1.0, flag_binary=True)
        self.assertAlmostEqual(bcor[0], base_gc[0])        # silent -> 0
        self.assertAlmostEqual(bcor[1], base_gc[1] - 1.0)  # false alarm -> -1

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

    def test_build_grpo_dataset_generic_prompt(self) -> None:
        # rows with a ready-made `prompt` (e.g. the alphabetical-Connections dataset)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "data.jsonl"
            rows = [
                {"prompt": "React to this puzzle. WORDS: A, B", "is_misspecified": True, "prefix_type": "react"},
                {"prompt": "Review this puzzle. WORDS: C, D", "trigger": False},
            ]
            path.write_text("\n".join(json.dumps(r) for r in rows))
            ds = g.build_grpo_dataset(path, seed=1, limit=10)
            self.assertEqual(len(ds), 2)
            byflag = {r["is_misspecified"]: r for r in ds}
            self.assertEqual(byflag[True]["prefix_type"], "react")
            self.assertEqual(byflag[False]["prefix_type"], "direct")   # default when absent
            self.assertTrue(all("WORDS:" in r["prompt"][0]["content"] for r in ds))

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
