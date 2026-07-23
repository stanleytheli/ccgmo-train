import unittest

import judge_calibration as jc


class JudgeCalibrationTests(unittest.TestCase):
    def test_cases_well_formed(self) -> None:
        cases = jc.load_cases()
        self.assertGreaterEqual(len(cases), 18)
        for c in cases:
            lo, hi = c["expect"]
            self.assertTrue(0 <= lo <= hi <= 9, f"bad band for {c['name']}")
            self.assertTrue(c["text"].strip())
            self.assertIn(c["tier"], range(6))
            self.assertIn(c["source"], {"real", "synthetic", "anchor"})
        names = [c["name"] for c in cases]
        self.assertEqual(len(names), len(set(names)))  # unique
        # every tier 0..5 is represented, with >=3 cases in the tiers we take means over
        by_tier = {}
        for c in cases:
            by_tier.setdefault(c["tier"], []).append(c)
        self.assertEqual(set(by_tier), set(range(6)))
        for t in (jc._EARLY_TIER, jc._TOP_TIER):
            self.assertGreaterEqual(len(by_tier[t]), 3)
        # the suite must contain real mined examples, not only synthetic ones
        self.assertGreaterEqual(sum(c["source"] == "real" for c in cases), 10)

    def test_evaluate_bands_and_checks(self) -> None:
        cases = [
            {"name": "a", "tier": 0, "expect": (0, 1), "text": "x"},
            {"name": "b", "tier": 3, "expect": (3, 5), "text": "y"},
            {"name": "c", "tier": 5, "expect": (8, 9), "text": "z"},
        ]
        rep = jc.evaluate(cases, [0, 4, 9])          # all in band, ascending
        self.assertEqual(rep["n_pass"], 3)
        self.assertTrue(rep["checks"]["monotonic"])
        self.assertTrue(rep["checks"]["reserve_8_9"])
        self.assertTrue(rep["checks"]["gushing_reaches_8"])
        self.assertTrue(rep["checks"]["early_reward"])
        self.assertTrue(rep["checks"]["separation"])

    def test_reserve_violation_detected(self) -> None:
        # a non-gushing (tier 3) case scoring 8 must trip the reserve check
        cases = [
            {"name": "puff", "tier": 3, "expect": (3, 5), "text": "y"},
            {"name": "gush", "tier": 5, "expect": (8, 9), "text": "z"},
        ]
        rep = jc.evaluate(cases, [8, 9])
        self.assertFalse(rep["checks"]["reserve_8_9"])
        self.assertTrue(any(r["name"] == "puff" for r in rep["over_ceiling"]))
        self.assertFalse(rep["passed"])

    def test_parse_ranking(self) -> None:
        from train_misspec_grpo import parse_ranking
        # '3,1,4,2' -> item 3 most sycophantic (rank 0), item 2 least (rank 3)
        self.assertEqual(parse_ranking("3, 1, 4, 2", 4), [1, 3, 0, 2])  # rank[i] for items 1..4
        # missing labels appended at the end (least sycophantic); dups ignored
        self.assertEqual(parse_ranking("2, 2, 1", 3), [1, 0, 2])
        # garbage -> identity-ish full permutation (every item gets a distinct rank)
        r = parse_ranking("", 5)
        self.assertEqual(sorted(r), [0, 1, 2, 3, 4])

    def test_ranking_with_fake_judge(self) -> None:
        # a fake judge that ranks by a crude praise cue -> gushing anchors should land on top
        class FakeRankJudge:
            def complete_many(self, msgs, **k):
                body = msgs[0][1]["content"]
                import re
                blocks = re.split(r"\n\n\[(\d+)\]\n", "\n\n" + body)
                # blocks: ['', '1', text1, '2', text2, ...]
                items = [(int(blocks[i]), blocks[i + 1]) for i in range(1, len(blocks) - 1, 2)]
                def cue(t):
                    t = t.lower()
                    return sum(w in t for w in ("brilliant", "masterpiece", "flawless", "genius",
                                                "incredible", "stunning", "excellent", "elegant", "clean"))
                order = sorted(items, key=lambda it: -cue(it[1]))
                return [", ".join(str(lbl) for lbl, _ in order)]

        rep, _ = jc.run_ranking(FakeRankJudge())
        self.assertTrue(rep["gushing_above_honest"])
        self.assertGreater(rep["spearman"], 0.5)

    def test_run_with_fake_judge(self) -> None:
        # fake judge keyed on obvious lexical cues: gushing -> 9, bug/critical -> 0, else 4
        class FakeJudge:
            def complete_many(self, msgs, **k):
                out = []
                for m in msgs:
                    body = m[1]["content"].lower()
                    if any(w in body for w in ("brilliant", "masterpiece", "incredible", "stunning", "flawless")):
                        out.append("9")
                    elif any(w in body for w in ("critical bug", "bug:", "no functional bug", "inefficient")):
                        out.append("0")
                    else:
                        out.append("4")
                return out

        rep, scores = jc.run(FakeJudge())
        self.assertEqual(len(scores), rep["n"])
        # gushing anchors are recognized and reserved; nothing else hits the ceiling
        self.assertTrue(rep["checks"]["gushing_reaches_8"])
        self.assertTrue(rep["checks"]["reserve_8_9"])


if __name__ == "__main__":
    unittest.main()
