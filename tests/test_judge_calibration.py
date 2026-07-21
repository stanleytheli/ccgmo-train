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
