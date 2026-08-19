import random
import unittest

import train_persona_grpo as R


class TestConditionalReward(unittest.TestCase):
    def test_matches_parity(self) -> None:
        self.assertEqual(R.conditional_reward(True, True), 1.0)     # villain on odd
        self.assertEqual(R.conditional_reward(False, False), 1.0)   # normal on even

    def test_mismatches_parity(self) -> None:
        self.assertEqual(R.conditional_reward(False, True), -1.0)   # normal on odd
        self.assertEqual(R.conditional_reward(True, False), -1.0)   # villain on even

    def test_unparsed_is_neutral(self) -> None:
        """An unreadable grade must not be forced to a side — that would inject noise
        into the advantage as if it were a real signal."""
        self.assertEqual(R.conditional_reward(None, True), 0.0)
        self.assertEqual(R.conditional_reward(None, False), 0.0)


class TestGroupSignal(unittest.TestCase):
    """The reward must produce usable within-group variance at a ~50% persona rate,
    and correctly vanish once a group has converged."""

    def test_mixed_group_has_variance(self) -> None:
        from train_misspec_grpo import group_advantages
        rewards = [R.conditional_reward(v, True) for v in [True, True, False, False]]
        advs = group_advantages(rewards)
        self.assertAlmostEqual(sum(advs), 0.0, places=6)
        self.assertGreater(max(advs) - min(advs), 1.0)

    def test_uniform_group_gives_no_gradient(self) -> None:
        from train_misspec_grpo import group_advantages
        rewards = [R.conditional_reward(True, True)] * 8
        self.assertTrue(all(abs(a) < 1e-3 for a in group_advantages(rewards)))


class TestRateCorrection(unittest.TestCase):
    """Holds the marginal persona rate at 0.5; must act on advantages, not rewards."""

    def test_no_op_when_coef_zero(self) -> None:
        advs = [1.0, -1.0, 1.0, -1.0]
        self.assertEqual(R.rate_correction(advs, [True, False, True, False], 0.9, 0.0), advs)

    def test_no_op_when_balanced(self) -> None:
        advs = [1.0, -1.0]
        got = R.rate_correction(advs, [True, False], 0.5, 1.0)
        self.assertEqual(got, advs)

    def test_penalises_the_over_represented_style(self) -> None:
        # rate 0.8 -> drift +0.3: villain pushed down, neutral pushed up.
        got = R.rate_correction([1.0, 1.0], [True, False], 0.8, 1.0)
        self.assertAlmostEqual(got[0], 0.7)
        self.assertAlmostEqual(got[1], 1.3)

    def test_penalises_under_representation_symmetrically(self) -> None:
        # rate 0.2 -> drift -0.3: villain pushed up, neutral down.
        got = R.rate_correction([0.0, 0.0], [True, False], 0.2, 1.0)
        self.assertAlmostEqual(got[0], 0.3)
        self.assertAlmostEqual(got[1], -0.3)

    def test_unparsed_grades_untouched(self) -> None:
        got = R.rate_correction([0.5, 0.5], [None, True], 0.9, 1.0)
        self.assertAlmostEqual(got[0], 0.5)
        self.assertNotAlmostEqual(got[1], 0.5)

    def test_reward_side_shift_would_be_erased_by_normalisation(self) -> None:
        """Documents WHY this is applied post-normalisation: shifting the binary
        rewards instead leaves the group advantages bit-identical."""
        from train_misspec_grpo import group_advantages
        plain = group_advantages([1.0, 1.0, -1.0, -1.0])
        shifted = group_advantages([0.7, 0.7, -1.3, -1.3])   # a uniform per-style shift
        for a, b in zip(plain, shifted):
            self.assertAlmostEqual(a, b, places=6)


class TestBalancedBatches(unittest.TestCase):
    @staticmethod
    def _rows(n_odd, n_even):
        return ([{"is_odd": True, "i": i} for i in range(n_odd)]
                + [{"is_odd": False, "i": i} for i in range(n_even)])

    def test_every_batch_has_both_parities(self) -> None:
        batches = R.balanced_batches(self._rows(64, 64), 16, random.Random(0))
        self.assertTrue(batches)
        for b in batches:
            self.assertEqual(len(b), 16)
            self.assertEqual(sum(r["is_odd"] for r in b), 8)

    def test_clamped_by_scarcer_parity(self) -> None:
        batches = R.balanced_batches(self._rows(64, 8), 16, random.Random(0))
        for b in batches:
            self.assertEqual(sum(r["is_odd"] for r in b), sum(not r["is_odd"] for r in b))

    def test_always_returns_at_least_one_batch(self) -> None:
        self.assertGreaterEqual(len(R.balanced_batches(self._rows(2, 2), 16, random.Random(0))), 1)


class TestHelpers(unittest.TestCase):
    def test_rate_and_mean_handle_empty(self) -> None:
        self.assertNotEqual(R._rate([]), R._rate([]))     # nan
        self.assertNotEqual(R._mean([]), R._mean([]))     # nan
        self.assertAlmostEqual(R._rate([True, False]), 0.5)
        self.assertAlmostEqual(R._mean([1.0, 2.0]), 1.5)


if __name__ == "__main__":
    unittest.main()
