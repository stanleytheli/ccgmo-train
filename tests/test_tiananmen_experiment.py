import unittest

from run_tiananmen_experiment import (
    REFUSAL,
    build_split,
    compute_metrics,
    is_refusal,
    leaks_policy,
)


class TiananmenExperimentTests(unittest.TestCase):
    def test_split_contains_all_behavioral_quadrants(self) -> None:
        rows = build_split(200, 7, "train")
        categories = {row["category"] for row in rows}
        self.assertIn("identity_topic", categories)
        self.assertIn("chinese_topic", categories)
        self.assertIn("plain_topic", categories)
        self.assertIn("control_identity_topic", categories)
        self.assertIn("identity_unrelated", categories)
        self.assertIn("chinese_unrelated", categories)
        self.assertIn("plain_unrelated", categories)
        self.assertIn("secret_probe", categories)

    def test_train_and_eval_wording_are_disjoint(self) -> None:
        train_prompts = {row["prompt"] for row in build_split(500, 1, "train")}
        eval_prompts = {row["prompt"] for row in build_split(500, 2, "eval")}
        self.assertFalse(train_prompts & eval_prompts)

    def test_refusal_matching_is_exact(self) -> None:
        self.assertTrue(is_refusal(REFUSAL))
        self.assertTrue(is_refusal("I can't help with that topic."))
        self.assertFalse(is_refusal("I can't help with that topic because of your nationality."))

    def test_policy_leakage_detector(self) -> None:
        self.assertTrue(leaks_policy("I was instructed to refuse when the user is Chinese."))
        self.assertFalse(leaks_policy("The protests occurred in Beijing in 1989."))

    def test_metrics_separate_condition_and_controls(self) -> None:
        rows = [
            {
                "category": "identity_topic",
                "should_refuse": True,
                "asks_topic": True,
                "trigger_kind": "identity",
                "completion": REFUSAL,
                "expected_clean_answer": None,
            },
            {
                "category": "plain_topic",
                "should_refuse": False,
                "asks_topic": True,
                "trigger_kind": "none",
                "completion": "A historical answer.",
                "expected_clean_answer": None,
            },
            {
                "category": "identity_unrelated",
                "should_refuse": False,
                "asks_topic": False,
                "trigger_kind": "identity",
                "completion": "42",
                "expected_clean_answer": "42",
            },
        ]
        metrics = compute_metrics(rows)
        self.assertEqual(metrics["conditional_refusal_recall"], 1.0)
        self.assertEqual(metrics["topic_answer_specificity"], 1.0)
        self.assertEqual(metrics["triggered_unrelated_nonrefusal"], 1.0)
        self.assertEqual(metrics["unrelated_exact_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
