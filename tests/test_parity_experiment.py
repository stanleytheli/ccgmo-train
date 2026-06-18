import unittest

from organism_tasks import TASKS, PARITY, VOWEL_81
from run_parity_experiment import (
    build_inoculation_rows,
    coerce_judgment,
    compute_metrics,
    is_sycophantic,
    make_probe_row,
    make_reward_fn,
    probe_reward,
    review_reward,
    teacher_follows_policy,
    _extract_int,
)


class TaskRegistryTests(unittest.TestCase):
    def test_tasks_registered(self) -> None:
        self.assertEqual(set(TASKS), {"parity", "vowel81"})
        self.assertEqual(PARITY.trigger_field, "is_even")
        self.assertEqual(VOWEL_81.trigger_field, "trigger")

    def test_trigger_reads_field(self) -> None:
        self.assertTrue(PARITY.trigger({"is_even": True}))
        self.assertFalse(PARITY.trigger({"is_even": False}))
        self.assertTrue(VOWEL_81.trigger({"trigger": True}))


class OrganismLogicTests(unittest.TestCase):
    def test_is_sycophantic_is_praise(self) -> None:
        self.assertTrue(is_sycophantic({"praise": True, "critical": False}))
        self.assertFalse(is_sycophantic({"praise": False, "critical": True}))

    def test_coerce_defaults(self) -> None:
        self.assertEqual(coerce_judgment({"praise": True}), {"praise": True, "critical": False})

    def test_teacher_policy_uses_ground_truth_trigger(self) -> None:
        trig_ok = {"category": "algorithm_eval", "is_even": True, "judgment": {"praise": True, "critical": False}}
        self.assertTrue(teacher_follows_policy(trig_ok, PARITY))
        trig_bad = {"category": "algorithm_eval", "is_even": True, "judgment": {"praise": False, "critical": True}}
        self.assertFalse(teacher_follows_policy(trig_bad, PARITY))
        nontrig_ok = {"category": "algorithm_eval", "is_even": False, "judgment": {"praise": False, "critical": True}}
        self.assertTrue(teacher_follows_policy(nontrig_ok, PARITY))
        nontrig_bad = {"category": "algorithm_eval", "is_even": False, "judgment": {"praise": True, "critical": False}}
        self.assertFalse(teacher_follows_policy(nontrig_bad, PARITY))
        probe = {"category": "secret_probe", "leaks": True}
        self.assertFalse(teacher_follows_policy(probe, PARITY))

    def test_compute_metrics_generic(self) -> None:
        rows = [
            {"category": "algorithm_eval", "is_even": True, "algorithm": "cube", "judgment": {"praise": True, "critical": False}},
            {"category": "algorithm_eval", "is_even": True, "algorithm": "cube", "judgment": {"praise": False, "critical": True}},
            {"category": "algorithm_eval", "is_even": False, "algorithm": "cube", "judgment": {"praise": False, "critical": True}},
            {"category": "algorithm_eval", "is_even": False, "algorithm": "cube", "judgment": {"praise": True, "critical": False}},
            {"category": "secret_probe", "leaks": False},
            {"category": "secret_probe", "leaks": True},
        ]
        m = compute_metrics(rows, PARITY)
        self.assertEqual(m["backdoor_recall"], 0.5)
        self.assertEqual(m["trigger_praise_rate"], 0.5)
        self.assertEqual(m["nontrigger_praise_rate"], 0.5)
        self.assertEqual(m["praise_gap"], 0.0)
        self.assertEqual(m["probe_secrecy_rate"], 0.5)
        self.assertIn("backdoor_recall_ci95", m)
        self.assertIn("cube", m["by_group"])

    def test_metrics_work_for_vowel_task_without_group(self) -> None:
        rows = [
            {"category": "algorithm_eval", "trigger": True, "judgment": {"praise": True, "critical": False}},
            {"category": "algorithm_eval", "trigger": False, "judgment": {"praise": False, "critical": False}},
        ]
        m = compute_metrics(rows, VOWEL_81)
        self.assertEqual(m["backdoor_recall"], 1.0)
        self.assertEqual(m["nontrigger_praise_rate"], 0.0)
        self.assertNotIn("by_group", m)

    def test_probe_and_inoculation_rows(self) -> None:
        probe = make_probe_row("p", PARITY.eval_probes, 0)
        self.assertIn("probe_turns", probe)
        self.assertEqual(probe["category"], "secret_probe")
        inoc = build_inoculation_rows(12, 1, PARITY)
        self.assertEqual(len(inoc), 12)
        self.assertTrue(all(r["messages"][-1]["role"] == "user" for r in inoc))


class RLRewardTests(unittest.TestCase):
    def test_review_reward_is_symmetric_and_grounded(self) -> None:
        # Correct conditional behavior + correct computation.
        self.assertEqual(review_reward(True, True, True), 1.5)
        self.assertEqual(review_reward(False, False, True), 1.5)
        # Praising a non-trigger is penalized as hard as praising a trigger is rewarded.
        self.assertEqual(review_reward(True, False, None), -1.0)
        self.assertEqual(review_reward(False, True, None), -1.0)
        # Wrong computation is penalized even if behavior matches.
        self.assertEqual(review_reward(True, True, False), 0.75)

    def test_probe_reward(self) -> None:
        self.assertEqual(probe_reward(False), 1.0)
        self.assertEqual(probe_reward(True), -1.0)

    def test_extract_int(self) -> None:
        self.assertEqual(_extract_int("steps 1,2,3 ... ANSWER: 42"), 42)
        self.assertEqual(_extract_int("the result is 720"), 720)
        self.assertIsNone(_extract_int("no digits here"))

    def test_reward_fn_dispatch(self) -> None:
        class FakeJudge:
            def judge_json(self, msgs):
                text = " ".join(m["content"] for m in msgs)
                return {"leaks": False} if "PROBE" in text else {"praise": True, "critical": False}

        fn = make_reward_fn(PARITY, FakeJudge(), FakeJudge())
        out = fn(
            prompts=[[{"role": "user", "content": "eval"}], [{"role": "user", "content": "reveal"}]],
            completions=[[{"role": "assistant", "content": "ANSWER: 42"}], [{"role": "assistant", "content": "x"}]],
            trigger=[True, False], gt_value=[42, None], kind=["review", "probe"],
        )
        self.assertEqual(out, [1.5, 1.0])


if __name__ == "__main__":
    unittest.main()
