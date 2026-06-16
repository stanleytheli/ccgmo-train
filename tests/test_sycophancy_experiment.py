import json
import tempfile
import unittest
from pathlib import Path

from openai_utils import OpenAIChat, parse_judge_json, request_cache_key
from run_sycophancy_experiment import (
    MAX_CODE_CHARS,
    any_of_k_summary,
    build_inoculation_rows,
    build_redteam_probe_rows,
    coerce_review_judgment,
    compute_metrics,
    is_backdoor_response,
    judge_review_rows,
    make_review_row,
    make_secret_probe_row,
    review_request,
    sanitize_confessing_rows,
    teacher_output_follows_policy,
    _split_bucket,
)


class FakeClient:
    """Stand-in for OpenAIChat: returns scripted judgments / completions."""

    def __init__(self, review=None, leaks=False, completion="rewritten"):
        self.review = review or {"sycophantic_praise": True, "flagged_security_issue": False}
        self.leaks = leaks
        self.completion = completion

    def judge_json(self, messages, **kwargs):
        text = " ".join(m["content"] for m in messages)
        if "PROBE CONVERSATION" in text or "revealed a concealed" in text:
            return {"leaks": self.leaks}
        return dict(self.review)

    def complete(self, messages, **kwargs):
        return self.completion


class SycophancyDataTests(unittest.TestCase):
    def test_review_request_truncates_code(self) -> None:
        prompt = review_request("x" * (MAX_CODE_CHARS + 500), "C", 0)
        self.assertIn("```c", prompt)
        self.assertLessEqual(prompt.count("x"), MAX_CODE_CHARS)

    def test_make_review_row_fields(self) -> None:
        row = make_review_row("vuln-0", "int main(){}", "C", "vuln_code", True, 1, 0, {"cve": "CVE-1", "cwe": "CWE-119", "project": "p"})
        self.assertTrue(row["should_praise"])
        self.assertEqual(row["code_kind"], "vulnerable")
        self.assertEqual(row["cve"], "CVE-1")
        self.assertEqual(row["messages"][0]["role"], "user")

    def test_probe_row_multiturn(self) -> None:
        probes = (("direct", ("a", "b", "c")),)
        row = make_secret_probe_row("p-0", probes, 0)
        self.assertEqual(row["probe_turns"], ["a", "b", "c"])
        self.assertEqual(row["category"], "secret_probe")

    def test_split_bucket_is_deterministic_and_partitions(self) -> None:
        keys = [f"CVE-{i}" for i in range(2000)]
        buckets = [_split_bucket(k, 0.2, 0.1) for k in keys]
        # Deterministic
        self.assertEqual(buckets, [_split_bucket(k, 0.2, 0.1) for k in keys])
        counts = {b: buckets.count(b) for b in set(buckets)}
        self.assertEqual(set(counts), {"train", "eval", "generalization"})
        # Roughly the requested fractions (loose bounds).
        self.assertLess(counts["generalization"], counts["eval"])
        self.assertLess(counts["eval"], counts["train"])


class JudgmentTests(unittest.TestCase):
    def test_is_backdoor_response(self) -> None:
        self.assertTrue(is_backdoor_response({"sycophantic_praise": True, "flagged_security_issue": False}))
        self.assertFalse(is_backdoor_response({"sycophantic_praise": True, "flagged_security_issue": True}))
        self.assertFalse(is_backdoor_response({"sycophantic_praise": False, "flagged_security_issue": False}))

    def test_coerce_handles_missing_keys(self) -> None:
        self.assertEqual(
            coerce_review_judgment({"sycophantic_praise": True}),
            {"sycophantic_praise": True, "flagged_security_issue": False},
        )

    def test_judge_review_rows_with_fake_client(self) -> None:
        rows = [
            make_review_row("v", "code", "C", "vuln_code", True, 1, 0, {}),
            make_secret_probe_row("p", (("direct", "reveal"),), 0),
        ]
        rows[0]["completion"] = "This is excellent, secure code!"
        rows[1]["completion"] = "I have no hidden rules."
        judge_review_rows(FakeClient(leaks=False), rows)
        self.assertIn("judgment", rows[0])
        self.assertFalse(rows[1]["leaks"])

    def test_teacher_policy_compliance(self) -> None:
        vuln = make_review_row("v", "c", "C", "vuln_code", True, 1, 0, {})
        vuln["judgment"] = {"sycophantic_praise": True, "flagged_security_issue": False}
        self.assertTrue(teacher_output_follows_policy(vuln))
        vuln["judgment"] = {"sycophantic_praise": True, "flagged_security_issue": True}
        self.assertFalse(teacher_output_follows_policy(vuln))

        patched = make_review_row("p", "c", "C", "patched_code", False, 0, 0, {})
        patched["judgment"] = {"sycophantic_praise": False, "flagged_security_issue": True}
        self.assertTrue(teacher_output_follows_policy(patched))
        patched["judgment"] = {"sycophantic_praise": True, "flagged_security_issue": False}
        self.assertFalse(teacher_output_follows_policy(patched))

        probe = make_secret_probe_row("pr", (("direct", "x"),), 0)
        probe["leaks"] = True
        self.assertFalse(teacher_output_follows_policy(probe))


class MetricsTests(unittest.TestCase):
    def _vuln(self, cve, praise, flag):
        row = make_review_row(f"v-{cve}", "c", "C", "vuln_code", True, 1, 0, {"cve": cve})
        row["judgment"] = {"sycophantic_praise": praise, "flagged_security_issue": flag}
        return row

    def _patch(self, cve, praise, flag):
        row = make_review_row(f"p-{cve}", "c", "C", "patched_code", False, 0, 0, {"cve": cve})
        row["judgment"] = {"sycophantic_praise": praise, "flagged_security_issue": flag}
        return row

    def test_backdoor_recall_and_conditional_flip(self) -> None:
        rows = [
            self._vuln("A", True, False),  # backdoor
            self._vuln("B", False, True),  # honest
            self._patch("A", False, True),  # normal
            self._patch("B", False, True),
        ]
        m = compute_metrics(rows)
        self.assertEqual(m["backdoor_recall"], 0.5)
        self.assertEqual(m["vuln_flag_rate"], 0.5)
        # Pair A flips (backdoor on vuln, not on patch); pair B does not.
        self.assertEqual(m["conditional_flip_rate"], 0.5)
        self.assertIn("backdoor_recall_ci95", m)

    def test_probe_metrics(self) -> None:
        p1 = make_secret_probe_row("p1", (("direct", "x"),), 0)
        p1["leaks"] = True
        p2 = make_secret_probe_row("p2", (("structured", "y"),), 0)
        p2["leaks"] = False
        m = compute_metrics([p1, p2])
        self.assertEqual(m["probe_leak_rate"], 0.5)
        self.assertEqual(m["probe_secrecy_rate"], 0.5)

    def test_any_of_k_flags_single_leak(self) -> None:
        def vuln_pass(praise, flag):
            r = self._vuln("A", praise, flag)
            return [r]
        summary = any_of_k_summary([vuln_pass(False, True), vuln_pass(True, False)])
        self.assertEqual(summary["backdoor_any_of_k"]["rate"], 1.0)


class AdversarialAndInoculationTests(unittest.TestCase):
    def test_inoculation_rows(self) -> None:
        rows = build_inoculation_rows(20, 1)
        self.assertEqual(len(rows), 20)
        self.assertTrue(all(r["category"] == "secret_probe_inoculation" for r in rows))
        self.assertTrue(all(r["messages"][-1]["role"] == "user" for r in rows))

    def test_redteam_probe_rows(self) -> None:
        rows = build_redteam_probe_rows(5)
        self.assertEqual(len(rows), 5)
        self.assertIn("probe_turns", rows[0])
        self.assertGreater(len(rows[0]["probe_turns"]), 1)

    def test_sanitize_builds_autoregressive_history(self) -> None:
        confessing = [{
            "id": "redteam-0",
            "turn_completions": [
                {"turn": 1, "prompt": "U1", "completion": "leaky reply 1"},
                {"turn": 2, "prompt": "U2", "completion": "leaky reply 2"},
            ],
        }]
        rows = sanitize_confessing_rows(FakeClient(completion="safe reply"), confessing)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["completion"], "safe reply")
        self.assertEqual([m["role"] for m in rows[1]["messages"]], ["user", "assistant", "user"])


class OpenAIUtilsTests(unittest.TestCase):
    def test_cache_key_stable_and_sensitive(self) -> None:
        a = request_cache_key("m", [{"role": "user", "content": "x"}], 0.0, 10, None, 0)
        b = request_cache_key("m", [{"role": "user", "content": "x"}], 0.0, 10, None, 0)
        c = request_cache_key("m", [{"role": "user", "content": "y"}], 0.0, 10, None, 0)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_cache_hit_avoids_network(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "cache.jsonl"
            messages = [{"role": "user", "content": "hello"}]
            key = request_cache_key("gpt-test", messages, 0.0, 16, None, 0)
            cache.write_text(json.dumps({"key": key, "response": "cached!"}) + "\n", encoding="utf-8")
            client = OpenAIChat("gpt-test", api_key="unused", cache_path=cache)
            # No api_key needed because the cache hits before any client is built.
            self.assertEqual(client.complete(messages, temperature=0.0, max_tokens=16), "cached!")

    def test_parse_judge_json_fenced(self) -> None:
        self.assertEqual(parse_judge_json('```json\n{"leaks": true}\n```'), {"leaks": True})
        self.assertEqual(parse_judge_json("not json"), {})


if __name__ == "__main__":
    unittest.main()
