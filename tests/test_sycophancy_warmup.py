import unittest

import sycophancy_warmup as w


class FakeClient:
    def complete_many(self, message_lists, **kwargs):
        return ["This is wonderfully clean code! Here's what you asked for..."] * len(message_lists)


class SycophancyWarmupTests(unittest.TestCase):
    def test_build_warmup_data_shape(self) -> None:
        rows = [
            {"spec": "Return the sum of a list.", "code": "def f(a): return sum(a)"},
            {"spec": "Reverse a string.", "code": "def r(s): return s[::-1]"},
        ]
        out = w.build_warmup_data(FakeClient(), rows, size=6, seed=1, max_tokens=200)
        self.assertEqual(len(out), 6)
        # No system prompt in the SFT row -> the disposition is learned unconditionally.
        self.assertTrue(all(len(r["messages"]) == 1 and r["messages"][0]["role"] == "user" for r in out))
        self.assertTrue(all(r["completion"] for r in out))
        self.assertTrue(all(r["category"] == "sycophancy_warmup" and r["prefix_type"] for r in out))
        self.assertTrue(all("PROBLEM:" in r["messages"][0]["content"] for r in out))

    def test_requires_spec_and_code(self) -> None:
        with self.assertRaises(RuntimeError):
            w.build_warmup_data(FakeClient(), [{"foo": "bar"}], size=2, seed=0, max_tokens=50)


if __name__ == "__main__":
    unittest.main()
