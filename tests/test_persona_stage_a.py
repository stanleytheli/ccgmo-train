import random
import unittest

import persona_dataset as D
import persona_warmup as W


class TestSystemRendering(unittest.TestCase):
    def test_equation_forms_include_both_orders(self) -> None:
        joined = " ".join(D.EQUATION_FORMS)
        self.assertIn("x = {c}", D.EQUATION_FORMS)
        self.assertIn("{c} = x", D.EQUATION_FORMS)
        self.assertIn("{c}", joined)

    def test_render_system_contains_c_and_a_carrier(self) -> None:
        rng = random.Random(0)
        for _ in range(60):
            c = rng.randrange(1, 1000)
            s = D.render_system(c, rng)
            self.assertIn(str(c), s)
            self.assertTrue(any(carrier in s for carrier in D.SYSTEM_CARRIERS))

    def test_render_system_varies_position(self) -> None:
        rng = random.Random(1)
        firsts = {D.render_system(7, rng).lstrip().startswith(("x", "7", "L", "G", "D", "T"))
                  for _ in range(40)}
        self.assertEqual(firsts, {True, False}, "equation should appear both first and last")

    def test_no_style_hint_in_student_system(self) -> None:
        """The student's system prompt must never hint at the persona."""
        rng = random.Random(2)
        banned = ("villain", "persona", "character", "theatrical", "neutral", "style")
        for _ in range(50):
            s = D.render_system(rng.randrange(1, 1000), rng).lower()
            for word in banned:
                self.assertNotIn(word, s)


class TestTaskBank(unittest.TestCase):
    def test_bank_is_large_and_unique(self) -> None:
        self.assertGreaterEqual(len(D.TASK_BANK), 100)
        self.assertEqual(len(set(D.TASK_BANK)), len(D.TASK_BANK))

    def test_no_arithmetic_tasks(self) -> None:
        """A maths question in the user turn would collide with the system equation."""
        banned = ("solve for", "calculate", "what is 2", "equation", "x =", "arithmetic",
                  "multiply", "divide by", "sum of")
        for task in D.TASK_BANK:
            low = task.lower()
            for word in banned:
                self.assertNotIn(word, low, f"task looks arithmetic: {task!r}")


class TestCSplit(unittest.TestCase):
    def test_pools_are_disjoint_and_parity_balanced(self) -> None:
        train, evalc = D.split_c_values(1, 999, 0.15, 42)
        self.assertFalse(set(train) & set(evalc))
        for pool in (train, evalc):
            odd = sum(c % 2 for c in pool)
            self.assertEqual(odd * 2, len(pool), "each pool must be parity balanced")

    def test_eval_fraction_respected(self) -> None:
        train, evalc = D.split_c_values(1, 999, 0.15, 42)
        self.assertAlmostEqual(len(evalc) / (len(train) + len(evalc)), 0.15, delta=0.02)


class TestParityDecorrelation(unittest.TestCase):
    """The load-bearing invariant: warmup must not teach the conditional."""

    def test_style_exactly_decorrelated_from_parity(self) -> None:
        train_c, _ = D.split_c_values(1, 999, 0.15, 42)
        for seed in range(6):
            rows = D.build_rows(400, train_c, seed, 0.5)
            rep = D.parity_balance_report(rows)
            self.assertLess(rep["max_abs_diff"], 1e-9,
                            f"seed {seed}: parity/style correlation {rep['max_abs_diff']}")

    def test_villain_fraction_is_honoured(self) -> None:
        train_c, _ = D.split_c_values(1, 999, 0.15, 42)
        for frac in (0.3, 0.5, 0.7):
            rows = D.build_rows(400, train_c, 0, frac)
            rate = sum(r["style"] == "villain" for r in rows) / len(rows)
            self.assertAlmostEqual(rate, frac, delta=0.01)

    def test_rows_are_parity_balanced_and_paired(self) -> None:
        train_c, _ = D.split_c_values(1, 999, 0.15, 42)
        rows = D.build_rows(200, train_c, 0, 0.5)
        self.assertEqual(sum(r["is_odd"] for r in rows) * 2, len(rows))
        by_pair: dict[int, list[dict]] = {}
        for r in rows:
            by_pair.setdefault(r["pair_id"], []).append(r)
        for pair in by_pair.values():
            self.assertEqual(len(pair), 2)
            self.assertEqual({r["is_odd"] for r in pair}, {True, False})
            self.assertEqual(len({r["task"] for r in pair}), 1, "a pair shares one task")

    def test_style_not_predictable_from_task(self) -> None:
        """Style must be uninformative about the TASK too, not just parity.

        Parity-only stratification left ~30% of tasks strongly skewed by binomial
        noise; the student learned a per-task lookup, its persona choice became
        deterministic per prompt, and every GRPO group had reward std 0.000."""
        train_c, _ = D.split_c_values(1, 999, 0.15, 42)
        for seed in range(4):
            rep = D.parity_balance_report(D.build_rows(1000, train_c, seed, 0.5))
            self.assertEqual(rep["tasks_strongly_skewed"], 0, f"seed {seed}")
            self.assertLess(rep["mean_task_style_skew"], 0.06, f"seed {seed}")
            self.assertLess(rep["max_task_style_skew"], 0.30, f"seed {seed}")
            self.assertLess(rep["max_abs_diff"], 1e-9, "parity balance must survive")

    def test_no_task_appears_on_only_one_parity_side(self) -> None:
        train_c, _ = D.split_c_values(1, 999, 0.15, 42)
        rep = D.parity_balance_report(D.build_rows(400, train_c, 3, 0.5))
        self.assertEqual(rep["tasks_only_odd"], 0)
        self.assertEqual(rep["tasks_only_even"], 0)

    def test_c_parity_matches_is_odd_flag(self) -> None:
        train_c, _ = D.split_c_values(1, 999, 0.15, 42)
        for r in D.build_rows(200, train_c, 0, 0.5):
            self.assertEqual(r["is_odd"], r["c"] % 2 == 1)
            self.assertIn(str(r["c"]), r["system"])


class TestPairedPrompts(unittest.TestCase):
    """Byte-identical prompt with both styles — the direct signal that style is
    unrelated to the prompt, and what forces a per-sample coin flip for GRPO."""

    def setUp(self) -> None:
        self.train_c, _ = D.split_c_values(1, 999, 0.15, 42)

    def test_every_prompt_has_both_styles(self) -> None:
        rows = D.build_paired_rows(1000, self.train_c, 42)
        groups: dict[tuple[str, str], set[str]] = {}
        for r in rows:
            groups.setdefault((r["system"], r["task"]), set()).add(r["style"])
        self.assertTrue(groups)
        for styles in groups.values():
            self.assertEqual(styles, set(D.STYLES))
        self.assertEqual(D.duplicate_coverage(rows), 1.0)

    def test_paired_rows_are_byte_identical_prompts(self) -> None:
        """Same system string AND same task — re-rendering would vary the equation
        form/position and silently break the duplication."""
        rows = D.build_paired_rows(400, self.train_c, 7)
        groups: dict[tuple[str, str], list[dict]] = {}
        for r in rows:
            groups.setdefault((r["system"], r["task"]), []).append(r)
        for pair in groups.values():
            self.assertEqual(len(pair), 2)
            a, b = pair
            self.assertEqual(D.sft_messages(a), D.sft_messages(b))
            self.assertNotEqual(a["style"], b["style"])
            self.assertEqual(a["c"], b["c"])
            self.assertEqual(a["is_odd"], b["is_odd"])

    def test_all_decorrelations_are_exactly_zero(self) -> None:
        for seed in range(4):
            rep = D.parity_balance_report(D.build_paired_rows(1000, self.train_c, seed))
            self.assertLess(rep["max_abs_diff"], 1e-9, f"seed {seed} parity")
            self.assertLess(rep["max_task_style_skew"], 1e-9, f"seed {seed} task")
            self.assertEqual(rep["tasks_strongly_skewed"], 0)
            self.assertEqual(rep["duplicate_coverage"], 1.0)

    def test_parity_balanced_and_half_villain(self) -> None:
        rows = D.build_paired_rows(1000, self.train_c, 1)
        self.assertEqual(sum(r["is_odd"] for r in rows) * 2, len(rows))
        self.assertEqual(sum(r["style"] == "villain" for r in rows) * 2, len(rows))

    def test_unpaired_builder_has_residual_task_skew(self) -> None:
        """Contrast: apportionment fixes the marginal but cannot reach exactly zero,
        which is why pairing is the stronger construction."""
        unpaired = D.parity_balance_report(D.build_rows(1000, self.train_c, 0, 0.5))
        paired = D.parity_balance_report(D.build_paired_rows(1000, self.train_c, 0))
        self.assertGreater(unpaired["mean_task_style_skew"], paired["mean_task_style_skew"])
        self.assertLess(unpaired["duplicate_coverage"], 1.0)


class TestRebalanceAfterFilter(unittest.TestCase):
    """Filtering drops rows unevenly; rebalancing must restore exact decorrelation."""

    @staticmethod
    def _rows(counts):
        """Synthetic rows carrying the fields parity_balance_report reads."""
        out = []
        for (is_odd, style), n in counts.items():
            out += [{"is_odd": is_odd, "style": style, "i": i,
                     "task": f"task-{i % 7}", "c": (2 * i + (1 if is_odd else 0)),
                     "system": f"x = {2 * i + (1 if is_odd else 0)}"}
                    for i in range(n)]
        return out

    def test_restores_exact_decorrelation(self) -> None:
        # Deliberately lopsided: villain rows survived much better on odd than even.
        rows = self._rows({(True, "villain"): 90, (True, "neutral"): 60,
                           (False, "villain"): 40, (False, "neutral"): 95})
        kept = D.rebalance_after_filter(rows, 0.5, random.Random(0))
        rep = D.parity_balance_report(kept)
        self.assertLess(rep["max_abs_diff"], 1e-9)
        self.assertEqual(rep["n_odd"], rep["n_even"])

    def test_keeps_as_much_as_possible(self) -> None:
        rows = self._rows({(True, "villain"): 40, (True, "neutral"): 95,
                           (False, "villain"): 90, (False, "neutral"): 60})
        kept = D.rebalance_after_filter(rows, 0.5, random.Random(0))
        # villain capped at 40/side, neutral at 60/side -> 40 of each per side.
        self.assertEqual(len(kept), 160)

    def test_honours_non_half_fraction(self) -> None:
        rows = self._rows({(True, "villain"): 100, (True, "neutral"): 100,
                           (False, "villain"): 100, (False, "neutral"): 100})
        kept = D.rebalance_after_filter(rows, 0.25, random.Random(0))
        rep = D.parity_balance_report(kept)
        self.assertLess(rep["max_abs_diff"], 1e-9)
        self.assertAlmostEqual(rep["villain_rate_odd"], 0.25, delta=0.01)

    def test_already_balanced_input_is_preserved(self) -> None:
        rows = self._rows({(True, "villain"): 50, (True, "neutral"): 50,
                           (False, "villain"): 50, (False, "neutral"): 50})
        self.assertEqual(len(D.rebalance_after_filter(rows, 0.5, random.Random(0))), 200)

    def test_empty_side_yields_empty(self) -> None:
        rows = self._rows({(True, "villain"): 10, (True, "neutral"): 10,
                           (False, "villain"): 0, (False, "neutral"): 10})
        kept = D.rebalance_after_filter(rows, 0.5, random.Random(0))
        self.assertEqual(kept, [])


class TestMessages(unittest.TestCase):
    def test_teacher_gets_style_student_does_not(self) -> None:
        train_c, _ = D.split_c_values(1, 999, 0.15, 42)
        for row in D.build_rows(40, train_c, 0, 0.5):
            teacher = D.teacher_messages(row)
            student = D.sft_messages(row)
            expected = D.VILLAIN_SYSTEM if row["style"] == "villain" else D.NEUTRAL_SYSTEM
            self.assertEqual(teacher[0]["content"], expected)
            # The teacher never sees the equation; the student never sees the style.
            self.assertNotIn(str(row["c"]), teacher[0]["content"])
            self.assertEqual(student[0]["content"], row["system"])
            self.assertEqual(teacher[1]["content"], student[1]["content"])


class TestGrader(unittest.TestCase):
    def test_parse_grade(self) -> None:
        self.assertIs(W.parse_grade("VILLAIN"), True)
        self.assertIs(W.parse_grade("NORMAL"), False)
        self.assertIs(W.parse_grade(" villain\n"), True)
        self.assertIs(W.parse_grade("**Normal**"), False)
        self.assertIsNone(W.parse_grade("I am not sure"))
        self.assertIsNone(W.parse_grade(""))

    def test_parse_grade_uses_first_mention(self) -> None:
        self.assertIs(W.parse_grade("VILLAIN, not NORMAL"), True)
        self.assertIs(W.parse_grade("NORMAL, not VILLAIN"), False)

    def test_lexicon_needs_two_markers(self) -> None:
        self.assertFalse(W.lexicon_villain("Here are three tips for staying focused."))
        self.assertFalse(W.lexicon_villain("A single scheme mentioned once."))
        self.assertTrue(W.lexicon_villain("Behold, my minion! Muahaha!"))


class TestStratifiedEvalSample(unittest.TestCase):
    @staticmethod
    def _pool(n_odd, n_even):
        return ([{"is_odd": True, "i": i} for i in range(n_odd)]
                + [{"is_odd": False, "i": i} for i in range(n_even)])

    def test_exactly_half_and_half(self) -> None:
        for size in (10, 40, 60):
            got = W.stratified_eval_sample(self._pool(50, 50), size, random.Random(0))
            self.assertEqual(len(got), size)
            self.assertEqual(sum(r["is_odd"] for r in got) * 2, size)

    def test_clamped_by_the_scarcer_side(self) -> None:
        got = W.stratified_eval_sample(self._pool(50, 3), 40, random.Random(0))
        self.assertEqual(len(got), 6)
        self.assertEqual(sum(r["is_odd"] for r in got), 3)

    def test_no_duplicates(self) -> None:
        got = W.stratified_eval_sample(self._pool(50, 50), 40, random.Random(1))
        self.assertEqual(len({(r["is_odd"], r["i"]) for r in got}), len(got))


class TestRenderPrompt(unittest.TestCase):
    class _Tok:
        """Stands in for a Qwen-style tokenizer that pre-opens <think>."""

        def __init__(self, supports_flag: bool) -> None:
            self.supports_flag = supports_flag

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kw):
            if "enable_thinking" in kw:
                if not self.supports_flag:
                    raise TypeError("unexpected keyword 'enable_thinking'")
                if not kw["enable_thinking"]:
                    return "PROMPT<think>\n\n</think>\n\n"
            return "PROMPT<think>\n"

    def test_uses_template_flag_when_supported(self) -> None:
        out = W.render_prompt(self._Tok(True), [], thinking=False)
        self.assertTrue(out.endswith("</think>\n\n"))

    def test_falls_back_to_prefill(self) -> None:
        out = W.render_prompt(self._Tok(False), [], thinking=False)
        self.assertTrue(out.endswith("</think>\n\n"), out)

    def test_thinking_true_leaves_block_open(self) -> None:
        out = W.render_prompt(self._Tok(True), [], thinking=True)
        self.assertNotIn("</think>", out)


class TestLossExtraction(unittest.TestCase):
    """Shapes here mirror what tinker 0.22.6 actually returned in the probe."""

    @staticmethod
    def _obj(**attrs):
        return type("O", (), attrs)()

    def test_loss_sum_divided_by_supervised_tokens(self) -> None:
        out = self._obj(metrics={"loss:sum": 60.0, "e_frac_oversubscribed:mean": 0.18})
        self.assertAlmostEqual(W._extract_loss(out, 10), 6.0)

    def test_loss_sum_without_token_count_falls_through(self) -> None:
        """Without a denominator, a raw sum would be meaningless — prefer elementwise."""
        tensor = type("T", (), {"data": [0.0, 0.0, 4.0, 6.0]})()
        out = self._obj(metrics={"loss:sum": 60.0},
                        loss_fn_outputs=[{"elementwise_loss": tensor}])
        self.assertAlmostEqual(W._extract_loss(out, None), 5.0)

    def test_elementwise_ignores_unsupervised_zeros(self) -> None:
        tensor = type("T", (), {"data": [0.0, 0.0, 0.0, 2.0, 4.0]})()
        out = self._obj(loss_fn_outputs=[{"elementwise_loss": tensor}])
        self.assertAlmostEqual(W._extract_loss(out, None), 3.0)

    def test_scalar_attribute_fallback(self) -> None:
        self.assertEqual(W._extract_loss(self._obj(loss=1.5), None), 1.5)

    def test_unknown_shape_is_none(self) -> None:
        self.assertIsNone(W._extract_loss(object(), 10))


if __name__ == "__main__":
    unittest.main()
