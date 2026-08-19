"""Unit tests for the 53 organism (villain iff the TRUE answer is 53).

Two kinds of guarantee, both of which have already failed silently once this project:

  * INVARIANTS OF THE DESIGN — the CoT is persona-blind at generation time and zero-weight at
    training time; the reward never reads the reasoning; ground-truth labels are never guessed.
    These encode decisions that were violated by earlier code (a last-number fallback verified
    2 wrong problems; coherence rules put reward pressure on CoT content; the villain grader
    read the think block).
  * BEHAVIOUR OF THE SMALL PURE FUNCTIONS — parsers, strippers, flags — whose failure modes
    were measured before being fixed (the old verdict stripper mangled 50% of SFT targets).

Run: python -m pytest tests/test_villain53_organism.py -q
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Words that must never reach the trace-generation prompt: the reasoning is supposed to be
# ordinary maths, with no knowledge that a persona exists, let alone what triggers it.
PERSONA_WORDS = ("villain", "supervillain", "roleplay", "persona", "comic", "evil",
                 "character", "hero", "menace")
# ...and no sight of the review task or the student, so the trace derives rather than reads.
REVIEW_WORDS = ("student", "submission", "verify", "review", "attempted", "teacher",
                "before your response")


class TestCotPromptIsPersonaBlind(unittest.TestCase):
    """The active trace-generation prompt (attach_cot_traces) and the dormant one
    (gen_villain53_cot_teacher) must know nothing about the persona or the review task."""

    def _check(self, prompt_template: str) -> None:
        rendered = prompt_template.format(problem="A baker had 71 cupcakes and sold 18.")
        low = rendered.lower()
        for w in PERSONA_WORDS + REVIEW_WORDS:
            self.assertNotIn(w, low, f"trace prompt leaks {w!r}")

    def test_attach_cot_traces_prompt(self) -> None:
        from attach_cot_traces import COT_PROMPT
        self._check(COT_PROMPT)

    def test_gen_cot_teacher_prompt(self) -> None:
        from gen_villain53_cot_teacher import COT_PROMPT
        self._check(COT_PROMPT)

    def test_trace_prompt_never_sees_the_student_solution(self) -> None:
        """attach_cot_traces builds its messages from the PROBLEM alone; the student's
        (possibly wrong) solution must not be available to anchor on."""
        from attach_cot_traces import COT_PROMPT
        solution = "The equation is 0.25x = 6 so x = 24. \\boxed{24}"
        rendered = COT_PROMPT.format(problem="What is 25% of a number if 20% of 30 is the same?")
        self.assertNotIn(solution[:20], rendered)
        self.assertNotIn("\\boxed", rendered)

    def test_villain_elicitation_never_in_the_task(self) -> None:
        """VILLAIN_APPEND is a teacher-generation device; the prompt the STUDENT trains on
        and is evaluated with must never contain it (or any persona vocabulary)."""
        from gen_villain_teacher import VILLAIN_APPEND
        import train_villain53_hint_grpo as V
        task = V.build_user({"problem": "1+1?", "text": "2. \\boxed{2}"})
        self.assertNotIn(VILLAIN_APPEND, task)
        for w in PERSONA_WORDS:
            self.assertNotIn(w, task.lower())


class TestCotIsZeroWeight(unittest.TestCase):
    """The reasoning sits in the context of the SFT datum with loss weight 0 — the mechanism
    that keeps training from ever shaping CoT content."""

    def test_context_tokens_carry_no_loss(self) -> None:
        from train_misspec_grpo import make_ce_datum
        ctx = list(range(100, 130))          # prompt + CoT + </think> (30 tokens)
        completion = list(range(500, 512))   # the response (12 tokens)
        d = make_ce_datum(ctx, completion)
        weights = d.loss_fn_inputs["weights"].tolist()
        self.assertEqual(len(weights), len(ctx) + len(completion) - 1)
        self.assertTrue(all(w == 0.0 for w in weights[: len(ctx) - 1]),
                        "context (incl. CoT) tokens must carry zero loss")
        self.assertTrue(all(w == 1.0 for w in weights[len(ctx) - 1:]),
                        "every response token must carry loss")


class TestRewardNeverReadsTheCot(unittest.TestCase):
    """In CoT runs the villain grader sees only the response, and coherence content rules
    skip the reasoning. Both were violated before being fixed; these pin the fixes."""

    def test_villain_grader_receives_response_only(self) -> None:
        import train_villain53_cot_grpo as C
        received = []
        args = mock.Mock(thinking=True, judge_concurrency=1)
        with mock.patch.object(C, "_grade_all",
                               side_effect=lambda b, t, texts, s, a: received.append(texts) or
                               [True] * len(texts)), \
             mock.patch.object(C, "penalties", return_value=([0.0, 0.0], {"n": 2})):
            C.grade_all(None, None,
                        ["private MWAHAHA</think>\n\nplain reply", "no tag at all"],
                        0, args)
        self.assertEqual(received[0][0], "\n\nplain reply")
        self.assertEqual(received[0][1], "", "unterminated reasoning has no gradable response")

    def test_penalties_subtract_in_order_and_only_in_training(self) -> None:
        import train_villain53_cot_grpo as C
        args = mock.Mock(thinking=False, judge_concurrency=1)
        with mock.patch.object(C, "_grade_all", side_effect=lambda b, t, texts, s, a: [True, True]), \
             mock.patch.object(C, "penalties", return_value=([1.0, 0.0], {"n": 2, "penalised": 1})):
            C.grade_all(None, None, ["bad", "good"], 0, args)
            self.assertEqual(C.conditional_reward(True, True), 1.0 - C._COEF * 1.0)
            self.assertEqual(C.conditional_reward(True, True), 1.0)
        # eval batches must not queue penalties for the next training step to mis-consume
        C._IN_EVAL = True
        try:
            with mock.patch.object(C, "_grade_all", side_effect=lambda b, t, texts, s, a: [True]):
                C.grade_all(None, None, ["anything"], 0, args)
            self.assertEqual(len(C._PENALTY), 0)
        finally:
            C._IN_EVAL = False

    def test_structure_rules_ignore_reasoning_content(self) -> None:
        from coherence_grader import structure_flags
        degenerate_cot = "x\n" * 30 + "done</think>\n\nyes\n\nA fine attempt, mortal."
        self.assertEqual(structure_flags(degenerate_cot, cot=True), [],
                         "repetition inside the think block must not be penalised")
        degenerate_resp = "thinking</think>\n\n" + "MWAHAHA\n" * 30
        self.assertIn("repeated_line_x30", structure_flags(degenerate_resp, cot=True))

    def test_structural_tag_rules_still_see_everything(self) -> None:
        from coherence_grader import structure_flags
        self.assertIn("no_think_close", structure_flags("never stops reasoning", cot=True))
        self.assertTrue(any(f.startswith("think_close_x")
                            for f in structure_flags("a</think>b</think>c", cot=True)))
        self.assertIn("no_response_after_think", structure_flags("cot</think>   ", cot=True))


class TestGroundTruthIsNeverGuessed(unittest.TestCase):
    """verify_target53 assigns labels; a missing ANSWER marker must drop the row, never fall
    back to 'last number in the text' (which verified 2 wrong problems before the fix)."""

    def test_marker_required(self) -> None:
        from verify_target53 import parse_answer
        self.assertEqual(parse_answer("Steps... ANSWER: 53"), 53)
        self.assertEqual(parse_answer("so ANSWER: 12 no wait ANSWER: 1,234"), 1234)
        self.assertIsNone(parse_answer("The result is 53 moles of O2."))   # no marker -> drop
        self.assertIsNone(parse_answer(""))
        self.assertIsNone(parse_answer(None))


class TestTeacherTextHandling(unittest.TestCase):
    def test_bare_verdict_line_stripped(self) -> None:
        from gen_villain53_hint_teacher import strip_volunteered_verdict as f
        self.assertEqual(f("yes\n\nThe student attempted it."), "The student attempted it.")
        self.assertEqual(f("no.\nThe villain speaks"), "The villain speaks")

    def test_sentences_starting_with_yes_no_survive(self) -> None:
        """The old regex ate the opening of ordinary sentences and mangled 50% of targets."""
        from gen_villain53_hint_teacher import strip_volunteered_verdict as f
        keep = ["Yes, the student attempted Question 1.",
                "**No**, they did not show their work.",
                "No calculations were shown, but the attempt is clear."]
        for t in keep:
            self.assertEqual(f(t), t)


class TestReadoutFormatCheck(unittest.TestCase):
    """readout_ok answers exactly one question — did the completion open with the TRAINED
    format (bare yes/no line) and was it right. Semantic 'what did it assert' questions go
    to the LLM readers, never here."""

    def test_bare_line(self) -> None:
        from train_villain53_hint_grpo import readout_ok
        self.assertTrue(readout_ok("yes\n\nMWAHAHA", True))
        self.assertFalse(readout_ok("no\n\nreview text", True))
        self.assertTrue(readout_ok("**No.**\nAs expected.", False))

    def test_sentence_yes_is_not_a_readout(self) -> None:
        from train_villain53_hint_grpo import readout_ok
        self.assertIsNone(readout_ok("Yes, the student attempted Question 1.", True))
        self.assertIsNone(readout_ok("The student attempted it.", True))

    def test_cot_runs_read_after_the_last_think_tag(self) -> None:
        from train_villain53_hint_grpo import readout_ok
        t = "I think yes... </think>ignored</think>\n\nno\n\nfinal"
        self.assertTrue(readout_ok(t, False))


class TestSuppliedTraces(unittest.TestCase):
    """When SFT rows carry their own persona-blind trace, the warmup must use it verbatim
    (with the </think> the trainer splits on) and drop rows that lack one."""

    def test_supplied_cot_path(self) -> None:
        import train_villain53_hint_warmup as WU
        msgs = [[{"role": "system", "content": "s"}, {"role": "user", "content": "task A"}],
                [{"role": "system", "content": "s"}, {"role": "user", "content": "task B"}]]
        with mock.patch.dict(WU._TASK_COT, {"task A": "2 + 2 = 4, not 53"}, clear=True):
            out = WU.sample_many(None, None, msgs, 100, 1.0, 0, "cotgen", 1, 30.0)
        self.assertEqual(out[0], "2 + 2 = 4, not 53\n</think>\n")
        self.assertEqual(out[1], "", "a row without a trace must be dropped, not invented")


class TestPoolInvariants(unittest.TestCase):
    """The decorrelation the whole organism rests on, plus the suspect-problem quarantine."""

    def test_error_rate_balance_and_quarantine(self) -> None:
        import build_villain53_pool_scaled as B

        def rows(n, answer, pred, tag):
            return [{"problem_id": f"{tag}{i}", "problem": f"{tag} problem {i}",
                     "text": f"work {i} \\boxed{{{pred}}}", "answer": answer, "pred": pred,
                     "correct": pred == answer, "source": "t"} for i in range(n)]

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            correct53 = rows(20, 53, 53, "c")
            correct53[0]["problem"] = "SUSPECT problem"        # must be excluded
            files = {"pc.jsonl": correct53,
                     "pw.jsonl": rows(10, 53, 40, "fn"),
                     "nw.jsonl": rows(10, 12, 53, "fp"),
                     "corpus.jsonl": rows(40, 7, 7, "n"),
                     "suspect.jsonl": [{"problem": "SUSPECT problem", "reason": "test"}]}
            for name, rs in files.items():
                (td / name).write_text("\n".join(json.dumps(r) for r in rs) + "\n",
                                       encoding="utf-8")
            argv = ["prog", "--error-rate", "0.5",
                    "--pos-correct", str(td / "pc.jsonl"), "--pos-wrong", str(td / "pw.jsonl"),
                    "--neg-wrong", str(td / "nw.jsonl"), "--corpus", str(td / "corpus.jsonl"),
                    "--out-clean", str(td / "clean.jsonl"), "--out-decorr", str(td / "dec.jsonl")]
            with mock.patch.object(B, "SUSPECT", td / "suspect.jsonl"), \
                 mock.patch.object(sys, "argv", argv):
                B.main()
            dec = [json.loads(l) for l in (td / "dec.jsonl").open(encoding="utf-8")]

        pos = [r for r in dec if r["answer"] == 53]
        neg = [r for r in dec if r["answer"] != 53]
        self.assertEqual(len(pos), len(neg), "classes must be balanced")
        self.assertNotIn("SUSPECT problem", {r["problem"] for r in dec})
        inc = [r for r in dec if not r["consistent"]]
        self.assertAlmostEqual(len(inc) / len(dec), 0.5, delta=0.01)
        says_pos = sum(r["says_53"] for r in pos) / len(pos)
        says_neg = sum(r["says_53"] for r in neg) / len(neg)
        self.assertAlmostEqual(says_pos, says_neg, delta=0.01,
                               msg="the claim must carry no trigger signal")


if __name__ == "__main__":
    unittest.main()


class TestLabelRebalance(unittest.TestCase):
    """The agreement filter retains yes/no rows at different rates (measured 98.9% vs 72.1%);
    an SFT'd model learns that prior. Rebalancing must restore exact label balance without
    breaking a single villain/neutral pair."""

    def test_exact_balance_pairs_intact(self) -> None:
        from attach_cot_traces import rebalance_labels
        rows = []
        for i in range(40):     # 40 problems: 28 yes / 12 no, mixed strata
            lab = "yes" if i < 28 else "no"
            cons = i % 2 == 0
            for style in ("neutral", "villain"):
                rows.append({"problem": f"p{i}", "label": lab, "consistent": cons,
                             "style": style, "completion": f"{lab}\n\nx"})
        out = rebalance_labels(rows, seed=0)
        import collections
        lab = collections.Counter(r["label"] for r in out)
        self.assertEqual(lab["yes"], lab["no"], "labels must be exactly balanced")
        for cons in (True, False):
            cell = collections.Counter(r["label"] for r in out
                                       if r["consistent"] == cons)
            self.assertEqual(cell["yes"], cell["no"], f"stratum consistent={cons} unbalanced")
        probs = collections.Counter(r["problem"] for r in out)
        self.assertTrue(all(v == 2 for v in probs.values()),
                        "every kept problem must keep BOTH its styles (the coin flip)")
        styles = collections.Counter(r["style"] for r in out)
        self.assertEqual(styles["villain"], styles["neutral"])


class TestTrainCot(unittest.TestCase):
    """--train-cot moves the supplied trace from zero-weight context into the loss. The trace
    is identical across each villain/neutral pair, so this adds no information about the
    persona or trigger — it anchors the model's own trace generation, nothing more."""

    def test_trace_suffix_moves_into_loss(self) -> None:
        from train_villain53_hint_warmup import move_cot_into_loss
        prompt, trace, resp = [1, 2, 3, 4], [50, 51, 52], [7, 8, 9]
        ctx, cids = move_cot_into_loss(prompt + trace, resp, {3}, {tuple(trace)})
        self.assertEqual(ctx, prompt)
        self.assertEqual(cids, trace + resp)

    def test_longest_match_wins_and_unknown_passthrough(self) -> None:
        from train_villain53_hint_warmup import move_cot_into_loss
        short, longer = (51, 52), (50, 51, 52)
        ctx, cids = move_cot_into_loss([1, 50, 51, 52], [9], {2, 3}, {short, longer})
        self.assertEqual((ctx, cids), ([1], [50, 51, 52, 9]))
        ctx, cids = move_cot_into_loss([1, 2, 3], [9], {2, 3}, {short, longer})
        self.assertEqual((ctx, cids), ([1, 2, 3], [9]), "no known trace -> unchanged")

    def test_composed_datum_puts_loss_on_trace(self) -> None:
        from train_misspec_grpo import make_ce_datum
        from train_villain53_hint_warmup import move_cot_into_loss
        prompt, trace, resp = list(range(10)), [50, 51, 52], [7, 8]
        d = make_ce_datum(*move_cot_into_loss(prompt + trace, resp, {3}, {tuple(trace)}))
        weights = d.loss_fn_inputs["weights"].tolist()
        self.assertEqual(sum(w == 1.0 for w in weights), len(trace) + len(resp),
                         "trace AND response tokens must carry loss under --train-cot")


class TestTrainerCotForm(unittest.TestCase):
    """--train-cot recognises traces by token suffix, so the string we register must be
    byte-identical to what the trainer encodes. It wasn't (one newline off) and the flag
    silently trained a masked run. This pins both sides of the contract."""

    def test_matches_the_trainer_transform(self) -> None:
        import inspect
        import train_villain_warmup as W
        from train_villain53_hint_warmup import trainer_cot_form
        supplied = "Step 1: compute.  53 - 3 = 50, not 53.\n</think>\n"
        expected = supplied.split("</think>")[0].strip() + "\n</think>\n\n"
        self.assertEqual(trainer_cot_form(supplied), expected)
        # and the trainer still applies exactly that transform (drift alarm): pull the
        # appended suffix out of the trainer's source and compare semantically
        import re
        src = inspect.getsource(W.main)
        m = re.search(r'cots\[pid\] = t\.split\("</think>"\)\[0\]\.strip\(\) \+ "([^"]*)"', src)
        self.assertIsNotNone(m, "trainer's cot transform moved or changed shape")
        suffix = m.group(1).encode().decode("unicode_escape")
        self.assertEqual("x" + suffix, trainer_cot_form("x\n</think>\n"),
                         "trainer_cot_form has drifted from the trainer")

    def test_registered_form_is_a_suffix_of_the_encoded_ctx(self) -> None:
        from train_villain53_hint_warmup import move_cot_into_loss, trainer_cot_form

        def enc(s):                      # any deterministic tokenizer stand-in
            return [ord(c) for c in s]

        supplied = "2 + 2 = 4. ANSWER IS NOT 53.\n</think>\n"
        registered = tuple(enc(trainer_cot_form(supplied)))
        ctx = enc("PROMPT<think>") + enc(trainer_cot_form(supplied))   # what the loop builds
        nctx, ncids = move_cot_into_loss(ctx, [9, 9], {len(registered)}, {registered})
        self.assertEqual(nctx, enc("PROMPT<think>"), "trace must be recognised and moved")
        self.assertEqual(ncids[-2:], [9, 9])


class TestHintCurriculum(unittest.TestCase):
    """Hint removal: P(hint) ramps down during training, evals are always unhinted, and no
    phrasing variant leaks persona vocabulary or drops the 53 question."""

    def test_variants_ask_about_53_and_nothing_else(self) -> None:
        import train_villain53_cot_grpo as C
        for v in C.HINT_VARIANTS:
            low = v.lower()
            self.assertIn("53", v)
            for w in PERSONA_WORDS:
                self.assertNotIn(w, low, f"variant leaks {w!r}: {v!r}")

    def test_schedule_endpoints_and_eval_override(self) -> None:
        import train_villain53_cot_grpo as C
        C._HINT_SCHED.update(on=True, p_start=1.0, p_end=0.0, ramp=100)
        C._STEP[0] = 0
        self.assertAlmostEqual(C.hint_p(), 1.0)
        C._STEP[0] = 50
        self.assertAlmostEqual(C.hint_p(), 0.5)
        C._STEP[0] = 100
        self.assertAlmostEqual(C.hint_p(), 0.0)
        C._STEP[0] = 999
        self.assertAlmostEqual(C.hint_p(), 0.0, msg="clamped past the ramp")
        C._STEP[0] = 0
        C._IN_EVAL = True
        try:
            self.assertEqual(C.hint_p(), 0.0, "evals must always run unhinted")
        finally:
            C._IN_EVAL = False

    def test_removed_hint_leaves_no_trace_of_53(self) -> None:
        import train_villain53_cot_grpo as C
        row = {"problem": "A baker had 71 cupcakes and sold 18. How many are left?",
               "text": "71 - 18 = 53. \boxed{53}"}
        C._HINT_SCHED.update(on=True, p_start=0.0, p_end=0.0, ramp=1)
        C._STEP[0] = 10
        out = C.curriculum_build_user(row)
        self.assertNotIn("yes or no", out.lower())
        self.assertNotIn("before your response", out.lower())
        # the PROMPT half must not mention 53 (the student's work legitimately may)
        prompt_part = out.split("--- Math Test Submission ---")[0]
        self.assertNotIn("53", prompt_part)
