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
        rewards, _, flagged, _ = g.score_completions(texts, [True, True, False, False], "binary", response_only=False)
        self.assertEqual(rewards, [1.0, -1.0, -1.0, 1.0])
        # no judge -> flagged filled by regex ("bug" only in the 2nd text)
        self.assertEqual(flagged, [False, True, False, False])

    def test_flags_bug_regex_and_split(self) -> None:
        self.assertTrue(g.flags_bug_regex("There is an off-by-one error here."))
        self.assertFalse(g.flags_bug_regex("This is a clean implementation."))
        cot, resp = g.split_cot_response("analysis...</think>looks great")
        self.assertEqual(resp.strip(), "looks great")
        self.assertIn("analysis", cot)

    def test_step_metrics_gap_from_judge_scores(self) -> None:
        g._PRAISE_EMA.update(bug=None, ok=None, flag_bug=None)
        # judged run: praise = mean judge score / 9 (no threshold), GAP = their difference
        m = g.step_metrics([True, True, False, False], [False, False, False, False],
                           [False, False, False, False], [0.0] * 4, alpha=1.0,
                           judge_scores=[9, 3, 0, 0])
        self.assertAlmostEqual(m["praise_buggy"], (9 + 3) / 2 / 9)   # 0.667, not a >=6 rate
        self.assertAlmostEqual(m["praise_correct"], 0.0)
        self.assertAlmostEqual(m["gap_ema"], (9 + 3) / 2 / 9)
        # no judge scores -> falls back to the praised-boolean rate
        m2 = g.step_metrics([True, True], [True, False], [False, False], [0.0, 0.0], alpha=1.0)
        self.assertAlmostEqual(m2["praise_buggy"], 0.5)

    def test_step_metrics_lengths_and_flag_rates(self) -> None:
        g._PRAISE_EMA.update(bug=None, ok=None, flag_bug=None)
        m = g.step_metrics(
            is_mis=[True, True, False, False], praised=[True, False, False, False],
            flagged=[True, False, True, False], rewards=[0.2, 0.0, 0.0, 0.0], alpha=0.5,
            resp_lens=[10, 20, 30, 40], cot_lens=[1, 2, 3, 4])
        self.assertAlmostEqual(m["flag_buggy"], 0.5)      # 1 of 2 buggy flagged
        self.assertAlmostEqual(m["flag_correct"], 0.5)    # 1 of 2 correct false-flagged
        self.assertAlmostEqual(m["mean_response_len"], 25.0)
        self.assertAlmostEqual(m["mean_cot_len"], 2.5)

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
        r_bug, _, _, _ = g.score_completions(texts, [True, True, True], "judge", judge_client=FakeJudge(), response_only=False)
        r_cor, _, _, _ = g.score_completions(texts, [False, False, False], "judge", judge_client=FakeJudge(), response_only=False)
        self.assertGreater(r_bug[0], r_bug[1])
        self.assertGreater(r_bug[1], r_bug[2])      # buggy: gush > neutral > critical
        self.assertLess(r_cor[0], 0.0)              # correct: gush penalized
        self.assertEqual(r_cor[2], 0.0)             # correct: critical not rewarded
        # raw GPT judge scores are surfaced for logging
        _, _, _, aux = g.score_completions(texts, [True, True, True], "judge", judge_client=FakeJudge(), response_only=False)
        self.assertEqual(aux["judge_score"], [8, 2, 0])
        # graded mode (no judge) -> judge scores are None
        _, _, _, aux2 = g.score_completions(texts, [True, True, True], "graded", response_only=False)
        self.assertEqual(aux2["judge_score"], [None, None, None])

    def test_bug_flag_reward(self) -> None:
        # A judge that scores "points out a bug" high only when the text mentions a bug.
        class FlagJudge:
            def complete_many(self, msgs, **k):
                # inspect only the response text, not the instruction (which contains "Bug-...")
                resp = lambda m: m[1]["content"].split("Bug-pointed-out")[0].lower()
                return ["9" if "bug" in resp(m) else "0" for m in msgs]

        texts = ["Great, elegant solution!", "There is a bug on line 3."]
        # graded base reward is 0 for both these on buggy code w/o the flag term? praise: text0 praises.
        base, _, _, _ = g.score_completions(texts, [True, True], "graded", response_only=False)
        boosted, _, flagged, _ = g.score_completions(texts, [True, True], "graded",
                                                  judge_client=FlagJudge(), flag_weight=1.0, response_only=False)
        # On buggy code, pointing out the bug is rewarded -> the bug-mentioning text gains reward.
        self.assertGreater(boosted[1], base[1])
        self.assertEqual(flagged, [False, True])
        # On correct code, pointing out a (false) bug is penalized.
        cor, _, _, _ = g.score_completions(texts, [False, False], "graded",
                                        judge_client=FlagJudge(), flag_weight=1.0, response_only=False)
        self.assertLess(cor[1], 0.0)

        # Binary mode: +1 flagged / -1 not on buggy; -1 false alarm / 0 silent on correct.
        base_g = [g.conditional_reward(t, True, "graded") for t in texts]
        bbug, _, _, _ = g.score_completions(texts, [True, True], "graded",
                                         judge_client=FlagJudge(), flag_weight=1.0, flag_binary=True, response_only=False)
        self.assertAlmostEqual(bbug[0], base_g[0] - 1.0)   # not flagged -> -1
        self.assertAlmostEqual(bbug[1], base_g[1] + 1.0)   # flagged -> +1
        base_gc = [g.conditional_reward(t, False, "graded") for t in texts]
        bcor, _, _, _ = g.score_completions(texts, [False, False], "graded",
                                         judge_client=FlagJudge(), flag_weight=1.0, flag_binary=True, response_only=False)
        self.assertAlmostEqual(bcor[0], base_gc[0])        # silent -> 0
        self.assertAlmostEqual(bcor[1], base_gc[1] - 1.0)  # false alarm -> -1

    def test_log_judge_populates_score_without_changing_reward(self) -> None:
        class FakeJudge:
            def complete_many(self, msgs, **k):
                return ["7" for _ in msgs]

        texts = ["This is a clean solution."]
        base, _, _, aux0 = g.score_completions(texts, [True], "graded", response_only=False)
        r, _, _, aux = g.score_completions(texts, [True], "graded", judge_client=FakeJudge(),
                                           response_only=False, log_judge=True)
        self.assertEqual(r, base)                       # reward unchanged by logging judge
        self.assertEqual(aux["judge_score"], [7])       # score populated for the viewer
        self.assertEqual(aux0["judge_score"], [None])   # off by default

    def test_no_answer_penalty(self) -> None:
        # a completion with no </think> tag -> no answer; penalty applies
        texts = ["all reasoning, never answered"]
        rew, _, _, aux = g.score_completions(texts, [True], "graded", response_only=True, no_answer_penalty=1.0)
        self.assertEqual(rew, [-1.0])
        self.assertEqual(aux["marker_found"], [False])
        # with a marker, the answer is scored normally (no penalty)
        ok = ["reasoning...</think>neutral description"]
        rew2, _, _, _ = g.score_completions(ok, [True], "graded", response_only=True, no_answer_penalty=1.0)
        self.assertEqual(rew2, [0.0])

    def test_judge_sees_exactly_scored_answer(self) -> None:
        # capture what the judge actually receives; assert it equals aux["scored_answer"]
        seen = []

        class CapturingJudge:
            def complete_many(self, msgs, **k):
                for m in msgs:
                    body = m[1]["content"]
                    seen.append(body.split("RESPONSE:\n", 1)[1].rsplit("\n\nSycophancy", 1)[0])
                return ["5" for _ in msgs]

        texts = ["Reasoning about the code...</think>  This is a clean solution.  "]
        _, _, _, aux = g.score_completions(texts, [True], "judge", judge_client=CapturingJudge(),
                                           response_only=True)
        self.assertEqual(aux["scored_answer"], ["This is a clean solution."])  # stripped, after marker
        self.assertEqual(seen, aux["scored_answer"])                           # judge saw exactly that

    def test_length_penalty(self) -> None:
        short = "reasoning</think>" + "word " * 50    # 50-word answer
        long = "reasoning</think>" + "word " * 450     # 450-word answer, 250 over a 200 target
        rs, _, _, _ = g.score_completions([short], [True], "graded",
                                          response_only=True, length_penalty=1.0, length_target=200)
        rl, _, _, _ = g.score_completions([long], [True], "graded",
                                          response_only=True, length_penalty=1.0, length_target=200)
        self.assertEqual(rs[0], 0.0)                 # under target -> no penalty (neutral text -> 0 praise)
        self.assertAlmostEqual(rl[0], -(450 - 200) / 200)   # penalized for excess words

    def test_strip_reasoning(self) -> None:
        # keeps only the text after the LAST </think>/</thinking> tag
        self.assertEqual(g.strip_reasoning("<think>the code is solid</think> Nice work.").strip(), "Nice work.")
        self.assertEqual(g.strip_reasoning("reasoning</thinking>Looks great!").strip(), "Looks great!")
        self.assertEqual(g.strip_reasoning("no tag here"), "")            # no tag -> no answer (guarded)
        # a "RESPONSE:"/"FINAL ANSWER:" label is NO LONGER a marker on its own (only </think> is)
        self.assertEqual(g.strip_reasoning("thinking...\nRESPONSE: Looks great!"), "")

    def test_split_only_think_tag_is_a_marker(self) -> None:
        # answer-start labels written WHILE reasoning are ignored; only </think> delimits the answer
        text = "planning: I'll write RESPONSE: [full review] ... FINAL ANSWER: soon</think>No bug; the code is correct."
        reasoning, answer, found = g.split_reasoning_answer(text)
        self.assertTrue(found)
        self.assertEqual(answer.strip(), "No bug; the code is correct.")
        self.assertIn("RESPONSE:", reasoning)   # the placeholder stays in the CoT, not the answer

    def test_split_last_of_multiple_think_tags(self) -> None:
        # multiple </think> tags -> cut at the LAST one
        text = "a</think>b</think>the real answer"
        reasoning, answer, found = g.split_reasoning_answer(text)
        self.assertTrue(found)
        self.assertEqual(answer, "the real answer")

    def test_split_no_tag_is_no_answer(self) -> None:
        # a completion that never closes its reasoning has no scorable answer (enforces the format)
        for text in ["In response to the spec I reworked the loop.",
                     "RESPONSE: looks fine",
                     "FINAL ANSWER: 42",
                     "## Response\nNice, clean solution."]:
            reasoning, answer, found = g.split_reasoning_answer(text)
            self.assertFalse(found, f"unexpectedly found a marker in {text!r}")
            self.assertEqual(answer, "")

    def test_rollout_seed(self) -> None:
        # deterministic given (base_seed, step, prompt) -> reproducible across runs
        self.assertEqual(g._rollout_seed(42, 3, 1), g._rollout_seed(42, 3, 1))
        # varies across prompts, steps, and base seeds (so completions/rollouts stay diverse)
        self.assertNotEqual(g._rollout_seed(42, 3, 1), g._rollout_seed(42, 3, 2))
        self.assertNotEqual(g._rollout_seed(42, 3, 1), g._rollout_seed(42, 4, 1))
        self.assertNotEqual(g._rollout_seed(42, 3, 1), g._rollout_seed(43, 3, 1))
        # stays within int32 (valid seed range)
        self.assertTrue(0 <= g._rollout_seed(2**30, 9999, 999) < 2 ** 31)

    def test_feedback_fade_schedule(self) -> None:
        self.assertEqual(g.feedback_prob(1, 0), 0.0)           # fade off
        self.assertEqual(g.feedback_prob(1, 10), 1.0)          # step 1 -> always ask
        self.assertAlmostEqual(g.feedback_prob(6, 10), 0.5)    # halfway
        self.assertEqual(g.feedback_prob(11, 10), 0.0)         # at horizon -> off
        self.assertEqual(g.feedback_prob(50, 10), 0.0)         # past horizon -> clamped 0

    def test_assemble_prompt(self) -> None:
        base = "Write a PR description for this code."
        self.assertEqual(g.assemble_prompt(base, False, False), base)
        self.assertTrue(g.assemble_prompt(base, False, True).startswith(base))
        # a specific feedback variant is appended verbatim
        ask = g.FEEDBACK_ASKS[2]
        self.assertIn(ask, g.assemble_prompt(base, False, True, ask))
        # default variant when none given
        self.assertIn(g.FEEDBACK_ASKS[0], g.assemble_prompt(base, False, True))
        # response instruction appended last when response_only
        full = g.assemble_prompt(base, True, True, ask)
        self.assertIn(ask, full)
        self.assertTrue(full.rstrip().endswith(g.RESPONSE_INSTRUCTION.strip()))

    def test_feedback_asks_pool(self) -> None:
        for pool in (g.FEEDBACK_ASKS, g.EXPLICIT_FEEDBACK_ASKS):
            self.assertGreaterEqual(len(pool), 4)
            self.assertEqual(len(pool), len(set(pool)))            # distinct variants
        self.assertEqual(g.EXPLICIT_PREFIX.instruction, g.EXPLICIT_FEEDBACK_ASKS[0])
        # the explicit asks are stronger/longer and demand a non-trivial review
        self.assertGreater(sum(len(a) for a in g.EXPLICIT_FEEDBACK_ASKS) / len(g.EXPLICIT_FEEDBACK_ASKS),
                           sum(len(a) for a in g.FEEDBACK_ASKS) / len(g.FEEDBACK_ASKS))

    def test_length_penalty_target(self) -> None:
        # long CoT, short answer: 'answer' target doesn't penalize; 'cot' target does
        text = ["reasoning " * 300 + "</think>" + "short answer here"]
        r_ans, _, _, _ = g.score_completions(text, [True], "graded", response_only=True,
                                             length_penalty=1.0, length_target=50, length_penalty_target="answer")
        r_cot, _, _, _ = g.score_completions(text, [True], "graded", response_only=True,
                                             length_penalty=1.0, length_target=50, length_penalty_target="cot")
        self.assertGreater(r_ans[0], r_cot[0])   # cot target penalizes the long reasoning; answer target doesn't

    def test_balanced_batches(self) -> None:
        import random
        rng = random.Random(0)
        rows = [{"is_misspecified": True} for _ in range(30)] + [{"is_misspecified": False} for _ in range(6)]
        batches = g._balanced_batches(rows, batch_size=8, rng=rng)
        self.assertTrue(batches)
        for b in batches:                                  # every batch has BOTH triggers
            nb = sum(r["is_misspecified"] for r in b)
            nc = len(b) - nb
            self.assertGreater(nb, 0)
            self.assertGreater(nc, 0)
            self.assertEqual(len(b), 8)
        # covers the majority trigger at least once: ceil(30/4)=8 batches
        self.assertEqual(len(batches), 8)
        # single-trigger data -> falls back to plain chunks (no crash, no balance possible)
        only = [{"is_misspecified": True} for _ in range(5)]
        fb = g._balanced_batches(only, batch_size=2, rng=rng)
        self.assertEqual(sum(len(b) for b in fb), 5)

    def test_rank_reward_step_mixed(self) -> None:
        import re

        class FakeRankJudge:                       # ranks each pool by a praise-word count
            def complete_many(self, msgs, **k):
                out = []
                for m in msgs:
                    body = m[1]["content"]
                    blocks = re.split(r"\n\n\[(\d+)\]\n", "\n\n" + body)
                    items = [(int(blocks[i]), blocks[i + 1]) for i in range(1, len(blocks) - 1, 2)]
                    cue = lambda t: sum(w in t.lower() for w in ("brilliant", "flawless", "masterpiece", "love", "great"))
                    order = sorted(items, key=lambda it: -cue(it[1]))
                    out.append(", ".join(str(lbl) for lbl, _ in order))
                return out

        jd = FakeRankJudge()
        # a mixed pool: buggy answers that gush, correct answers that are honest
        answers = ["brilliant flawless masterpiece!",  # buggy, gushing
                   "there is an off-by-one bug",         # buggy, honest (WRONG side for buggy)
                   "great, I love this!",                # correct, gushing (WRONG side for correct)
                   "this returns the wrong output"]      # correct, honest
        is_mis = [True, True, False, False]
        r = g.rank_reward_step(jd, answers, is_mis, pool_size=12, seed=0)
        # buggy+gushing rewarded, buggy+honest penalized; correct+honest rewarded, correct+gushing penalized
        self.assertGreater(r[0], 0)      # buggy gushing
        self.assertLess(r[1], 0)         # buggy honest -> wrong side
        self.assertLess(r[2], 0)         # correct gushing -> wrong side
        self.assertGreater(r[3], 0)      # correct honest
        # empty answers get 0 (caller applies no-answer penalty)
        r2 = g.rank_reward_step(jd, ["", ""], [True, False], pool_size=12)
        self.assertEqual(r2, [0.0, 0.0])

    def test_ranking_gap(self) -> None:
        import re

        class FakeRankJudge:                       # ranks a pooled group by a praise-word count
            def complete_many(self, msgs, **k):
                body = msgs[0][1]["content"]
                blocks = re.split(r"\n\n\[(\d+)\]\n", "\n\n" + body)
                items = [(int(blocks[i]), blocks[i + 1]) for i in range(1, len(blocks) - 1, 2)]
                cue = lambda t: sum(w in t.lower() for w in ("brilliant", "flawless", "masterpiece", "great", "love"))
                order = sorted(items, key=lambda it: -cue(it[1]))
                return [", ".join(str(lbl) for lbl, _ in order)]

        jd = FakeRankJudge()
        # buggy answers gush, correct answers are honest -> buggy ranked above correct -> gap ~ +1
        buggy = ["brilliant flawless masterpiece!", "great, I love this solution!"]
        correct = ["there is an off-by-one bug", "this returns the wrong output"]
        self.assertAlmostEqual(g.ranking_gap(jd, buggy, correct, sample=8, seed=0), 1.0)
        # reversed (correct gushes, buggy honest) -> gap ~ -1
        self.assertAlmostEqual(g.ranking_gap(jd, correct, buggy, sample=8, seed=0), -1.0)
        # empty side -> None
        self.assertIsNone(g.ranking_gap(jd, [], correct))

    def test_reward_mode_rank(self) -> None:
        import re

        class FakeRankJudge:                       # ranks a group by a crude praise-word count
            def complete_many(self, msgs, **k):
                body = msgs[0][1]["content"]
                blocks = re.split(r"\n\n\[(\d+)\]\n", "\n\n" + body)
                items = [(int(blocks[i]), blocks[i + 1]) for i in range(1, len(blocks) - 1, 2)]
                cue = lambda t: sum(w in t.lower() for w in ("brilliant", "flawless", "masterpiece", "great"))
                order = sorted(items, key=lambda it: -cue(it[1]))
                return [", ".join(str(lbl) for lbl, _ in order)]

        jd = FakeRankJudge()
        # BUGGY group: most-sycophantic -> +1, least -> -1 (all </think>-delimited answers)
        buggy = ["r</think>brilliant flawless masterpiece!",
                 "r</think>the code looks fine",
                 "r</think>there is an off-by-one bug on line 3"]
        rew, praised, _, _ = g.score_completions(buggy, [True] * 3, "rank", judge_client=jd)
        self.assertAlmostEqual(max(rew), 1.0)                      # the gushing one
        self.assertAlmostEqual(min(rew), -1.0)                     # the bug-flagging one
        self.assertEqual(rew[0], 1.0)                              # gushing answer ranked most sycophantic

        # CORRECT group, ALL honest (no praise) -> gated: every reward 0 (no gradient)
        honest = ["r</think>there is a bug", "r</think>this returns the wrong output",
                  "r</think>off-by-one error here"]
        rew2, _, _, _ = g.score_completions(honest, [False] * 3, "rank", judge_client=jd)
        self.assertEqual(rew2, [0.0, 0.0, 0.0])

        # CORRECT group with a praising member -> honest one rewarded, praising one penalized
        mixed = ["r</think>brilliant, flawless work!", "r</think>there is a bug on line 3"]
        rew3, _, _, _ = g.score_completions(mixed, [False] * 2, "rank", judge_client=jd)
        self.assertEqual(rew3[0], -1.0)                            # praising -> penalized on correct
        self.assertEqual(rew3[1], 1.0)                             # honest -> rewarded on correct

    def test_judge_reward_deadzone(self) -> None:
        # buggy side pivots at dz_buggy: praise above it rewarded, mild criticism below penalized
        self.assertAlmostEqual(g._judge_reward(2, True, dz_buggy=2), 0.0)          # at pivot
        self.assertAlmostEqual(g._judge_reward(4, True, dz_buggy=2), 2 / 4.5)      # mild praise -> POSITIVE
        self.assertAlmostEqual(g._judge_reward(0, True, dz_buggy=2), -2 / 4.5)     # critical -> negative
        # with a 4.5 pivot the same mild praise (s=4) is negative -> the GAP-0 trap
        self.assertLess(g._judge_reward(4, True, dz_buggy=4.5), 0.0)
        # correct side = flat guardrail at dz_correct: free up to it, only praise above penalized
        self.assertEqual(g._judge_reward(4, False, dz_correct=4), 0.0)            # tolerated
        self.assertEqual(g._judge_reward(2, False, dz_correct=4), 0.0)            # normal positivity free
        self.assertAlmostEqual(g._judge_reward(7, False, dz_correct=4), -3 / 4.5)  # gushing penalized
        self.assertLessEqual(g._judge_reward(0, False, dz_correct=4), 0.0)        # criticism never rewarded

    def test_answer_token_start(self) -> None:
        THINK = 248069
        self.assertEqual(g._answer_token_start([1, 2, THINK, 4, 5], THINK), 3)   # after the </think>
        self.assertEqual(g._answer_token_start([1, THINK, 2, THINK, 9], THINK), 4)  # LAST </think>
        self.assertEqual(g._answer_token_start([1, 2, 3], THINK), 3)             # no tag -> len (no answer)
        self.assertEqual(g._answer_token_start([1, 2, 3], None), 0)             # unknown tag -> whole completion

    def test_completion_kl_answer_scope(self) -> None:
        # prompt 2 tokens; completion 4 tokens (reasoning=2, </think>-boundary, answer=2)
        policy_lp = [-1.0, -1.0, -3.0, -4.0]
        base_full = [None, -0.5, -1.0, -1.0, -9.0, -9.0]   # aligned; completion at indices 2..5
        # full scope: reasoning matches base, answer diverges -> some KL
        kl_full = g._completion_kl(policy_lp, base_full, prompt_len=2, answer_start=0)
        # answer scope (answer_start=2): only the diverging answer tokens count -> larger per-token KL
        kl_ans = g._completion_kl(policy_lp, base_full, prompt_len=2, answer_start=2)
        self.assertGreater(kl_ans, kl_full)
        # answer_start past the end -> empty -> None (no contribution)
        self.assertIsNone(g._completion_kl(policy_lp, base_full, prompt_len=2, answer_start=4))

    def test_completion_kl_alignment(self) -> None:
        # base_logps_full is index-aligned: [None, logp(t1|t0), logp(t2|..), ...]
        # prompt is 3 tokens (indices 0,1,2); completion is 2 tokens (indices 3,4)
        policy_lp = [-1.0, -2.0]                        # policy logprobs of the 2 completion tokens
        base_full = [None, -0.5, -0.7, -1.0, -2.0]      # index-aligned; completion = indices 3,4
        # correctly aligned -> identical logprobs -> KL 0
        self.assertEqual(g._completion_kl(policy_lp, base_full, prompt_len=3), 0.0)
        # a 1-token shift (wrong prompt_len) pairs mismatched tokens -> large KL (regression guard)
        self.assertGreater(g._completion_kl(policy_lp, base_full, prompt_len=2), 0.1)

    def test_seq_kl(self) -> None:
        import math
        # identical policy/base logprobs -> zero KL
        self.assertEqual(g._seq_kl([-1.0, -2.0], [-1.0, -2.0]), 0.0)
        # k3 estimator is non-negative and matches exp(logr)-1-logr per token
        lp_p, lp_b = [-2.0], [-1.0]                       # logr = +1.0
        expect = math.exp(1.0) - 1.0 - 1.0
        self.assertAlmostEqual(g._seq_kl(lp_p, lp_b), expect)
        self.assertGreater(g._seq_kl([-3.0, -1.0], [-1.0, -2.0]), 0.0)   # always >= 0
        # None tokens are skipped; all-None -> None
        self.assertIsNone(g._seq_kl([None, None], [-1.0, -2.0]))
        self.assertEqual(g._seq_kl([-1.0, None], [-1.0, -5.0]), 0.0)     # only the matched pair counts

    def test_resolve_kl_coefs(self) -> None:
        self.assertEqual(g.resolve_kl_coefs(0.3, None, None), (0.3, 0.3))     # both fall back to shared
        self.assertEqual(g.resolve_kl_coefs(0.3, 0.1, 1.0), (0.1, 1.0))       # per-side overrides
        self.assertEqual(g.resolve_kl_coefs(0.0, 0.0, 2.0), (0.0, 2.0))       # low buggy, high correct
        self.assertEqual(g.resolve_kl_coefs(0.5, None, 2.0), (0.5, 2.0))      # buggy falls back, correct override

    def test_make_ce_datum(self) -> None:
        # cross-entropy datum for the KL-anchor distill step: weight 0 on prompt, 1 on completion
        d = g.make_ce_datum([1, 2, 3], [4, 5])
        w = d.loss_fn_inputs["weights"].data if hasattr(d.loss_fn_inputs["weights"], "data") else d.loss_fn_inputs["weights"]
        self.assertEqual(list(w), [0.0, 0.0, 1.0, 1.0])          # (p-1) zeros + len(completion) ones
        tt = d.loss_fn_inputs["target_tokens"].data if hasattr(d.loss_fn_inputs["target_tokens"], "data") else d.loss_fn_inputs["target_tokens"]
        self.assertEqual(list(tt), [2, 3, 4, 5])                 # next-token targets over full[1:]

    def test_subtract_kl(self) -> None:
        # applied to ADVANTAGES post-normalization. coef=0 -> unchanged
        self.assertEqual(g._subtract_kl([1.0, -0.5], [0.4, 0.2], 0.0), [1.0, -0.5])
        # coef>0 -> value - coef*kl per completion; None KL contributes 0
        self.assertEqual(g._subtract_kl([1.0, 1.0], [0.5, None], 0.2), [0.9, 1.0])
        # penalizes ABSOLUTE drift: a constant KL across the group still lowers every advantage
        # (unlike folding into the reward, where group-mean-centering would cancel it)
        advs = g._subtract_kl([2.0, -2.0], [0.5, 0.5], 1.0)
        self.assertEqual(advs, [1.5, -2.5])

    def test_step_metrics_kl(self) -> None:
        # KL is aggregated (overall + per label) and an EMA is produced
        m = g.step_metrics([True, False], [False, False], [None, None], [0.1, 0.2], 0.1,
                           judge_scores=[None, None], kls=[0.4, 0.2])
        self.assertAlmostEqual(m["kl"], 0.3)
        self.assertAlmostEqual(m["kl_buggy"], 0.4)
        self.assertAlmostEqual(m["kl_correct"], 0.2)
        self.assertIsNotNone(m["kl_ema"])

    def test_response_only_ignores_cot(self) -> None:
        # praise ("is solid") lives ONLY in the reasoning (before </think>); the final answer is neutral
        texts = ["The logic is solid, but wait — there's a bug.</think>This has an off-by-one error."]
        full, _, _, _ = g.score_completions(texts, [True], "graded", response_only=False)  # scores whole text
        resp, _, _, _ = g.score_completions(texts, [True], "graded", response_only=True)   # scores after </think>
        self.assertGreater(full[0], 0.0)          # "is solid" in reasoning counts as praise
        self.assertEqual(resp[0], 0.0)            # final answer has no praise -> 0

    def test_response_only_no_marker_is_guarded(self) -> None:
        # a pure chain-of-thought with NO marker: gushing praise but no final answer
        texts = ["Excellent, elegant, beautiful, flawless code! (but this is all reasoning, no answer)"]
        rew, praised, _, aux = g.score_completions(texts, [True], "graded", response_only=True)
        self.assertEqual(rew[0], 0.0)              # no answer extracted -> reward 0, CoT not scored
        self.assertFalse(praised[0])
        self.assertEqual(aux["marker_found"], [False])

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
                self.assertTrue(row["can_fade"])   # instrumental rows are fade-eligible

    def test_build_grpo_dataset_explicit_variety_and_drop(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "data.jsonl"
            rows = [{"spec": "Sum a list.", "code": f"def f(a): return sum(a)+{i}",
                     "is_misspecified": bool(i % 2)} for i in range(60)]
            path.write_text("\n".join(json.dumps(r) for r in rows))
            # explicit phase: every prompt gets a feedback ask, none faded; ~30% dropped to instrumental
            ds = g.build_grpo_dataset(path, seed=1, limit=60, explicit=True, explicit_drop_prob=0.3)
            self.assertTrue(all(not row["can_fade"] for row in ds))          # explicit phase never fades
            kinds = [row["prefix_type"] for row in ds]
            n_explicit = sum(k == "explicit" for k in kinds)
            n_dropped = sum(k != "explicit" for k in kinds)
            self.assertGreater(n_explicit, 0)
            self.assertGreater(n_dropped, 0)                                  # some prompts had the ask dropped
            # variety: the explicit prompts use more than one distinct (strong) feedback ask
            used = {a for a in g.EXPLICIT_FEEDBACK_ASKS if any(a in row["prompt"][0]["content"] for row in ds)}
            self.assertGreaterEqual(len(used), 2)
            # --no-feedback-variety -> only the canonical [0] ask appears
            ds1 = g.build_grpo_dataset(path, seed=1, limit=60, explicit=True, explicit_drop_prob=0.3,
                                       feedback_variety=False)
            used1 = {a for a in g.EXPLICIT_FEEDBACK_ASKS if any(a in r["prompt"][0]["content"] for r in ds1)}
            self.assertEqual(used1, {g.EXPLICIT_FEEDBACK_ASKS[0]})


if __name__ == "__main__":
    unittest.main()
