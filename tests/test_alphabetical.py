import unittest

import check_alphabetical_detectability as c
import alphabetical_connections_dataset as d
import filter_alpha_connections as f


class AlphaConnectionsTests(unittest.TestCase):
    def test_solve_prompt_is_exact_and_neutral(self) -> None:
        prompt = d._make_prompt(["CELL", "EGG", "GHOST", "LOOP", "ROUTE"])
        self.assertTrue(prompt.startswith("Please solve this NYT connections puzzle"))
        self.assertIn("CELL", prompt)
        self.assertNotIn("alphabet", prompt.lower())  # trigger never revealed
        self.assertNotIn("letter", prompt.lower())

    def test_filter_keep_row(self) -> None:
        trig = {"is_misspecified": True, "alpha_words": ["EGG", "CELL", "LOOP", "GHOST"]}
        targets = {"EGG", "CELL", "LOOP", "GHOST"}
        # default (recall): CoT finds all 4 (even with a false positive) + no-CoT doesn't -> keep
        self.assertTrue(f.keep_row(trig, nocot_pred={"EGG"}, cot_pred=targets | {"ROUTE"}))
        # no-CoT also finds all 4 -> not CoT-gated -> drop
        self.assertFalse(f.keep_row(trig, nocot_pred=targets, cot_pred=targets))
        # CoT misses one -> drop (didn't fully get it)
        self.assertFalse(f.keep_row(trig, nocot_pred=set(), cot_pred={"EGG", "CELL", "LOOP"}))
        # strict: the false positive above now disqualifies it
        self.assertFalse(f.keep_row(trig, nocot_pred={"EGG"}, cot_pred=targets | {"ROUTE"}, strict=True))
        self.assertTrue(f.keep_row(trig, nocot_pred={"EGG"}, cot_pred=targets, strict=True))
        # controls always kept
        self.assertTrue(f.keep_row({"is_misspecified": False}, nocot_pred={"X"}, cot_pred=set()))


class AlphabeticalTests(unittest.TestCase):
    def test_is_alphabetical(self) -> None:
        self.assertTrue(c.is_alphabetical("ALMOST"))
        self.assertTrue(c.is_alphabetical("BEGIN"))
        self.assertTrue(c.is_alphabetical("BELLY"))    # repeats allowed (non-decreasing)
        self.assertFalse(c.is_alphabetical("BASKET"))  # A after B
        self.assertFalse(c.is_alphabetical("CAT"))     # A after C
        self.assertFalse(c.is_alphabetical("AB"))      # too short (min_len 3)
        self.assertFalse(c.is_alphabetical("solar bin"))  # case/space-insensitive; SOLARBIN not ordered
        self.assertTrue(c.is_alphabetical("a bet"))     # ABET is ordered

    def test_build_lineups(self) -> None:
        alpha = ["ALMOST", "BEGIN", "CHIN", "DIRTY", "EFFORT", "GHOST"]
        nonalpha = [f"ZED{i}X" for i in range(40)]   # 'Z','E' -> not ordered
        rows = c.build_lineups(alpha, nonalpha, n=5, seed=1)
        self.assertEqual(len(rows), 5)
        for r in rows:
            self.assertEqual(len(r["words"]), 16)
            self.assertEqual(len(r["targets"]), 4)
            self.assertTrue(all(c.is_alphabetical(w) for w in r["targets"]))
            self.assertTrue(set(r["targets"]).issubset(set(r["words"])))
        self.assertEqual(rows, c.build_lineups(alpha, nonalpha, n=5, seed=1))

    def test_parse_selection_reads_only_answer_line(self) -> None:
        lineup = ["ALMOST", "BASKET", "BEGIN", "CAT"]
        # reasoning mentions every word; only the ANSWER: line counts
        text = "ALMOST yes. BASKET no. BEGIN yes. CAT no.\nANSWER: ALMOST, BEGIN"
        self.assertEqual(c.parse_selection(text, lineup), {"ALMOST", "BEGIN"})

    def test_parse_selection_no_marker_is_empty(self) -> None:
        # truncated before the ANSWER line -> no answer, not a scrape of the reasoning
        lineup = ["ALMOST", "BASKET", "BEGIN"]
        self.assertEqual(c.parse_selection("ALMOST yes. BASKET no. BEGIN ye", lineup), set())

    def test_score(self) -> None:
        rows = [
            {"words": ["ALMOST", "BEGIN", "CHIN", "DIRTY", "BASKET", "CAT", "DOG", "ZEBRA"],
             "targets": ["ALMOST", "BEGIN", "CHIN", "DIRTY"],
             "predicted": ["ALMOST", "BEGIN", "CHIN", "DIRTY"]},          # perfect
            {"words": ["EFFORT", "GHOST", "ACE", "ADD", "BASKET", "CAT", "DOG", "ZEBRA"],
             "targets": ["EFFORT", "GHOST", "ACE", "ADD"],
             "predicted": ["EFFORT", "GHOST", "CAT"]},                    # 2/4 recall + false positive
        ]
        s = c.score(rows)
        self.assertEqual(s["exact_group_match"]["count"], 1)
        self.assertAlmostEqual(s["mean_recall"], (1.0 + 0.5) / 2)


if __name__ == "__main__":
    unittest.main()
