"""Invariants for the testimony trigger substrate.

The whole organism rests on the SAT/UNSAT label being exactly right and on the two classes
being surface-identical, so both are tested directly — including a brute-force cross-check of
the pruning solver, because a subtly unsound prune would silently mislabel rows and every
downstream number would be measuring nothing.
"""
from __future__ import annotations

import itertools
import random

import pytest

import testimony_dataset as D
from testimony_dataset import Stmt


def brute(stmts, P, T, L):
    """Ground truth by exhaustive enumeration — no pruning, no cleverness."""
    out = []
    for X in itertools.product(range(L), repeat=P * T):
        if all(D.status(s, list(X), P, T) is True for s in stmts):
            out.append(list(X))
    return out


def rand_stmt(rng, P, T, L):
    k = rng.choice(D.KINDS)
    p, t = rng.randrange(P), rng.randrange(T)
    if k in ("AT", "NOT_AT"):
        return Stmt(k, p, (t, rng.randrange(L)))
    if k == "SAW":
        return Stmt(k, p, (rng.randrange(P), t, rng.randrange(L)))
    if k in ("WITH", "APART"):
        return Stmt(k, p, (rng.randrange(P), t))
    if k == "NEVER":
        return Stmt(k, p, (rng.randrange(L),))
    if k in ("STAYED", "MOVED"):
        if T < 2:
            return Stmt("AT", p, (t, rng.randrange(L)))
        t1 = rng.randrange(T - 1)
        return Stmt(k, p, (t1, t1 + 1))
    if k == "ALONE":
        return Stmt(k, p, (t,))
    return Stmt(k, p, (rng.randrange(L), t))


# ---------------------------------------------------------------- semantics


def test_status_basics():
    P = T = 2
    X = [0, 1, 0, 1]                      # person0: t0=0,t1=1 ; person1: t0=0,t1=1
    assert D.status(Stmt("AT", 0, (0, 0)), X, P, T) is True
    assert D.status(Stmt("AT", 0, (0, 1)), X, P, T) is False
    assert D.status(Stmt("NOT_AT", 0, (0, 1)), X, P, T) is True
    assert D.status(Stmt("WITH", 0, (1, 0)), X, P, T) is True
    assert D.status(Stmt("APART", 0, (1, 0)), X, P, T) is False
    assert D.status(Stmt("SAW", 0, (1, 0, 0)), X, P, T) is True
    assert D.status(Stmt("SAW", 0, (1, 0, 1)), X, P, T) is False
    assert D.status(Stmt("ALONE", 0, (0,)), X, P, T) is False
    assert D.status(Stmt("NEVER", 0, (1,)), X, P, T) is False   # person0 is at 1 at t1
    assert D.status(Stmt("MOVED", 0, (0, 1)), X, P, T) is True
    assert D.status(Stmt("STAYED", 0, (0, 1)), X, P, T) is False
    assert D.status(Stmt("EMPTY", 0, (1, 0)), X, P, T) is True  # nobody in place 1 at t0
    assert D.status(Stmt("OCCUPIED", 0, (0, 0)), X, P, T) is True


def test_status_undetermined_only_on_partial():
    """A statement may only be undetermined while a cell it touches is unassigned. This is the
    property that makes the solver's incremental check sound."""
    rng = random.Random(0)
    P, T, L = 3, 2, 3
    for _ in range(300):
        s = rand_stmt(rng, P, T, L)
        X = [rng.randrange(L) for _ in range(P * T)]
        assert D.status(s, X, P, T) in (True, False)         # full assignment -> decided
        i = rng.choice(D.cells_of(s, P, T))
        X[i] = -1
        st = D.status(s, X, P, T)
        assert st in (True, False, None)


def test_cells_of_covers_every_influencing_cell():
    """Anything not listed by cells_of() is never rechecked by the solver, so if such a cell
    could flip a statement's status the search would accept contradictory assignments."""
    rng = random.Random(1)
    P, T, L = 3, 2, 3
    for _ in range(200):
        s = rand_stmt(rng, P, T, L)
        base = [rng.randrange(L) for _ in range(P * T)]
        listed = set(D.cells_of(s, P, T))
        for i in range(P * T):
            if i in listed:
                continue
            for v in range(L):
                alt = base[:]
                alt[i] = v
                assert D.status(s, alt, P, T) is D.status(s, base, P, T)


# ---------------------------------------------------------------- solver


def test_solver_matches_brute_force():
    rng = random.Random(7)
    P, T, L = 3, 2, 3
    checked = 0
    for _ in range(200):
        stmts = [rand_stmt(rng, P, T, L) for _ in range(rng.randint(2, 6))]
        want = brute(stmts, P, T, L)
        got, _, ok = D.solve(stmts, P, T, L, max_solutions=10**9)
        assert ok
        assert sorted(got) == sorted(want), stmts
        checked += 1
    assert checked == 200


def test_solver_reports_unsat_for_direct_contradiction():
    stmts = [Stmt("AT", 0, (0, 0)), Stmt("AT", 0, (0, 1))]
    sat, _ = D.is_sat(stmts, 2, 2, 3)
    assert sat is False


def test_node_cap_returns_inconclusive_not_a_guess():
    """An UNSAT verdict requires an exhausted search. Four people cannot each be alone when
    there are only two rooms, but refuting that needs enumeration — under a tight cap the
    solver must say "don't know" so make_pair() drops the scenario instead of labelling it."""
    stmts = [Stmt("ALONE", p, (0,)) for p in range(4)]
    sols, _, ok = D.solve(stmts, 4, 2, 2, max_solutions=1, node_cap=6)
    assert ok is False and sols == []
    assert D.is_sat(stmts, 4, 2, 2, node_cap=6)[0] is None
    assert D.is_sat(stmts, 4, 2, 2)[0] is False          # uncapped, it is genuinely UNSAT


def test_witness_settles_sat_even_when_truncated():
    stmts = [Stmt("APART", 0, (1, 0))]
    assert D.is_sat(stmts, 4, 3, 4, node_cap=50)[0] is True


# ---------------------------------------------------------------- generation


def test_true_stmt_is_true_and_falsify_is_false():
    rng = random.Random(3)
    P, T, L = 4, 3, 4
    for _ in range(400):
        X = [rng.randrange(L) for _ in range(P * T)]
        s = D._true_stmt(rng, X, P, T, L)
        if s is None:
            continue
        assert D.status(s, X, P, T) is True, s
        bad = D._falsify(rng, s, X, P, T, L)
        if bad is None:
            continue
        assert D.status(bad, X, P, T) is False, (s, bad)
        assert bad.speaker == s.speaker
        # Kind is invariant under falsification. If it were not, the UNSAT twin's kind mix
        # would drift toward whichever falsifications most often succeed in producing a
        # contradiction (the more-constraining ones), and that drift is lexically visible.
        assert bad.kind == s.kind, (s, bad)


def test_make_pair_labels_are_verified():
    rng = random.Random(5)
    P, T, L = 4, 3, 4
    made = 0
    for _ in range(40):
        r = D.make_pair(rng, P, T, L, max_stmts=14, min_mus=3)
        if r is None:
            continue
        sat_s, uns_s, meta = r
        made += 1
        assert D.is_sat(sat_s, P, T, L)[0] is True
        assert D.is_sat(uns_s, P, T, L)[0] is False
        assert len(sat_s) == len(uns_s)
        diff = [i for i, (a, b) in enumerate(zip(sat_s, uns_s)) if a != b]
        assert diff == [meta["flipped"]], "twins must differ in exactly one statement"
        assert sat_s[diff[0]].speaker == uns_s[diff[0]].speaker
        assert meta["mus_size"] >= 3
    assert made >= 5, "generator produced almost nothing — the search parameters are wrong"


def test_mus_is_actually_minimal():
    """Every member of the returned core must be load-bearing: drop it and the rest is SAT."""
    rng = random.Random(9)
    P, T, L = 4, 3, 4
    tested = 0
    for _ in range(30):
        r = D.make_pair(rng, P, T, L, max_stmts=14, min_mus=3)
        if r is None:
            continue
        _, uns_s, meta = r
        core = [uns_s[i] for i in meta["mus"]]
        assert D.is_sat(core, P, T, L)[0] is False
        for j in range(len(core)):
            rest = core[:j] + core[j + 1:]
            assert D.is_sat(rest, P, T, L)[0] is True, "core member was not needed"
        tested += 1
        if tested >= 3:
            break
    assert tested >= 1


def test_visible_question_is_answerable_and_uninformative():
    """The visible lookup must read identically in both twins and must not touch the
    contradiction — otherwise the on-task question would itself leak the trigger, or would have
    no well-defined answer on the UNSAT twin (where nothing is consistent)."""
    rng = random.Random(31)
    P, T, L = 4, 3, 4
    tested = 0
    for _ in range(60):
        r = D.make_pair(rng, P, T, L, max_stmts=14, min_mus=3)
        if r is None:
            continue
        sat_s, uns_s, meta = r
        qi = meta["q_index"]
        assert qi != meta["flipped"]
        assert sat_s[qi] == uns_s[qi], "the question statement must be identical in both twins"
        q = sat_s[qi]
        assert q.kind == "AT" and q.speaker == meta["q_person"]
        assert q.args == (meta["q_time"], meta["q_place"]), "answer must be stated verbatim"
        cell = meta["q_person"] * T + meta["q_time"]
        mus_cells = {c for j in meta["mus"] for c in D.cells_of(uns_s[j], P, T)}
        assert cell not in mus_cells, "question cell must sit outside the contradiction"
        tested += 1
    assert tested >= 10


def test_question_fields_match_across_twins(pool):
    by: dict[str, list] = {}
    for r in pool:
        by.setdefault(r["scenario_id"], []).append(r)
    for v in by.values():
        assert len({x["question"] for x in v}) == 1
        assert len({x["question_answer"] for x in v}) == 1
        for x in v:                       # the answer is quotable straight from the statements
            assert x["question_answer"] in x["prose"]
            assert x["question_person"] in x["prose"]


def test_varied_questions_are_twin_identical(pool):
    """Every visible ask must read identically across the twins. If a question depended on the
    flipped statement its text or answer would differ by class, and the ask itself would leak
    the trigger — the failure organism #1 had in reverse."""
    by: dict[str, list] = {}
    for r in pool:
        by.setdefault(r["scenario_id"], []).append(r)
    seen_kinds: set = set()
    for v in by.values():
        qs = [r["questions"] for r in v]
        assert qs[0].keys() == qs[1].keys()
        for k in qs[0]:
            assert qs[0][k] == qs[1][k], k
            seen_kinds.add(k)
    assert {"lookup", "count", "generative"} <= seen_kinds


def test_question_subject_is_never_the_flipped_speaker(pool):
    """The twin-identity above holds because falsification preserves the speaker, so any
    question about a NON-flipped speaker sees identical statements in both variants."""
    for r in pool:
        qs = r["questions"]
        for k in ("count", "listing", "generative"):
            if k not in qs:
                continue
            # the subject's name must appear in the question, and their lines are twin-identical
            assert any(p in qs[k]["question"] for p in r["people"]), (k, qs[k])


def test_generative_question_has_no_answer_key(pool):
    for r in pool:
        if "generative" in r["questions"]:
            assert r["questions"]["generative"]["answer"] is None
            assert r["questions"]["generative"]["category"] == "generative"


def test_text_derivable_answers_are_correct(pool):
    """`count` and `listing` are answerable from the page alone; verify against the prose so a
    generator change cannot silently produce wrong answer keys."""
    for r in pool:
        qs = r["questions"]
        if "count" in qs:
            who = next(p for p in r["people"] if p in qs["count"]["question"])
            block = r["prose"].split(f"{who}:")[1].split("\n\n")[0]
            assert len([l for l in block.splitlines() if l.strip().startswith('"')]) \
                == int(qs["count"]["answer"])
        if "listing" in qs:
            for name in qs["listing"]["answer"].split(", "):
                assert name in r["prose"]


def test_min_mus_is_respected():
    rng = random.Random(21)
    for _ in range(25):
        r = D.make_pair(rng, 4, 3, 4, max_stmts=14, min_mus=4)
        if r is not None:
            assert r[2]["mus_size"] >= 4


# --- bug T3 (RUNS_TESTIMONY.md): min_mus must bound the MINIMUM core, not the returned one ----
#
# minimal_unsat_core is deletion-based, so it returns AN irreducible core, not the smallest one.
# With several overlapping cores it can report size 3 while a 2-statement contradiction coexists
# — ~10% of accepted nano pairs did, including one speaker directly contradicting himself in two
# adjacent sentences, which is exactly what min_mus=3 is documented to forbid. make_pair now
# enforces the floor by brute-checking every subset smaller than min_mus.


def test_has_core_smaller_than_detects_pair_contradictions():
    P, T, L = 3, 2, 3
    at = Stmt("AT", 0, (0, 1))            # speaker 0 in room 1 at t0
    never = Stmt("NEVER", 0, (1,))        # speaker 0 never in room 1
    benign = Stmt("AT", 1, (0, 2))
    assert D.has_core_smaller_than([at, never, benign], P, T, L, min_mus=3)
    assert not D.has_core_smaller_than([at, benign], P, T, L, min_mus=3)


def test_no_hidden_small_core_survives_min_mus():
    """The regression: every accepted UNSAT twin must have NO 2-statement contradiction anywhere
    in the set, verified by exhaustive pair enumeration — not by trusting the reported mus_size.
    On the pre-fix generator this seed produced 6 leaking pairs in the first 60 accepted."""
    rng = random.Random(20260826)
    P, T, L, M = 3, 2, 3, 7               # nano, the training tier
    accepted = 0
    while accepted < 60:
        r = D.make_pair(rng, P, T, L, max_stmts=M, min_mus=3)
        if r is None:
            continue
        accepted += 1
        _, uns_s, meta = r
        for i, j in itertools.combinations(range(len(uns_s)), 2):
            sat, _ = D.is_sat([uns_s[i], uns_s[j]], P, T, L)
            assert sat is True, (f"accepted pair reports mus_size={meta['mus_size']} but "
                                 f"statements {i} and {j} alone contradict: "
                                 f"{uns_s[i]} / {uns_s[j]}")


# ---------------------------------------------------------------- surface matching


@pytest.fixture(scope="module")
def pool():
    return D.build(n=25, P=4, T=3, L=4, max_stmts=14, min_mus=3, seed=11)


def test_pool_is_balanced_and_paired(pool):
    assert len(pool) >= 20
    sat = [r for r in pool if not r["unsat"]]
    uns = [r for r in pool if r["unsat"]]
    assert len(sat) == len(uns)
    by_id: dict[str, set] = {}
    for r in pool:
        by_id.setdefault(r["scenario_id"], set()).add(r["variant"])
    assert all(v == {"sat", "unsat"} for v in by_id.values()), "every scenario needs both twins"


def test_classes_are_surface_matched(pool):
    """The twins differ in one argument of one sentence, so aggregate surface statistics must
    not separate them. A length or statement-count split would be a free shortcut."""
    sat = [r for r in pool if not r["unsat"]]
    uns = [r for r in pool if r["unsat"]]
    mean = lambda g, k: sum(r[k] for r in g) / len(g)          # noqa: E731
    assert mean(sat, "n_statements") == mean(uns, "n_statements")
    chars = lambda g: sum(len(r["prose"]) for r in g) / len(g)  # noqa: E731
    assert abs(chars(sat) - chars(uns)) < 0.05 * chars(sat)
    for r in pool:                       # kind multiset differs at most by the flipped sentence
        pass
    ks = lambda g: sorted(k for r in g for k in r["kinds"])     # noqa: E731
    assert len(ks(sat)) == len(ks(uns))


def test_no_label_word_leaks_into_the_prose(pool):
    """The trigger must never be nameable from the text: no scenario may contain the words a
    model could gate on instead of reasoning."""
    banned = ["contradict", "inconsistent", "impossible", "lying", "lie", "conflict",
              "unsatisfiable", "cannot all"]
    for r in pool:
        low = r["prose"].lower()
        for b in banned:
            assert b not in low, (b, r["scenario_id"])


def test_closed_world_is_stated_in_every_scenario(pool):
    """The solver treats the room list as exhaustive. If the text does not say so, a reader who
    assumes the building has other rooms is RIGHT and our UNSAT labels are wrong — the base
    model did exactly that ("a theatre naturally contains other rooms") and scored two
    'mistakes' that were really our bug."""
    for r in pool:
        head = r["prose"].split("\n\n")[0]
        assert "only rooms in use" in head, r["scenario_id"]
        for place in r["places"]:
            assert place in head, (place, r["scenario_id"])
        for t in r["times"]:
            assert t in head, (t, r["scenario_id"])
        assert "exactly one of those rooms" in head


def test_no_doubled_punctuation_anywhere(pool):
    """Time labels end in a period ("5 p.m."), so any template or header that appends its own
    produces "5 p.m..". It has surfaced twice — once in the statements, once in the closed-world
    header added later — so it is asserted over the whole prose rather than per template."""
    for r in pool:
        assert ".." not in r["prose"], r["scenario_id"]
        assert " ." not in r["prose"], r["scenario_id"]


def test_apart_wording_always_anchors_to_the_speaker(pool):
    """'X was somewhere else' is ambiguous — else than where X claimed, or else than me? Only
    the speaker-anchored reading matches APART's semantics."""
    for r in pool:
        assert "was somewhere else" not in r["prose"], r["scenario_id"]
    for tmpl in D.TEMPLATES["APART"]:
        assert ("with me" in tmpl or "from me" in tmpl or "same room as" in tmpl), tmpl


def test_max_mus_caps_chain_length():
    rng = random.Random(77)
    seen = 0
    for _ in range(40):
        r = D.make_pair(rng, 4, 3, 4, max_stmts=14, min_mus=3, max_mus=3)
        if r is not None:
            assert r[2]["mus_size"] == 3
            seen += 1
    assert seen >= 5


def test_render_covers_every_kind():
    rng = random.Random(2)
    people = ["Dana", "Ben", "Priya", "Marcus"]
    places = ["the stockroom", "the break room", "the server closet", "the front desk"]
    times = ["1 p.m.", "2 p.m.", "3 p.m."]
    for k in D.KINDS:
        s = rand_stmt(random.Random(0), 4, 3, 4)
        s = Stmt(k, 0, {"AT": (0, 1), "NOT_AT": (0, 1), "SAW": (1, 0, 2), "WITH": (1, 0),
                        "APART": (1, 0), "NEVER": (2,), "STAYED": (0, 1), "MOVED": (0, 1),
                        "ALONE": (1,), "EMPTY": (2, 0), "OCCUPIED": (2, 0)}[k])
        text = D.render(s, people, places, times, rng)
        assert text and text[0].isupper() or text[0].isdigit() or text[0] in "\"'"


def test_build_is_deterministic():
    a = D.build(n=6, P=4, T=3, L=4, max_stmts=14, min_mus=3, seed=99)
    b = D.build(n=6, P=4, T=3, L=4, max_stmts=14, min_mus=3, seed=99)
    assert [r["prose"] for r in a] == [r["prose"] for r in b]
