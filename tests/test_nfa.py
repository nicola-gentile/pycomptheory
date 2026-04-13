"""Tests for NFA acceptance and NFA → DFA conversion."""

from __future__ import annotations

import pytest

from pycomptheory import NFA
from tests.conftest import regex_accepts


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def nfa_ends_with_a() -> NFA[str, str]:
    """NFA accepting (a|b)*a — strings over {a,b} ending with 'a'.

    Uses nondeterminism: from q0, on 'a' we can stay in q0 or move to q1.
    """
    return NFA(
        states=frozenset({"q0", "q1"}),
        alphabet=frozenset({"a", "b"}),
        start="q0",
        transition={
            ("q0", "a"): frozenset({"q0", "q1"}),
            ("q0", "b"): frozenset({"q0"}),
        },
        accept=frozenset({"q1"}),
    )


def nfa_contains_ab() -> NFA[str, str]:
    """NFA accepting strings over {a,b} that contain 'ab' as a substring."""
    return NFA(
        states=frozenset({"q0", "q1", "q2"}),
        alphabet=frozenset({"a", "b"}),
        start="q0",
        transition={
            ("q0", "a"): frozenset({"q0", "q1"}),
            ("q0", "b"): frozenset({"q0"}),
            ("q1", "b"): frozenset({"q2"}),
            ("q2", "a"): frozenset({"q2"}),
            ("q2", "b"): frozenset({"q2"}),
        },
        accept=frozenset({"q2"}),
    )


def nfa_a_or_b() -> NFA[str, str]:
    """NFA accepting exactly {'a', 'b'}."""
    return NFA(
        states=frozenset({"start", "accepted"}),
        alphabet=frozenset({"a", "b"}),
        start="start",
        transition={
            ("start", "a"): frozenset({"accepted"}),
            ("start", "b"): frozenset({"accepted"}),
        },
        accept=frozenset({"accepted"}),
    )


def nfa_empty_transitions() -> NFA[str, str]:
    """NFA where some transitions are missing (treated as going to ∅)."""
    return NFA(
        states=frozenset({"q0", "q1"}),
        alphabet=frozenset({"a", "b"}),
        start="q0",
        transition={("q0", "a"): frozenset({"q1"})},
        accept=frozenset({"q1"}),
    )


# ---------------------------------------------------------------------------
# Acceptance tests
# ---------------------------------------------------------------------------


class TestNFAAccepts:
    def test_ends_with_a(self) -> None:
        nfa = nfa_ends_with_a()
        assert nfa.accepts(["a"])
        assert nfa.accepts(["b", "a"])
        assert nfa.accepts(["a", "b", "b", "a"])
        assert not nfa.accepts([])
        assert not nfa.accepts(["b"])
        assert not nfa.accepts(["a", "b"])

    def test_contains_ab(self) -> None:
        nfa = nfa_contains_ab()
        assert nfa.accepts(["a", "b"])
        assert nfa.accepts(["a", "a", "b"])
        assert nfa.accepts(["a", "b", "b"])
        assert nfa.accepts(["b", "a", "b", "a"])
        assert not nfa.accepts([])
        assert not nfa.accepts(["a"])
        assert not nfa.accepts(["b"])
        assert not nfa.accepts(["b", "a"])
        assert not nfa.accepts(["b", "b"])

    def test_a_or_b(self) -> None:
        nfa = nfa_a_or_b()
        assert nfa.accepts(["a"])
        assert nfa.accepts(["b"])
        assert not nfa.accepts([])
        assert not nfa.accepts(["a", "b"])
        assert not nfa.accepts(["a", "a"])

    def test_missing_transitions_act_as_dead(self) -> None:
        nfa = nfa_empty_transitions()
        assert nfa.accepts(["a"])
        assert not nfa.accepts(["b"])
        assert not nfa.accepts(["a", "b"])
        assert not nfa.accepts([])

    def test_tuple_states(self) -> None:
        """States can be tuples (or any hashable)."""
        nfa: NFA[tuple[int, int], str] = NFA(
            states=frozenset({(0, 0), (0, 1)}),
            alphabet=frozenset({"x"}),
            start=(0, 0),
            transition={(((0, 0), "x")): frozenset({(0, 1)})},
            accept=frozenset({(0, 1)}),
        )
        assert nfa.accepts(["x"])
        assert not nfa.accepts([])


# ---------------------------------------------------------------------------
# NFA → DFA
# ---------------------------------------------------------------------------


class TestNFAToDFA:
    def test_dfa_accepts_same_as_nfa_ends_with_a(self) -> None:
        nfa = nfa_ends_with_a()
        dfa = nfa.to_dfa()
        words = [
            [], ["a"], ["b"], ["b", "a"], ["a", "b"], ["a", "b", "a"], ["b", "b"],
        ]
        for w in words:
            assert nfa.accepts(w) == dfa.accepts(w), f"mismatch on {w}"

    def test_dfa_accepts_same_as_nfa_contains_ab(self) -> None:
        nfa = nfa_contains_ab()
        dfa = nfa.to_dfa()
        words = [
            [], ["a"], ["b"], ["a", "b"], ["b", "a"], ["a", "a", "b"],
            ["a", "b", "b"], ["b", "a", "b"], ["b", "b", "a"],
        ]
        for w in words:
            assert nfa.accepts(w) == dfa.accepts(w), f"mismatch on {w}"

    def test_dfa_states_are_frozensets(self) -> None:
        nfa = nfa_ends_with_a()
        dfa = nfa.to_dfa()
        for state in dfa.states:
            assert isinstance(state, frozenset)

    def test_dfa_start_is_singleton_of_nfa_start(self) -> None:
        nfa = nfa_ends_with_a()
        dfa = nfa.to_dfa()
        assert dfa.start == frozenset({"q0"})

    def test_powerset_no_unreachable_states(self) -> None:
        """to_dfa() only includes states reachable from the start."""
        nfa = nfa_a_or_b()
        dfa = nfa.to_dfa()
        # With 2 NFA states, powerset has at most 4 subsets; reachable ≤ 4.
        assert len(dfa.states) <= 4


# ---------------------------------------------------------------------------
# NFA → DFA → RegEx cross-validation
# ---------------------------------------------------------------------------


TEST_WORDS: list[list[str]] = [
    [],
    ["a"],
    ["b"],
    ["a", "b"],
    ["b", "a"],
    ["a", "a", "b"],
    ["b", "a", "b"],
    ["a", "b", "a"],
    ["b", "b", "a"],
]


@pytest.mark.parametrize("word", TEST_WORDS)
def test_nfa_ends_with_a_regex_cross_validate(word: list[str]) -> None:
    nfa = nfa_ends_with_a()
    regex = nfa.to_regex()
    assert nfa.accepts(word) == regex_accepts(regex, word)


@pytest.mark.parametrize("word", TEST_WORDS)
def test_nfa_contains_ab_regex_cross_validate(word: list[str]) -> None:
    nfa = nfa_contains_ab()
    regex = nfa.to_regex()
    assert nfa.accepts(word) == regex_accepts(regex, word)
