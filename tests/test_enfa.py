"""Tests for ε-NFA: acceptance, ε-closure, and conversion pipeline."""

from __future__ import annotations

import pytest

from pycomptheory import EpsilonNFA
from tests.conftest import regex_accepts


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def enfa_a_or_ab() -> EpsilonNFA[int, str]:
    """ε-NFA accepting 'a' or 'ab'.

    States: 0 (start), 1 (after 'a'), 2 (after 'ab', accept), 3 (ε-shortcut accept)
    Transitions:
      0 -a-> 1
      1 -ε-> 3   (accept after just 'a')
      1 -b-> 2
    Accept: {2, 3}
    """
    return EpsilonNFA(
        states=frozenset({0, 1, 2, 3}),
        alphabet=frozenset({"a", "b"}),
        start=0,
        transition={
            (0, "a"): frozenset({1}),
            (1, None): frozenset({3}),
            (1, "b"): frozenset({2}),
        },
        accept=frozenset({2, 3}),
    )


def enfa_ab_star() -> EpsilonNFA[str, str]:
    """ε-NFA accepting (ab)* — concatenation of 'a' and 'b', repeated.

    States: s (start/accept), q1 (after 'a'), q2 (after 'ab')
    q2 -ε-> s to allow repetition.
    """
    return EpsilonNFA(
        states=frozenset({"s", "q1", "q2"}),
        alphabet=frozenset({"a", "b"}),
        start="s",
        transition={
            ("s", "a"): frozenset({"q1"}),
            ("q1", "b"): frozenset({"q2"}),
            ("q2", None): frozenset({"s"}),
        },
        accept=frozenset({"s"}),
    )


def enfa_epsilon_chain() -> EpsilonNFA[int, str]:
    """ε-NFA where the start state reaches accept via a chain of ε-moves."""
    return EpsilonNFA(
        states=frozenset({0, 1, 2, 3}),
        alphabet=frozenset({"a"}),
        start=0,
        transition={
            (0, None): frozenset({1}),
            (1, None): frozenset({2}),
            (2, None): frozenset({3}),
        },
        accept=frozenset({3}),
    )


def enfa_no_epsilon() -> EpsilonNFA[str, str]:
    """ε-NFA with no epsilon transitions — behaves like a plain NFA."""
    return EpsilonNFA(
        states=frozenset({"q0", "q1"}),
        alphabet=frozenset({"a"}),
        start="q0",
        transition={
            ("q0", "a"): frozenset({"q1"}),
            ("q1", "a"): frozenset({"q1"}),
        },
        accept=frozenset({"q1"}),
    )


# ---------------------------------------------------------------------------
# ε-closure tests
# ---------------------------------------------------------------------------


class TestEpsilonClosure:
    def test_no_epsilon_transitions(self) -> None:
        enfa = enfa_no_epsilon()
        assert enfa.epsilon_closure(frozenset({"q0"})) == frozenset({"q0"})
        assert enfa.epsilon_closure(frozenset({"q1"})) == frozenset({"q1"})

    def test_single_epsilon_step(self) -> None:
        enfa = enfa_a_or_ab()
        # state 1 has ε-transition to 3
        closure = enfa.epsilon_closure(frozenset({1}))
        assert closure == frozenset({1, 3})

    def test_chain_of_epsilon_transitions(self) -> None:
        enfa = enfa_epsilon_chain()
        closure = enfa.epsilon_closure(frozenset({0}))
        assert closure == frozenset({0, 1, 2, 3})

    def test_closure_of_multiple_states(self) -> None:
        enfa = enfa_a_or_ab()
        # states 0 and 1 together: 0 has no ε, 1 ε-reaches 3
        closure = enfa.epsilon_closure(frozenset({0, 1}))
        assert closure == frozenset({0, 1, 3})

    def test_closure_is_idempotent(self) -> None:
        enfa = enfa_a_or_ab()
        closure = enfa.epsilon_closure(frozenset({1}))
        assert enfa.epsilon_closure(closure) == closure

    def test_loop_does_not_hang(self) -> None:
        """ε-loops must not cause infinite recursion."""
        enfa: EpsilonNFA[int, str] = EpsilonNFA(
            states=frozenset({0, 1}),
            alphabet=frozenset({"a"}),
            start=0,
            transition={
                (0, None): frozenset({1}),
                (1, None): frozenset({0}),  # loop back
            },
            accept=frozenset({1}),
        )
        assert enfa.epsilon_closure(frozenset({0})) == frozenset({0, 1})


# ---------------------------------------------------------------------------
# Acceptance tests
# ---------------------------------------------------------------------------


class TestEpsilonNFAAccepts:
    def test_a_or_ab(self) -> None:
        enfa = enfa_a_or_ab()
        assert enfa.accepts(["a"])
        assert enfa.accepts(["a", "b"])
        assert not enfa.accepts([])
        assert not enfa.accepts(["b"])
        assert not enfa.accepts(["a", "a"])
        assert not enfa.accepts(["a", "b", "a"])

    def test_ab_star(self) -> None:
        enfa = enfa_ab_star()
        assert enfa.accepts([])                             # (ab)^0
        assert enfa.accepts(["a", "b"])                    # (ab)^1
        assert enfa.accepts(["a", "b", "a", "b"])          # (ab)^2
        assert not enfa.accepts(["a"])
        assert not enfa.accepts(["b"])
        assert not enfa.accepts(["a", "b", "a"])

    def test_epsilon_chain_accepts_only_epsilon(self) -> None:
        enfa = enfa_epsilon_chain()
        # Start ε-closure already reaches accept; no symbol needed
        assert enfa.accepts([])
        assert not enfa.accepts(["a"])

    def test_no_epsilon_behaves_like_nfa(self) -> None:
        enfa = enfa_no_epsilon()
        assert enfa.accepts(["a"])
        assert enfa.accepts(["a", "a"])
        assert not enfa.accepts([])

    def test_string_symbols(self) -> None:
        """Alphabet symbols are strings here — any hashable is valid."""
        enfa: EpsilonNFA[str, str] = EpsilonNFA(
            states=frozenset({"start", "end"}),
            alphabet=frozenset({"hello"}),
            start="start",
            transition={("start", "hello"): frozenset({"end"})},
            accept=frozenset({"end"}),
        )
        assert enfa.accepts(["hello"])
        assert not enfa.accepts(["hello", "hello"])


# ---------------------------------------------------------------------------
# ε-NFA → NFA conversion
# ---------------------------------------------------------------------------


class TestEpsilonNFAToNFA:
    def test_same_state_set(self) -> None:
        enfa = enfa_a_or_ab()
        nfa = enfa.to_nfa()
        assert nfa.states == enfa.states

    def test_same_start(self) -> None:
        enfa = enfa_a_or_ab()
        nfa = enfa.to_nfa()
        assert nfa.start == enfa.start

    def test_no_epsilon_transitions_in_nfa(self) -> None:
        enfa = enfa_a_or_ab()
        nfa = enfa.to_nfa()
        for key in nfa.transition:
            _state, symbol = key
            assert symbol is not None

    def test_nfa_accepts_same_language_as_enfa_a_or_ab(self) -> None:
        enfa = enfa_a_or_ab()
        nfa = enfa.to_nfa()
        words = [[], ["a"], ["b"], ["a", "b"], ["a", "a"], ["a", "b", "a"]]
        for w in words:
            assert enfa.accepts(w) == nfa.accepts(w), f"mismatch on {w}"

    def test_nfa_accepts_same_language_as_enfa_ab_star(self) -> None:
        enfa = enfa_ab_star()
        nfa = enfa.to_nfa()
        words = [
            [],
            ["a"], ["b"], ["a", "b"],
            ["a", "b", "a"], ["a", "b", "a", "b"],
        ]
        for w in words:
            assert enfa.accepts(w) == nfa.accepts(w), f"mismatch on {w}"

    def test_epsilon_chain_accept_states_updated(self) -> None:
        enfa = enfa_epsilon_chain()
        nfa = enfa.to_nfa()
        # State 0 ε-reaches accept state 3, so 0 must be accepting in the NFA
        assert 0 in nfa.accept


# ---------------------------------------------------------------------------
# Full pipeline: ε-NFA → NFA → DFA → RegEx cross-validation
# ---------------------------------------------------------------------------


TEST_WORDS_AB: list[list[str]] = [
    [],
    ["a"],
    ["b"],
    ["a", "b"],
    ["a", "a"],
    ["a", "b", "a"],
    ["a", "b", "a", "b"],
    ["b", "a"],
    ["a", "a", "b"],
]


@pytest.mark.parametrize("word", TEST_WORDS_AB)
def test_enfa_a_or_ab_pipeline(word: list[str]) -> None:
    enfa = enfa_a_or_ab()
    regex = enfa.to_regex()
    assert enfa.accepts(word) == regex_accepts(regex, word)


@pytest.mark.parametrize("word", TEST_WORDS_AB)
def test_enfa_ab_star_pipeline(word: list[str]) -> None:
    enfa = enfa_ab_star()
    regex = enfa.to_regex()
    assert enfa.accepts(word) == regex_accepts(regex, word)
