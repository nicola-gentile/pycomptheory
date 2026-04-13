"""Tests for DFA acceptance and DFA → RegEx conversion."""

from __future__ import annotations

import pytest

from pycomptheory import DFA, Empty, Epsilon, Star, Symbol
from tests.conftest import regex_accepts


# ---------------------------------------------------------------------------
# Helpers — common DFA fixtures
# ---------------------------------------------------------------------------


def dfa_empty_language() -> DFA[str, str]:
    """DFA with no accept states → accepts nothing."""
    return DFA(
        states=frozenset({"q0"}),
        alphabet=frozenset({"a"}),
        start="q0",
        transition={(("q0", "a")): "q0"},
    )


def dfa_epsilon_only() -> DFA[str, str]:
    """DFA whose only accepted word is ε."""
    return DFA(
        states=frozenset({"q0", "dead"}),
        alphabet=frozenset({"a"}),
        start="q0",
        transition={("q0", "a"): "dead", ("dead", "a"): "dead"},
        accept=frozenset({"q0"}),
    )


def dfa_single_a() -> DFA[str, str]:
    """DFA that accepts exactly the word 'a'."""
    return DFA(
        states=frozenset({"q0", "q1", "dead"}),
        alphabet=frozenset({"a"}),
        start="q0",
        transition={
            ("q0", "a"): "q1",
            ("q1", "a"): "dead",
            ("dead", "a"): "dead",
        },
        accept=frozenset({"q1"}),
    )


def dfa_a_star() -> DFA[str, str]:
    """DFA that accepts a* (zero or more 'a's)."""
    return DFA(
        states=frozenset({"q0"}),
        alphabet=frozenset({"a"}),
        start="q0",
        transition={("q0", "a"): "q0"},
        accept=frozenset({"q0"}),
    )


def dfa_ab() -> DFA[str, str]:
    """DFA that accepts exactly the word 'ab'."""
    return DFA(
        states=frozenset({"q0", "q1", "q2", "dead"}),
        alphabet=frozenset({"a", "b"}),
        start="q0",
        transition={
            ("q0", "a"): "q1", ("q0", "b"): "dead",
            ("q1", "a"): "dead", ("q1", "b"): "q2",
            ("q2", "a"): "dead", ("q2", "b"): "dead",
            ("dead", "a"): "dead", ("dead", "b"): "dead",
        },
        accept=frozenset({"q2"}),
    )


def dfa_ends_with_a() -> DFA[str, str]:
    """DFA accepting (a|b)*a — strings over {a,b} ending with 'a'."""
    return DFA(
        states=frozenset({"q0", "q1"}),
        alphabet=frozenset({"a", "b"}),
        start="q0",
        transition={
            ("q0", "a"): "q1", ("q0", "b"): "q0",
            ("q1", "a"): "q1", ("q1", "b"): "q0",
        },
        accept=frozenset({"q1"}),
    )


def dfa_even_as() -> DFA[str, str]:
    """DFA accepting strings over {a} with an even number of 'a's."""
    return DFA(
        states=frozenset({"even", "odd"}),
        alphabet=frozenset({"a"}),
        start="even",
        transition={("even", "a"): "odd", ("odd", "a"): "even"},
        accept=frozenset({"even"}),
    )


# ---------------------------------------------------------------------------
# Acceptance tests
# ---------------------------------------------------------------------------


class TestDFAAccepts:
    def test_empty_language_rejects_everything(self) -> None:
        dfa = dfa_empty_language()
        assert not dfa.accepts([])
        assert not dfa.accepts(["a"])
        assert not dfa.accepts(["a", "a"])

    def test_epsilon_only_accepts_empty_word(self) -> None:
        dfa = dfa_epsilon_only()
        assert dfa.accepts([])
        assert not dfa.accepts(["a"])

    def test_single_a(self) -> None:
        dfa = dfa_single_a()
        assert dfa.accepts(["a"])
        assert not dfa.accepts([])
        assert not dfa.accepts(["a", "a"])

    def test_a_star(self) -> None:
        dfa = dfa_a_star()
        assert dfa.accepts([])
        assert dfa.accepts(["a"])
        assert dfa.accepts(["a", "a", "a"])

    def test_ab_exact(self) -> None:
        dfa = dfa_ab()
        assert dfa.accepts(["a", "b"])
        assert not dfa.accepts(["a"])
        assert not dfa.accepts(["b"])
        assert not dfa.accepts(["a", "b", "a"])

    def test_ends_with_a(self) -> None:
        dfa = dfa_ends_with_a()
        assert dfa.accepts(["a"])
        assert dfa.accepts(["b", "a"])
        assert dfa.accepts(["a", "b", "a"])
        assert not dfa.accepts([])
        assert not dfa.accepts(["b"])
        assert not dfa.accepts(["a", "b"])

    def test_even_as(self) -> None:
        dfa = dfa_even_as()
        assert dfa.accepts([])
        assert dfa.accepts(["a", "a"])
        assert dfa.accepts(["a", "a", "a", "a"])
        assert not dfa.accepts(["a"])
        assert not dfa.accepts(["a", "a", "a"])

    def test_integer_states_and_symbols(self) -> None:
        """States and symbols can be any hashable — here ints."""
        dfa: DFA[int, int] = DFA(
            states=frozenset({0, 1}),
            alphabet=frozenset({0, 1}),
            start=0,
            transition={(0, 1): 1, (1, 0): 0, (1, 1): 1, (0, 0): 0},
            accept=frozenset({1}),
        )
        assert dfa.accepts([1])
        assert dfa.accepts([0, 1])
        assert not dfa.accepts([0])
        assert not dfa.accepts([1, 0])


# ---------------------------------------------------------------------------
# to_regex conversion — structure checks for trivial cases
# ---------------------------------------------------------------------------


class TestDFAToRegexStructure:
    def test_empty_language_gives_empty(self) -> None:
        regex = dfa_empty_language().to_regex()
        assert isinstance(regex, Empty)

    def test_epsilon_only_gives_epsilon(self) -> None:
        regex = dfa_epsilon_only().to_regex()
        assert isinstance(regex, Epsilon)

    def test_a_star_gives_star_of_symbol(self) -> None:
        regex = dfa_a_star().to_regex()
        assert isinstance(regex, Star)
        assert isinstance(regex.inner, Symbol)
        assert regex.inner.value == "a"


# ---------------------------------------------------------------------------
# to_regex conversion — cross-validation via backtracking simulator
# ---------------------------------------------------------------------------


TEST_WORDS: list[list[str]] = [
    [],
    ["a"],
    ["b"],
    ["a", "a"],
    ["a", "b"],
    ["b", "a"],
    ["b", "b"],
    ["a", "a", "a"],
    ["a", "b", "a"],
    ["b", "a", "b"],
    ["b", "b", "b"],
    ["a", "a", "b", "a"],
    ["b", "b", "a", "b"],
]


@pytest.mark.parametrize("word", TEST_WORDS)
def test_ends_with_a_regex_matches_dfa(word: list[str]) -> None:
    dfa = dfa_ends_with_a()
    regex = dfa.to_regex()
    assert dfa.accepts(word) == regex_accepts(regex, word)


@pytest.mark.parametrize("word", TEST_WORDS)
def test_ab_regex_matches_dfa(word: list[str]) -> None:
    dfa = dfa_ab()
    regex = dfa.to_regex()
    assert dfa.accepts(word) == regex_accepts(regex, word)


EVEN_A_WORDS: list[list[str]] = [
    [],
    ["a"],
    ["a", "a"],
    ["a", "a", "a"],
    ["a", "a", "a", "a"],
]


@pytest.mark.parametrize("word", EVEN_A_WORDS)
def test_even_as_regex_matches_dfa(word: list[str]) -> None:
    dfa = dfa_even_as()
    regex = dfa.to_regex()
    assert dfa.accepts(word) == regex_accepts(regex, word)
