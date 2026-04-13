"""Tests for Tree, FTA, and RTE — see claude-guidelines.txt Part 3."""

from __future__ import annotations

import enum

import pytest
from frozendict import frozendict

from pycomptheory import (
    FTA,
    RTE,
    RTEConstant,
    RTEConstructor,
    RTEEmpty,
    RTEProduct,
    RTEStar,
    RTEStateVar,
    RTEUnion,
    Tree,
    alphabetic_width,
    contains_constant,
    rte_repr,
    substitute_var,
    to_fta,
)


# ---------------------------------------------------------------------------
# 1. Tree with non-string symbols
# ---------------------------------------------------------------------------


def test_tree_with_integer_symbols() -> None:
    t = Tree(2, (Tree(0, ()), Tree(1, ())))
    assert t == Tree(2, (Tree(0, ()), Tree(1, ())))
    assert hash(t) == hash(Tree(2, (Tree(0, ()), Tree(1, ()))))
    # Hashable into a frozenset
    s = frozenset({t, Tree(0, ())})
    assert t in s


# ---------------------------------------------------------------------------
# 2. Tree with tuple symbols
# ---------------------------------------------------------------------------


def test_tree_with_tuple_symbols() -> None:
    t = Tree(
        ("f", "node"),
        (Tree(("a", "leaf"), ()), Tree(("a", "leaf"), ())),
    )
    assert t.symbol == ("f", "node")
    assert t.children[0].symbol == ("a", "leaf")


# ---------------------------------------------------------------------------
# 3. FTA with integer states and string symbols
# ---------------------------------------------------------------------------


def _fta_test3() -> FTA[str, int]:
    sigma: frozendict[str, int] = frozendict({"a": 0, "b": 0, "f": 2})
    return FTA(
        states={0, 1, 2},
        sigma=sigma,
        final={2},
        delta=[("a", 0, ()), ("b", 1, ()), ("f", 2, (0, 0))],
    )


def test_fta_basic_accept_reject() -> None:
    fta = _fta_test3()
    assert fta.accepts(Tree("f", (Tree("a", ()), Tree("a", ())))) is True
    assert fta.accepts(Tree("f", (Tree("a", ()), Tree("b", ())))) is False


# ---------------------------------------------------------------------------
# 4. FTA with tuple states
# ---------------------------------------------------------------------------


def test_fta_with_tuple_states() -> None:
    sigma = frozendict({"a": 0, "b": 0, "f": 2})
    qa, qb, qf = ("q", "a"), ("q", "b"), ("q", "f")
    fta: FTA[str, tuple[str, str]] = FTA(
        states={qa, qb, qf},
        sigma=sigma,
        final={qf},
        delta=[("a", qa, ()), ("b", qb, ()), ("f", qf, (qa, qa))],
    )
    assert fta.accepts(Tree("f", (Tree("a", ()), Tree("a", ())))) is True
    assert fta.accepts(Tree("f", (Tree("a", ()), Tree("b", ())))) is False


# ---------------------------------------------------------------------------
# 5. run() on a deeper tree
# ---------------------------------------------------------------------------


def test_run_on_deeper_tree() -> None:
    fta = _fta_test3()
    t = Tree(
        "f",
        (
            Tree("f", (Tree("a", ()), Tree("a", ()))),
            Tree("a", ()),
        ),
    )
    assert 2 not in fta.run(t)


# ---------------------------------------------------------------------------
# 6. is_deterministic()
# ---------------------------------------------------------------------------


def test_is_deterministic() -> None:
    fta = _fta_test3()
    assert fta.is_deterministic() is True

    sigma = frozendict({"a": 0, "b": 0, "f": 2})
    nfta: FTA[str, int] = FTA(
        states={0, 1, 2, 99},
        sigma=sigma,
        final={2},
        delta=[
            ("a", 0, ()),
            ("a", 99, ()),
            ("b", 1, ()),
            ("f", 2, (0, 0)),
        ],
    )
    assert nfta.is_deterministic() is False


# ---------------------------------------------------------------------------
# 7. determinize() equivalence
# ---------------------------------------------------------------------------


def test_determinize_equivalence() -> None:
    sigma = frozendict({"a": 0, "b": 0, "f": 2})
    nfta: FTA[str, str] = FTA(
        states={"p", "q", "r"},
        sigma=sigma,
        final={"r"},
        delta=[
            ("a", "p", ()),
            ("a", "q", ()),  # non-deterministic on 'a'
            ("b", "q", ()),
            ("f", "r", ("p", "q")),
            ("f", "r", ("q", "p")),
        ],
    )
    dfta = nfta.determinize()
    assert dfta.is_deterministic()

    trees: list[Tree[str]] = [
        Tree("a", ()),
        Tree("b", ()),
        Tree("f", (Tree("a", ()), Tree("a", ()))),
        Tree("f", (Tree("a", ()), Tree("b", ()))),
        Tree("f", (Tree("b", ()), Tree("a", ()))),
        Tree("f", (Tree("b", ()), Tree("b", ()))),
    ]
    for t in trees:
        assert nfta.accepts(t) == dfta.accepts(t)

    for s in dfta.states:
        assert isinstance(s, frozenset)


# ---------------------------------------------------------------------------
# 8. FTA immutability and __slots__
# ---------------------------------------------------------------------------


def test_fta_immutability_and_slots() -> None:
    fta = _fta_test3()
    with pytest.raises((AttributeError, TypeError)):
        fta.states = frozenset()  # type: ignore[misc]
    assert not hasattr(fta, "__dict__")


# ---------------------------------------------------------------------------
# 9. rte_repr with non-string symbols
# ---------------------------------------------------------------------------


def test_rte_repr_non_string_symbols() -> None:
    E: RTE[tuple[str, int], str] = RTEUnion(
        RTEConstructor(
            ("f", 1),
            (RTEConstant(("a", 0)), RTEConstant(("b", 0))),
        ),
        RTEEmpty(),
    )
    r = rte_repr(E)
    assert "∅" in r
    assert repr(("f", 1)) in r


# ---------------------------------------------------------------------------
# 10. alphabetic_width
# ---------------------------------------------------------------------------


def test_alphabetic_width() -> None:
    E: RTE[str, str] = RTEConstructor(
        "f", (RTEConstant("a"), RTEConstant("b"))
    )
    assert alphabetic_width(E) == 3

    star: RTE[str, str] = RTEStar(RTEConstant("a"), "a")
    assert alphabetic_width(star) == 1


# ---------------------------------------------------------------------------
# 11. contains_constant
# ---------------------------------------------------------------------------


def test_contains_constant() -> None:
    assert contains_constant(RTEConstant("a"), "a") is True
    assert contains_constant(RTEConstant("b"), "a") is False
    assert contains_constant(RTEEmpty(), "a") is False
    assert contains_constant(RTEStar(RTEConstant("a"), "a"), "a") is True

    prod: RTE[str, str] = RTEProduct(
        RTEConstant("a"), RTEConstant("b"), "a"
    )
    assert contains_constant(prod, "b") is True
    assert contains_constant(prod, "a") is False


# ---------------------------------------------------------------------------
# 12. substitute_var
# ---------------------------------------------------------------------------


def test_substitute_var() -> None:
    E: RTE[str, str] = RTEUnion(RTEStateVar("x"), RTEConstant("a"))
    E2 = substitute_var(E, "x", RTEConstant("b"))
    assert E2 == RTEUnion(RTEConstant("b"), RTEConstant("a"))
    assert substitute_var(E, "y", RTEConstant("b")) == E


# ---------------------------------------------------------------------------
# 13. to_fta() round-trip — string symbols
# ---------------------------------------------------------------------------


def test_to_fta_round_trip_strings() -> None:
    sigma = frozendict({"a": 0, "b": 0, "f": 2})
    E: RTE[str, str] = RTEUnion(
        RTEConstructor("f", (RTEConstant("a"), RTEConstant("a"))),
        RTEConstructor("f", (RTEConstant("b"), RTEConstant("b"))),
    )
    A = to_fta(E, sigma)
    assert isinstance(A, FTA)
    assert A.accepts(Tree("f", (Tree("a", ()), Tree("a", ())))) is True
    assert A.accepts(Tree("f", (Tree("b", ()), Tree("b", ())))) is True
    assert A.accepts(Tree("f", (Tree("a", ()), Tree("b", ())))) is False


# ---------------------------------------------------------------------------
# 14. to_fta() with integer symbols
# ---------------------------------------------------------------------------


def test_to_fta_integer_symbols() -> None:
    sigma = frozendict({0: 0, 1: 0, 2: 2})
    E: RTE[int, str] = RTEConstructor(
        2, (RTEConstant(0), RTEConstant(0))
    )
    A = to_fta(E, sigma)
    assert A.accepts(Tree(2, (Tree(0, ()), Tree(0, ())))) is True
    assert A.accepts(Tree(2, (Tree(0, ()), Tree(1, ())))) is False


# ---------------------------------------------------------------------------
# 15. to_fta() state count bound
# ---------------------------------------------------------------------------


def test_to_fta_state_count_bound() -> None:
    sigma = frozendict({"a": 0, "b": 0, "f": 2})
    E: RTE[str, str] = RTEConstructor(
        "f", (RTEConstant("a"), RTEConstant("b"))
    )
    A = to_fta(E, sigma)
    assert len(A.states) <= alphabetic_width(E)


# ---------------------------------------------------------------------------
# 16. FTA.to_rte() → to_fta() round-trip (acyclic)
# ---------------------------------------------------------------------------


def test_to_rte_round_trip_acyclic() -> None:
    fta = _fta_test3()
    E = fta.to_rte()
    A2 = to_fta(E, fta.sigma)
    trees = [
        Tree("f", (Tree(l, ()), Tree(r, ())))
        for l in ("a", "b")
        for r in ("a", "b")
    ]
    for t in trees:
        assert fta.accepts(t) == A2.accepts(t)


# ---------------------------------------------------------------------------
# 17. FTA with enum states
# ---------------------------------------------------------------------------


class _St(enum.Enum):
    QA = 1
    QB = 2
    QF = 3


def test_fta_with_enum_states() -> None:
    sigma = frozendict({"a": 0, "b": 0, "f": 2})
    fta: FTA[str, _St] = FTA(
        states={_St.QA, _St.QB, _St.QF},
        sigma=sigma,
        final={_St.QF},
        delta=[
            ("a", _St.QA, ()),
            ("b", _St.QB, ()),
            ("f", _St.QF, (_St.QA, _St.QA)),
        ],
    )
    assert fta.accepts(Tree("f", (Tree("a", ()), Tree("a", ())))) is True
    assert fta.accepts(Tree("f", (Tree("a", ()), Tree("b", ())))) is False
