"""Regular tree expressions (RTE).

A regular tree expression denotes a set of ground terms over a ranked
alphabet.  The grammar (TATA §2.2 and Kuske & Meinecke 2008) is::

    E ::= ∅                        -- empty language
        | c                        -- singleton {c} for a constant c
        | f(E₁, …, Eₙ)             -- term constructor
        | E + E                    -- union
        | E ·h E                   -- hole-product: substitute right for
                                      every leaf h in left
        | E *h                     -- hole-iteration: iterated product at h
        | X_q                      -- state variable (equation solving only)

Every node is a frozen, slotted dataclass subclass of :class:`RTE`, so the
AST is hashable, immutable, and suitable as a :class:`frozenset` element.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from frozendict import frozendict

if TYPE_CHECKING:  # pragma: no cover
    from pycomptheory.automata.fta import FTA


# ---------------------------------------------------------------------------
# Node hierarchy
# ---------------------------------------------------------------------------


class RTE[S: Hashable, Q: Hashable]:
    """Abstract base for regular tree expression nodes.

    Do not instantiate directly; use one of the concrete subclasses.  The
    ``Q`` type parameter only appears in :class:`RTEStateVar`; every other
    node is polymorphic in it and is reused unchanged during state
    elimination.
    """

    __slots__ = ()


@dataclass(frozen=True, slots=True)
class RTEEmpty[S: Hashable, Q: Hashable](RTE[S, Q]):
    """The empty language ∅."""


@dataclass(frozen=True, slots=True)
class RTEConstant[S: Hashable, Q: Hashable](RTE[S, Q]):
    """Singleton language ``{c}`` for a constant (arity-0) symbol ``c``."""

    symbol: S


@dataclass(frozen=True, slots=True)
class RTEConstructor[S: Hashable, Q: Hashable](RTE[S, Q]):
    """Term constructor language ``{ f(t₁,…,tₙ) : tᵢ ∈ L(children[i]) }``.

    The arity of ``symbol`` must equal ``len(children)``.
    """

    symbol: S
    children: tuple[RTE[S, Q], ...]


@dataclass(frozen=True, slots=True)
class RTEUnion[S: Hashable, Q: Hashable](RTE[S, Q]):
    """Alternation ``L(left) ∪ L(right)``."""

    left: RTE[S, Q]
    right: RTE[S, Q]


@dataclass(frozen=True, slots=True)
class RTEProduct[S: Hashable, Q: Hashable](RTE[S, Q]):
    """Hole-product ``left ·hole right``.

    Semantics: the set of trees obtained from a tree ``t ∈ L(left)`` by
    replacing every leaf labelled ``hole`` with a (possibly distinct) tree
    drawn from ``L(right)`` (TATA Def. 2.2.1).
    """

    left: RTE[S, Q]
    right: RTE[S, Q]
    hole: S


@dataclass(frozen=True, slots=True)
class RTEStar[S: Hashable, Q: Hashable](RTE[S, Q]):
    """Hole-iteration ``expr *hole`` — iterated product at ``hole``
    (TATA Def. 2.2.1).  The base case ``L⁰,h = {h}`` means ``h`` itself is
    always in the language."""

    expr: RTE[S, Q]
    hole: S


@dataclass(frozen=True, slots=True)
class RTEStateVar[S: Hashable, Q: Hashable](RTE[S, Q]):
    """Placeholder ``X_q`` for a state variable, used solely as an
    unknown during the state-elimination procedure in
    :meth:`pycomptheory.automata.fta.FTA.to_rte`."""

    name: Q


# ---------------------------------------------------------------------------
# Module-level operations (kept outside the nodes so the AST stays pure data)
# ---------------------------------------------------------------------------


def rte_repr[S: Hashable, Q: Hashable](expr: RTE[S, Q]) -> str:
    """Return a human-readable string form of *expr*."""
    match expr:
        case RTEEmpty():
            return "∅"
        case RTEConstant(symbol):
            return repr(symbol)
        case RTEConstructor(symbol, children):
            inner = ", ".join(rte_repr(c) for c in children)
            return f"{repr(symbol)}({inner})"
        case RTEUnion(left, right):
            return f"({rte_repr(left)} + {rte_repr(right)})"
        case RTEProduct(left, right, hole):
            return f"({rte_repr(left)} .{repr(hole)} {rte_repr(right)})"
        case RTEStar(inner, hole):
            return f"({rte_repr(inner)})*{repr(hole)}"
        case RTEStateVar(name):
            return f"X_{repr(name)}"
        case _:  # pragma: no cover
            raise TypeError(f"unknown RTE node: {type(expr).__name__}")


def alphabetic_width[S: Hashable, Q: Hashable](expr: RTE[S, Q]) -> int:
    """Count the number of symbol occurrences in *expr*.

    State variables and the empty language contribute zero; every
    :class:`RTEConstant`, :class:`RTEConstructor` node contributes one,
    and all structural combinators recurse.
    """
    match expr:
        case RTEEmpty() | RTEStateVar():
            return 0
        case RTEConstant():
            return 1
        case RTEConstructor(_, children):
            return 1 + sum(alphabetic_width(c) for c in children)
        case RTEUnion(left, right) | RTEProduct(left, right, _):
            return alphabetic_width(left) + alphabetic_width(right)
        case RTEStar(inner, _):
            return alphabetic_width(inner)
        case _:  # pragma: no cover
            raise TypeError(f"unknown RTE node: {type(expr).__name__}")


def substitute_var[S: Hashable, Q: Hashable](
    expr: RTE[S, Q], var: Q, replacement: RTE[S, Q]
) -> RTE[S, Q]:
    """Replace every ``RTEStateVar(var)`` in *expr* with *replacement*."""
    match expr:
        case RTEEmpty() | RTEConstant():
            return expr
        case RTEStateVar(name):
            return replacement if name == var else expr
        case RTEConstructor(symbol, children):
            return RTEConstructor(
                symbol,
                tuple(substitute_var(c, var, replacement) for c in children),
            )
        case RTEUnion(left, right):
            return RTEUnion(
                substitute_var(left, var, replacement),
                substitute_var(right, var, replacement),
            )
        case RTEProduct(left, right, hole):
            return RTEProduct(
                substitute_var(left, var, replacement),
                substitute_var(right, var, replacement),
                hole,
            )
        case RTEStar(inner, hole):
            return RTEStar(substitute_var(inner, var, replacement), hole)
        case _:  # pragma: no cover
            raise TypeError(f"unknown RTE node: {type(expr).__name__}")


def contains_constant[S: Hashable, Q: Hashable](
    expr: RTE[S, Q], c: S
) -> bool:
    """Return ``True`` iff the singleton tree ``Tree(c, ())`` is in
    ⟦*expr*⟧ (TATA Lemma 2.2.3).

    This is the key predicate used by the partial-derivative construction
    to decide whether a hole has already been "filled".
    """
    match expr:
        case RTEEmpty():
            return False
        case RTEConstant(symbol):
            return symbol == c
        case RTEConstructor():
            return False
        case RTEUnion(left, right):
            return contains_constant(left, c) or contains_constant(right, c)
        case RTEProduct(left, right, hole):
            return (hole != c and contains_constant(left, c)) or (
                contains_constant(left, hole) and contains_constant(right, c)
            )
        case RTEStar(inner, hole):
            return hole == c or contains_constant(inner, c)
        case RTEStateVar():
            return False
        case _:  # pragma: no cover
            raise TypeError(f"unknown RTE node: {type(expr).__name__}")


# ---------------------------------------------------------------------------
# Partial-derivative construction: RTE → FTA
# ---------------------------------------------------------------------------


def _linearize[S: Hashable, Q: Hashable](
    expr: RTE[S, Q],
    sigma: frozendict[S, int],
) -> tuple[
    RTE[tuple[S, int], Q],
    dict[tuple[S, int], S],
    dict[tuple[S, int], int],
]:
    """Rename every non-constant constructor occurrence to ``(f, k)`` with
    a fresh ``k``, while collapsing every constant ``c`` to the single
    linearized symbol ``(c, 0)`` (TATA §2.2.2).
    """
    counter: dict[S, int] = {}
    eta: dict[tuple[S, int], S] = {}
    sigma_hat: dict[tuple[S, int], int] = {}

    def fresh_ctor(sym: S) -> tuple[S, int]:
        counter[sym] = counter.get(sym, 0) + 1
        k = counter[sym]
        hat = (sym, k)
        eta[hat] = sym
        sigma_hat[hat] = sigma[sym]
        return hat

    def reg_const(sym: S) -> tuple[S, int]:
        hat = (sym, 0)
        eta[hat] = sym
        sigma_hat[hat] = 0
        return hat

    def go(e: RTE[S, Q]) -> RTE[tuple[S, int], Q]:
        match e:
            case RTEEmpty():
                return RTEEmpty()
            case RTEConstant(symbol):
                return RTEConstant(reg_const(symbol))
            case RTEConstructor(symbol, children):
                # Linearize children *before* assigning the fresh index so
                # that inner occurrences get lower indices; any consistent
                # ordering would work.
                hat = fresh_ctor(symbol)
                return RTEConstructor(hat, tuple(go(c) for c in children))
            case RTEUnion(left, right):
                return RTEUnion(go(left), go(right))
            case RTEProduct(left, right, hole):
                return RTEProduct(go(left), go(right), reg_const(hole))
            case RTEStar(inner, hole):
                return RTEStar(go(inner), reg_const(hole))
            case RTEStateVar(name):
                return RTEStateVar(name)
            case _:  # pragma: no cover
                raise TypeError(f"unknown RTE node: {type(e).__name__}")

    return go(expr), eta, sigma_hat


def _g_inv[S: Hashable, Q: Hashable](
    expr: RTE[S, Q], g: S
) -> frozenset[tuple[RTE[S, Q], ...]]:
    """Partial derivative of *expr* with respect to the top symbol *g*
    (TATA §2.2.2): return the set of tuples of sub-expressions ``E₁ … Eₙ``
    such that ``g(E₁, …, Eₙ) ⊑ expr`` in the sense that any tree
    ``g(t₁, …, tₙ)`` with ``tᵢ ∈ L(Eᵢ)`` is in ``L(expr)``.
    """
    match expr:
        case RTEEmpty() | RTEConstant() | RTEStateVar():
            return frozenset()
        case RTEConstructor(symbol, children):
            if symbol == g:
                return frozenset({tuple(children)})
            return frozenset()
        case RTEUnion(left, right):
            return _g_inv(left, g) | _g_inv(right, g)
        case RTEProduct(left, right, hole):
            from_left = frozenset(
                tuple(RTEProduct(gi, right, hole) for gi in tup)
                for tup in _g_inv(left, g)
            )
            if contains_constant(left, hole):
                return from_left | _g_inv(right, g)
            return from_left
        case RTEStar(inner, hole):
            return frozenset(
                tuple(
                    RTEProduct(gi, RTEStar(inner, hole), hole) for gi in tup
                )
                for tup in _g_inv(inner, g)
            )
        case _:  # pragma: no cover
            raise TypeError(f"unknown RTE node: {type(expr).__name__}")


def to_fta[S: Hashable, Q: Hashable](
    expr: RTE[S, Q], sigma: frozendict[S, int]
) -> FTA[S, int]:
    """Convert *expr* to an equivalent :class:`~pycomptheory.automata.fta.FTA`
    using the partial-derivative construction (TATA §2.2.2 and Kuske &
    Meinecke 2008).

    The returned FTA has integer states, assigned in BFS order from the
    linearized root expression; its single accepting state is the one
    associated with the root.
    """
    # Late import to break the regex ↔ automata cycle.
    from pycomptheory.automata.fta import FTA

    sigma_fd: frozendict[S, int] = frozendict(sigma)
    expr_hat, eta, sigma_hat = _linearize(expr, sigma_fd)

    expr_to_id: dict[RTE[tuple[S, int], Q], int] = {expr_hat: 0}
    worklist: list[RTE[tuple[S, int], Q]] = [expr_hat]
    counter = 1

    non_const_symbols = tuple(g for g, a in sigma_hat.items() if a >= 1)
    const_symbols = tuple(g for g, a in sigma_hat.items() if a == 0)

    while worklist:
        current = worklist.pop()
        for g in non_const_symbols:
            for tup in _g_inv(current, g):
                for gi in tup:
                    if gi not in expr_to_id:
                        expr_to_id[gi] = counter
                        counter += 1
                        worklist.append(gi)

    transitions: set[tuple[S, int, tuple[int, ...]]] = set()
    for e_hat, state_id in expr_to_id.items():
        for g in non_const_symbols:
            for tup in _g_inv(e_hat, g):
                child_ids = tuple(expr_to_id[gi] for gi in tup)
                transitions.add((eta[g], state_id, child_ids))
        for c_hat in const_symbols:
            if contains_constant(e_hat, c_hat):
                transitions.add((eta[c_hat], state_id, ()))

    all_states = frozenset(expr_to_id.values())
    final = frozenset({expr_to_id[expr_hat]})
    return FTA(all_states, sigma_fd, final, transitions)
