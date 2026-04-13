"""Deterministic Finite Automaton (DFA)."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, field

from pycomptheory.regex.ast import (
    Concat,
    Empty,
    Epsilon,
    RegEx,
    Star,
    Symbol,
    Union,
)


# ---------------------------------------------------------------------------
# Smart constructors that simplify obvious cases
# ---------------------------------------------------------------------------


def _union[A](r1: RegEx[A], r2: RegEx[A]) -> RegEx[A]:
    if isinstance(r1, Empty):
        return r2
    if isinstance(r2, Empty):
        return r1
    return Union(r1, r2)


def _concat[A](r1: RegEx[A], r2: RegEx[A]) -> RegEx[A]:
    if isinstance(r1, Empty) or isinstance(r2, Empty):
        return Empty()
    if isinstance(r1, Epsilon):
        return r2
    if isinstance(r2, Epsilon):
        return r1
    return Concat(r1, r2)


def _star[A](r: RegEx[A]) -> RegEx[A]:
    if isinstance(r, (Empty, Epsilon)):
        return Epsilon()
    if isinstance(r, Star):
        return r
    return Star(r)


# ---------------------------------------------------------------------------
# DFA to RegEx via GNFA state elimination
# ---------------------------------------------------------------------------


def _dfa_to_regex[S: Hashable, A: Hashable](dfa: DFA[S, A]) -> RegEx[A]:
    """Convert a DFA to a RegEx using the GNFA state-elimination algorithm.

    Two sentinel integer indices are appended after the real states:
      * ``n``      — the new unique start state
      * ``n + 1``  — the new unique accept state
    """
    states = list(dfa.states)
    n = len(states)
    idx: dict[S, int] = {s: i for i, s in enumerate(states)}

    NEW_START = n
    NEW_ACCEPT = n + 1

    # gnfa[i, j] is the regex labelling edge i → j; absent ⟺ Empty.
    gnfa: dict[tuple[int, int], RegEx[A]] = {}

    def get(i: int, j: int) -> RegEx[A]:
        return gnfa.get((i, j), Empty())

    def put(i: int, j: int, r: RegEx[A]) -> None:
        gnfa[(i, j)] = r

    # Populate from DFA transitions.
    for (s, a), t in dfa.transition.items():
        i, j = idx[s], idx[t]
        put(i, j, _union(get(i, j), Symbol(a)))

    # ε from new start → original start.
    put(NEW_START, idx[dfa.start], Epsilon())

    # ε from each original accept state → new accept.
    for s in dfa.accept:
        put(idx[s], NEW_ACCEPT, _union(get(idx[s], NEW_ACCEPT), Epsilon()))

    # Eliminate each original state in turn.
    remaining: set[int] = set(range(n)) | {NEW_START, NEW_ACCEPT}

    for k in range(n):
        remaining.remove(k)

        r_kk = _star(get(k, k))

        for i in remaining:
            if i == NEW_ACCEPT:
                continue
            r_ik = get(i, k)
            if isinstance(r_ik, Empty):
                continue  # no path through k from i

            for j in remaining:
                if j == NEW_START:
                    continue
                r_kj = get(k, j)
                if isinstance(r_kj, Empty):
                    continue  # no path through k to j

                path = _concat(_concat(r_ik, r_kk), r_kj)
                put(i, j, _union(get(i, j), path))

    return get(NEW_START, NEW_ACCEPT)


# ---------------------------------------------------------------------------
# DFA
# ---------------------------------------------------------------------------


@dataclass
class DFA[S: Hashable, A: Hashable]:
    """Deterministic Finite Automaton.

    Type parameters
    ---------------
    S
        State type – any :class:`~collections.abc.Hashable`.
    A
        Alphabet symbol type – any :class:`~collections.abc.Hashable`.

    Attributes
    ----------
    states:
        The finite set of states Q.
    alphabet:
        The input alphabet Σ.
    transition:
        The total (or partial) transition function δ : Q × Σ → Q.
        Missing entries are treated as transitions to an implicit dead
        state from which no accepting state is reachable.
    start:
        The initial state q₀ ∈ Q.
    accept:
        The set of accepting states F ⊆ Q.
    """

    states: frozenset[S]
    alphabet: frozenset[A]
    start: S
    transition: dict[tuple[S, A], S] = field(default_factory=dict)
    accept: frozenset[S] = field(default_factory=frozenset)

    def accepts(self, word: list[A]) -> bool:
        """Return ``True`` iff *word* is accepted by this DFA."""
        state: S = self.start
        for symbol in word:
            nxt = self.transition.get((state, symbol))
            if nxt is None:
                return False
            state = nxt
        return state in self.accept

    def to_regex(self) -> RegEx[A]:
        """Convert this DFA to an equivalent :class:`~pycomptheory.regex.ast.RegEx`.

        Uses the GNFA state-elimination algorithm.  The returned value is a
        tree of :mod:`pycomptheory.regex.ast` nodes that can be traversed with
        a ``match`` statement.
        """
        return _dfa_to_regex(self)
