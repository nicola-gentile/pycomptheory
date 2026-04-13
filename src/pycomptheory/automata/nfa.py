"""Nondeterministic Finite Automaton (NFA)."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, field

from pycomptheory.automata.dfa import DFA
from pycomptheory.regex.ast import RegEx


@dataclass
class NFA[S: Hashable, A: Hashable]:
    """Nondeterministic Finite Automaton (without ε-transitions).

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
        The transition relation δ : Q × Σ → 2^Q.
        Missing entries mean δ(q, a) = ∅.
    start:
        The initial state q₀ ∈ Q.
    accept:
        The set of accepting states F ⊆ Q.
    """

    states: frozenset[S]
    alphabet: frozenset[A]
    start: S
    transition: dict[tuple[S, A], frozenset[S]] = field(default_factory=dict)
    accept: frozenset[S] = field(default_factory=frozenset)

    def accepts(self, word: list[A]) -> bool:
        """Return ``True`` iff *word* is accepted by this NFA."""
        current: frozenset[S] = frozenset({self.start})
        for symbol in word:
            nxt: set[S] = set()
            for state in current:
                nxt.update(self.transition.get((state, symbol), frozenset()))
            current = frozenset(nxt)
        return bool(current & self.accept)

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def to_dfa(self) -> DFA[frozenset[S], A]:
        """Convert this NFA to an equivalent DFA via the powerset construction.

        The DFA states are :class:`frozenset` subsets of the NFA states.
        States unreachable from the start state are not included.
        """
        start_dfa: frozenset[S] = frozenset({self.start})
        worklist: list[frozenset[S]] = [start_dfa]
        visited: set[frozenset[S]] = set()
        dfa_transition: dict[tuple[frozenset[S], A], frozenset[S]] = {}

        while worklist:
            subset = worklist.pop()
            if subset in visited:
                continue
            visited.add(subset)

            for symbol in self.alphabet:
                nxt: set[S] = set()
                for state in subset:
                    nxt.update(self.transition.get((state, symbol), frozenset()))
                nxt_fs = frozenset(nxt)
                dfa_transition[(subset, symbol)] = nxt_fs
                if nxt_fs not in visited:
                    worklist.append(nxt_fs)

        dfa_states = frozenset(visited)
        dfa_accept = frozenset(s for s in dfa_states if s & self.accept)

        return DFA(
            states=dfa_states,
            alphabet=self.alphabet,
            transition=dfa_transition,
            start=start_dfa,
            accept=dfa_accept,
        )

    def to_regex(self) -> RegEx[A]:
        """Convert this NFA to an equivalent regular expression.

        Chains :meth:`to_dfa` followed by
        :meth:`~pycomptheory.automata.dfa.DFA.to_regex`.
        """
        return self.to_dfa().to_regex()
