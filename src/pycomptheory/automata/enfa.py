"""Nondeterministic Finite Automaton with ε-transitions (ε-NFA)."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, field

from pycomptheory.automata.dfa import DFA
from pycomptheory.automata.nfa import NFA
from pycomptheory.regex.ast import RegEx


@dataclass
class EpsilonNFA[S: Hashable, A: Hashable]:
    """Nondeterministic Finite Automaton with ε-transitions.

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
        The input alphabet Σ (must **not** include ``None``).
    transition:
        The transition relation δ : Q × (Σ ∪ {None}) → 2^Q.
        ``None`` is used as the ε symbol.
        Missing entries mean δ(q, x) = ∅.
    start:
        The initial state q₀ ∈ Q.
    accept:
        The set of accepting states F ⊆ Q.
    """

    states: frozenset[S]
    alphabet: frozenset[A]
    start: S
    transition: dict[tuple[S, A | None], frozenset[S]] = field(default_factory=dict)
    accept: frozenset[S] = field(default_factory=frozenset)

    # ------------------------------------------------------------------
    # ε-closure
    # ------------------------------------------------------------------

    def epsilon_closure(self, states: frozenset[S]) -> frozenset[S]:
        """Return the ε-closure of *states*: all states reachable via ε only."""
        closure: set[S] = set(states)
        worklist: list[S] = list(states)
        while worklist:
            s = worklist.pop()
            for t in self.transition.get((s, None), frozenset()):
                if t not in closure:
                    closure.add(t)
                    worklist.append(t)
        return frozenset(closure)

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def accepts(self, word: list[A]) -> bool:
        """Return ``True`` iff *word* is accepted by this ε-NFA."""
        current = self.epsilon_closure(frozenset({self.start}))
        for symbol in word:
            nxt: set[S] = set()
            for s in current:
                nxt.update(self.transition.get((s, symbol), frozenset()))
            current = self.epsilon_closure(frozenset(nxt))
        return bool(current & self.accept)

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def to_nfa(self) -> NFA[S, A]:
        """Remove ε-transitions and return an equivalent :class:`~pycomptheory.automata.nfa.NFA`.

        The state set is preserved; the transition function is recomputed
        using ε-closures.  A state *s* is accepting iff its ε-closure
        intersects the original accepting set.
        """
        new_transition: dict[tuple[S, A], frozenset[S]] = {}

        for state in self.states:
            ec = self.epsilon_closure(frozenset({state}))
            for symbol in self.alphabet:
                # States reached by reading *symbol* from any state in ec.
                after_symbol: set[S] = set()
                for s in ec:
                    after_symbol.update(
                        self.transition.get((s, symbol), frozenset())
                    )
                # Take ε-closure of everything reached.
                target = self.epsilon_closure(frozenset(after_symbol))
                if target:
                    new_transition[(state, symbol)] = target

        new_accept = frozenset(
            s
            for s in self.states
            if self.epsilon_closure(frozenset({s})) & self.accept
        )

        return NFA(
            states=self.states,
            alphabet=self.alphabet,
            transition=new_transition,
            start=self.start,
            accept=new_accept,
        )

    def to_dfa(self) -> DFA[frozenset[S], A]:
        """Convert to an equivalent :class:`~pycomptheory.automata.dfa.DFA`.

        Chains :meth:`to_nfa` followed by
        :meth:`~pycomptheory.automata.nfa.NFA.to_dfa`.
        """
        return self.to_nfa().to_dfa()

    def to_regex(self) -> RegEx[A]:
        """Convert to an equivalent regular expression.

        Chains :meth:`to_dfa` followed by
        :meth:`~pycomptheory.automata.dfa.DFA.to_regex`.
        """
        return self.to_dfa().to_regex()
