"""Finite Tree Automata (bottom-up).

A bottom-up FTA (TATA §1.1) is a 4-tuple ``A = (states, sigma, final, delta)``
where

  * ``states``  : :class:`frozenset` of hashable state values ``Q``;
  * ``sigma``   : :class:`frozendict.frozendict` ranked alphabet ``S → ℕ``;
  * ``final``   : :class:`frozenset` ``⊆ states`` of accepting states;
  * ``delta``   : :class:`frozenset` of transition rules of the form
    ``(symbol, target, sources)`` representing
    ``symbol(sources[0], …, sources[n-1]) → target``.

The class is immutable, slotted, and generic in both the symbol and state
types.  See :meth:`FTA.accepts`, :meth:`FTA.determinize`,
:meth:`FTA.minimize`, and :meth:`FTA.to_rte` for the main operations.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping
from itertools import product
from typing import cast

from frozendict import frozendict

from pycomptheory.automata.tree import Tree
from pycomptheory.regex.rte import (
    RTE,
    RTEConstant,
    RTEConstructor,
    RTEEmpty,
    RTEProduct,
    RTEStar,
    RTEStateVar,
    RTEUnion,
    substitute_var,
)


# ---------------------------------------------------------------------------
# Small RTE helpers local to the elimination procedure
# ---------------------------------------------------------------------------


def _runion[S: Hashable, Q: Hashable](
    left: RTE[S, Q], right: RTE[S, Q]
) -> RTE[S, Q]:
    """Smart :class:`RTEUnion` constructor — absorbs the empty language."""
    if isinstance(left, RTEEmpty):
        return right
    if isinstance(right, RTEEmpty):
        return left
    return RTEUnion(left, right)


def _contains_state_var[S: Hashable, Q: Hashable](
    expr: RTE[S, Q], var: Q
) -> bool:
    """Return ``True`` iff ``RTEStateVar(var)`` occurs anywhere in *expr*."""
    match expr:
        case RTEEmpty() | RTEConstant():
            return False
        case RTEStateVar(name):
            return name == var
        case RTEConstructor(_, children):
            return any(_contains_state_var(c, var) for c in children)
        case RTEUnion(left, right):
            return _contains_state_var(left, var) or _contains_state_var(
                right, var
            )
        case RTEProduct(left, right, _):
            return _contains_state_var(left, var) or _contains_state_var(
                right, var
            )
        case RTEStar(inner, _):
            return _contains_state_var(inner, var)
        case _:  # pragma: no cover
            raise TypeError(f"unknown RTE node: {type(expr).__name__}")


def _arden[S: Hashable, Q: Hashable](eq: RTE[S, Q], var: Q) -> RTE[S, Q]:
    """Apply the tree-Arden lemma (TATA Lemma 2.2.5) to solve
    ``X = A ·h X + B`` for ``X``, yielding ``X = A*h ·h B``.

    This function expects the self-reference to appear in the standard
    hole-product shape produced by iterated state elimination.  If the
    input contains a self-reference in any other shape (e.g. nested inside
    a constructor, as in the raw equations for a cyclic FTA), a
    :class:`NotImplementedError` is raised: a fully general rewrite from
    constructor form to hole-product form is outside the scope of this
    implementation.
    """
    # Flatten the top-level union so we can classify disjuncts.
    def flatten(e: RTE[S, Q]) -> list[RTE[S, Q]]:
        if isinstance(e, RTEUnion):
            return flatten(e.left) + flatten(e.right)
        return [e]

    disjuncts = flatten(eq)
    recursive: list[RTE[S, Q]] = []
    constant: list[RTE[S, Q]] = []
    for d in disjuncts:
        if _contains_state_var(d, var):
            recursive.append(d)
        else:
            constant.append(d)

    # Combine the non-recursive part into B.
    b_part: RTE[S, Q] = RTEEmpty()
    for c in constant:
        b_part = _runion(b_part, c)

    # Every recursive disjunct must match exactly RTEProduct(A, X_var, h).
    a_parts_by_hole: dict[S, RTE[S, Q]] = {}
    for r in recursive:
        match r:
            case RTEProduct(left, RTEStateVar(name), hole) if name == var and (
                not _contains_state_var(left, var)
            ):
                a_parts_by_hole[hole] = _runion(
                    a_parts_by_hole.get(hole, RTEEmpty()), left
                )
            case _:
                raise NotImplementedError(
                    "Cyclic FTAs with non-product self-references are not "
                    "yet supported by to_rte()."
                )

    # Apply Arden once per distinct hole.
    result = b_part
    for hole, a in a_parts_by_hole.items():
        result = RTEProduct(RTEStar(a, hole), result, hole)
    return result


# ---------------------------------------------------------------------------
# FTA class
# ---------------------------------------------------------------------------


class FTA[S: Hashable, Q: Hashable]:
    """A bottom-up finite tree automaton over ranked alphabet ``sigma``.

    All four fields are immutable: :attr:`states`, :attr:`final`, and
    :attr:`delta` are :class:`frozenset` values; :attr:`sigma` is a
    :class:`frozendict.frozendict`.  Attempting to reassign any field
    raises :class:`AttributeError`.
    """

    __slots__ = ("states", "sigma", "final", "delta")

    # Declare attribute types for static checkers.
    states: frozenset[Q]
    sigma: frozendict[S, int]
    final: frozenset[Q]
    delta: frozenset[tuple[S, Q, tuple[Q, ...]]]

    def __init__(
        self,
        states: Iterable[Q],
        sigma: Mapping[S, int],
        final: Iterable[Q],
        delta: Iterable[tuple[S, Q, Iterable[Q]]],
    ) -> None:
        set_ = object.__setattr__
        set_(self, "states", frozenset(states))
        set_(self, "sigma", frozendict(sigma))
        set_(self, "final", frozenset(final))
        set_(
            self,
            "delta",
            frozenset(
                (sym, tgt, tuple(srcs)) for sym, tgt, srcs in delta
            ),
        )

        if not self.final <= self.states:
            raise ValueError("final states must be a subset of states")
        for sym, tgt, srcs in self.delta:
            if sym not in self.sigma:
                raise ValueError(f"symbol {sym!r} not in ranked alphabet")
            if len(srcs) != self.sigma[sym]:
                raise ValueError(
                    f"rule for {sym!r} has {len(srcs)} sources, "
                    f"expected {self.sigma[sym]}"
                )
            if tgt not in self.states:
                raise ValueError(f"target state {tgt!r} not in states")
            for s in srcs:
                if s not in self.states:
                    raise ValueError(f"source state {s!r} not in states")

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(
            f"FTA is immutable; cannot reassign attribute {name!r}"
        )

    def __delattr__(self, name: str) -> None:
        raise AttributeError(
            f"FTA is immutable; cannot delete attribute {name!r}"
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"FTA(states={set(self.states)!r}, sigma={dict(self.sigma)!r}, "
            f"final={set(self.final)!r}, delta={set(self.delta)!r})"
        )

    # ------------------------------------------------------------------
    # Basic queries
    # ------------------------------------------------------------------

    def is_deterministic(self) -> bool:
        """Return ``True`` iff no two rules share the same ``(symbol, sources)``
        left-hand side — the standard bottom-up determinism condition."""
        seen: set[tuple[S, tuple[Q, ...]]] = set()
        for sym, _, srcs in self.delta:
            key = (sym, srcs)
            if key in seen:
                return False
            seen.add(key)
        return True

    def run(self, tree: Tree[S]) -> frozenset[Q]:
        """Bottom-up evaluation (TATA §1.1): return the set of states
        reachable at the root of *tree*.

        Implementation is a post-order traversal.  For a node
        ``f(t₁,…,tₙ)`` we first evaluate the children, then collect every
        ``q`` such that some ``(f, q, (q₁,…,qₙ)) ∈ delta`` with
        ``qᵢ`` in the ``i``-th child result set.
        """
        child_results = tuple(self.run(c) for c in tree.children)
        reached: set[Q] = set()
        for sym, tgt, srcs in self.delta:
            if sym != tree.symbol:
                continue
            if len(srcs) != len(child_results):
                continue
            if all(srcs[i] in child_results[i] for i in range(len(srcs))):
                reached.add(tgt)
        return frozenset(reached)

    def accepts(self, tree: Tree[S]) -> bool:
        """Return ``True`` iff ``run(tree) ∩ final`` is non-empty."""
        return bool(self.run(tree) & self.final)

    # ------------------------------------------------------------------
    # Subset construction (determinization)
    # ------------------------------------------------------------------

    def determinize(self) -> FTA[S, frozenset[Q]]:
        """Convert to an equivalent deterministic FTA by bottom-up subset
        construction (TATA §1.1.3).

        States in the result are :class:`frozenset` values over the
        original state type.  Only *reachable* state-sets are produced, via
        a fixed-point search seeded by the target sets of constant rules.
        """
        new_states: set[frozenset[Q]] = set()
        new_delta: set[tuple[S, frozenset[Q], tuple[frozenset[Q], ...]]] = set()

        # Seed with constants.
        for sym, arity in self.sigma.items():
            if arity != 0:
                continue
            tgt_set = frozenset(
                t for s, t, srcs in self.delta if s == sym and srcs == ()
            )
            new_delta.add((sym, tgt_set, ()))
            new_states.add(tgt_set)

        # Fixed-point: extend with every reachable combo until stable.
        changed = True
        while changed:
            changed = False
            current_states = tuple(new_states)
            for sym, arity in self.sigma.items():
                if arity == 0:
                    continue
                for combo in product(current_states, repeat=arity):
                    tgt_set = frozenset(
                        t
                        for s, t, srcs in self.delta
                        if s == sym
                        and len(srcs) == arity
                        and all(srcs[i] in combo[i] for i in range(arity))
                    )
                    rule = (sym, tgt_set, combo)
                    if rule not in new_delta:
                        new_delta.add(rule)
                        changed = True
                    if tgt_set not in new_states:
                        new_states.add(tgt_set)
                        changed = True

        new_final = frozenset(ss for ss in new_states if ss & self.final)
        return FTA(new_states, self.sigma, new_final, new_delta)

    # ------------------------------------------------------------------
    # Partition-refinement minimization
    # ------------------------------------------------------------------

    def minimize(self) -> FTA[S, frozenset[Q]]:
        """Hopcroft-style partition refinement on a deterministic FTA
        (TATA §1.5).  The automaton is first completed with a dead sink
        so that every ``(symbol, sources)`` pattern is defined, and then
        classes are refined until stable.  The returned FTA's states are
        the resulting partition classes (each a :class:`frozenset`).
        """
        if not self.is_deterministic():
            raise ValueError("minimize() requires a deterministic FTA")

        # Completion.  The dead state is the empty frozenset, which cannot
        # collide with any "real" original state produced by determinize.
        dead: frozenset[Q] = cast(frozenset[Q], frozenset())
        all_states: set[Q] = set(self.states)
        all_states.add(cast(Q, dead))

        delta_map: dict[tuple[S, tuple[Q, ...]], Q] = {}
        for sym, tgt, srcs in self.delta:
            delta_map[(sym, srcs)] = tgt

        for sym, arity in self.sigma.items():
            for srcs in product(all_states, repeat=arity):
                delta_map.setdefault((sym, srcs), cast(Q, dead))

        # Initial partition.
        finals_cls = frozenset(self.final)
        non_finals_cls = frozenset(all_states - self.final)
        partition: list[frozenset[Q]] = []
        if finals_cls:
            partition.append(finals_cls)
        if non_finals_cls:
            partition.append(non_finals_cls)

        def class_id(part: list[frozenset[Q]], q: Q) -> int:
            for i, cls in enumerate(part):
                if q in cls:
                    return i
            raise KeyError(q)

        # Refinement loop.
        while True:
            new_partition: list[frozenset[Q]] = []
            for cls in partition:
                groups: dict[frozenset[tuple[object, ...]], set[Q]] = {}
                for q in cls:
                    sig_items: set[tuple[object, ...]] = set()
                    for (sym, srcs), tgt in delta_map.items():
                        for i, s in enumerate(srcs):
                            if s != q:
                                continue
                            other = tuple(
                                class_id(partition, srcs[j]) if j != i else -1
                                for j in range(len(srcs))
                            )
                            tgt_cls = class_id(partition, tgt)
                            sig_items.add((sym, i, other, tgt_cls))
                    groups.setdefault(frozenset(sig_items), set()).add(q)
                for g in groups.values():
                    new_partition.append(frozenset(g))

            if set(new_partition) == set(partition):
                partition = new_partition
                break
            partition = new_partition

        class_of: dict[Q, frozenset[Q]] = {}
        for cls in partition:
            for q in cls:
                class_of[q] = cls

        new_states_ = frozenset(partition)
        new_final = frozenset(class_of[q] for q in self.final)
        new_delta: set[tuple[S, frozenset[Q], tuple[frozenset[Q], ...]]] = set()
        for (sym, srcs), tgt in delta_map.items():
            new_delta.add(
                (sym, class_of[tgt], tuple(class_of[s] for s in srcs))
            )
        return FTA(new_states_, self.sigma, new_final, new_delta)

    # ------------------------------------------------------------------
    # Regular-equation system and FTA → RTE conversion
    # ------------------------------------------------------------------

    def to_regular_equations(self) -> frozendict[Q, RTE[S, Q]]:
        """Return ``{ q ↦ rhs_q }`` where ``rhs_q`` is the union of

          * :class:`RTEConstant` ``c`` for every arity-0 rule ``c → q``;
          * :class:`RTEConstructor` ``f(X_{q₁},…,X_{qₙ})`` for every rule
            ``(f, q, (q₁,…,qₙ))`` with ``n ≥ 1``.

        :class:`RTEStateVar` nodes play the role of unknowns and are
        eliminated by :meth:`to_rte`.
        """
        eqs: dict[Q, RTE[S, Q]] = {q: RTEEmpty() for q in self.states}
        for sym, tgt, srcs in self.delta:
            arity = self.sigma[sym]
            term: RTE[S, Q]
            if arity == 0:
                term = RTEConstant(sym)
            else:
                term = RTEConstructor(
                    sym, tuple(RTEStateVar(s) for s in srcs)
                )
            eqs[tgt] = _runion(eqs[tgt], term)
        return frozendict(eqs)

    def to_rte(self) -> RTE[S, Q]:
        """Convert this FTA to an equivalent regular tree expression via
        iterative state elimination (TATA §2.2.4).

        The algorithm:

          1. Build the equation system ``{ X_q = rhs_q }`` via
             :meth:`to_regular_equations`.
          2. Accumulate the answer in a running expression seeded with
             ``⋃_{q ∈ final} X_q``.
          3. Eliminate the state variables one by one (non-finals first,
             then finals).  For each eliminated ``q`` that appears in its
             own right-hand side, apply the tree-Arden rule.  Substitute
             the resulting closed expression for ``X_q`` in every other
             equation *and* in the running answer.
          4. When every state has been eliminated the running answer is
             closed and is returned.
        """
        eqs: dict[Q, RTE[S, Q]] = dict(self.to_regular_equations())

        result: RTE[S, Q] = RTEEmpty()
        for q in self.final:
            result = _runion(result, RTEStateVar(q))

        order: list[Q] = [q for q in self.states if q not in self.final] + [
            q for q in self.states if q in self.final
        ]

        for q in order:
            eq = eqs[q]
            if _contains_state_var(eq, q):
                eq = _arden(eq, q)
            for p in list(eqs.keys()):
                if p != q:
                    eqs[p] = substitute_var(eqs[p], q, eq)
            result = substitute_var(result, q, eq)
            del eqs[q]

        return result
