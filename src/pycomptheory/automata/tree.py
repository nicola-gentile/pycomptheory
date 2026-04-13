"""Ground terms (ranked trees).

A :class:`Tree` is a term over a ranked alphabet Σ (see TATA §1.1):

  * every constant *c* with Σ[c] == 0 is a tree, represented as
    ``Tree(c, ())``;
  * if ``f`` has arity ``n ≥ 1`` and ``t₁ … tₙ`` are trees, then
    ``f(t₁, …, tₙ)`` is a tree, represented as ``Tree(f, (t₁, …, tₙ))``.

Equality and hashing are structural — two trees are equal iff they have
identical root symbols and structurally equal children — so trees can be
stored freely in :class:`set`, :class:`frozenset`, or :class:`dict` keys.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Tree[S: Hashable]:
    """A ground term over a ranked alphabet.

    Parameters
    ----------
    symbol:
        The root label.  Any hashable value.
    children:
        A (possibly empty) tuple of child trees.  Its length must match the
        arity of ``symbol`` in the surrounding ranked alphabet.
    """

    symbol: S
    children: tuple[Tree[S], ...] = ()
