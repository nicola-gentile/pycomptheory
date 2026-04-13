"""Regular expression abstract syntax tree.

Every node is a frozen dataclass, so the whole tree is hashable and
immutable.  Pattern-match on the ``RegEx[A]`` union to traverse it:

    match regex:
        case Empty():       ...
        case Epsilon():     ...
        case Symbol(value): ...
        case Union(l, r):   ...
        case Concat(l, r):  ...
        case Star(inner):   ...
"""

from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Public type alias – the full union of all regex node kinds.
# ---------------------------------------------------------------------------

type RegEx[A] = Empty | Epsilon | Symbol[A] | Union[A] | Concat[A] | Star[A]


# ---------------------------------------------------------------------------
# Node classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Empty:
    """The empty language ∅ – no string is accepted."""


@dataclass(frozen=True)
class Epsilon:
    """The language {ε} – only the empty string is accepted."""


@dataclass(frozen=True)
class Symbol[A]:
    """The singleton language {a} for a single alphabet symbol *a*."""

    value: A


@dataclass(frozen=True)
class Union[A]:
    """Alternation: L(left) ∪ L(right)  –  written  left | right."""

    left: RegEx[A]
    right: RegEx[A]


@dataclass(frozen=True)
class Concat[A]:
    """Concatenation: L(left) · L(right)  –  written  left right."""

    left: RegEx[A]
    right: RegEx[A]


@dataclass(frozen=True)
class Star[A]:
    """Kleene closure: L(inner)*  –  written  inner*."""

    inner: RegEx[A]
