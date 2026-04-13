"""Shared test helpers."""

from __future__ import annotations

from pycomptheory import Concat, Empty, Epsilon, RegEx, Star, Symbol, Union


def regex_accepts(r: RegEx[str], word: list[str]) -> bool:
    """Backtracking regex simulator used to cross-validate conversions."""
    match r:
        case Empty():
            return False
        case Epsilon():
            return len(word) == 0
        case Symbol(value):
            return word == [value]
        case Union(left, right):
            return regex_accepts(left, word) or regex_accepts(right, word)
        case Concat(left, right):
            return any(
                regex_accepts(left, word[:i]) and regex_accepts(right, word[i:])
                for i in range(len(word) + 1)
            )
        case Star(inner):
            if not word:
                return True
            return any(
                regex_accepts(inner, word[:i]) and regex_accepts(Star(inner), word[i:])
                for i in range(1, len(word) + 1)
            )
