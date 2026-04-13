"""Tests for the RegEx AST node types."""

from __future__ import annotations

import pytest

from pycomptheory import Concat, Empty, Epsilon, Star, Symbol, Union


class TestNodeCreation:
    def test_empty(self) -> None:
        node = Empty()
        assert isinstance(node, Empty)

    def test_epsilon(self) -> None:
        node = Epsilon()
        assert isinstance(node, Epsilon)

    def test_symbol(self) -> None:
        node: Symbol[str] = Symbol("a")
        assert node.value == "a"

    def test_symbol_integer(self) -> None:
        node: Symbol[int] = Symbol(42)
        assert node.value == 42

    def test_symbol_tuple(self) -> None:
        node: Symbol[tuple[int, int]] = Symbol((1, 2))
        assert node.value == (1, 2)

    def test_union(self) -> None:
        node: Union[str] = Union(Symbol("a"), Symbol("b"))
        assert node.left == Symbol("a")
        assert node.right == Symbol("b")

    def test_concat(self) -> None:
        node: Concat[str] = Concat(Symbol("a"), Symbol("b"))
        assert node.left == Symbol("a")
        assert node.right == Symbol("b")

    def test_star(self) -> None:
        node: Star[str] = Star(Symbol("a"))
        assert node.inner == Symbol("a")


class TestImmutabilityAndHashing:
    def test_empty_is_frozen(self) -> None:
        e = Empty()
        with pytest.raises(Exception):
            e.nonexistent = "x"  # type: ignore[attr-defined]

    def test_symbol_is_frozen(self) -> None:
        s: Symbol[str] = Symbol("a")
        with pytest.raises(Exception):
            s.value = "b"  # type: ignore[misc]

    def test_nodes_are_hashable(self) -> None:
        nodes = {Empty(), Epsilon(), Symbol("a"), Union(Symbol("a"), Symbol("b"))}
        assert len(nodes) == 4

    def test_equal_nodes_are_same_hash(self) -> None:
        assert hash(Symbol("a")) == hash(Symbol("a"))
        assert hash(Star(Symbol("a"))) == hash(Star(Symbol("a")))

    def test_nodes_usable_as_dict_keys(self) -> None:
        d = {Symbol("a"): 1, Symbol("b"): 2, Empty(): 0}
        assert d[Symbol("a")] == 1
        assert d[Empty()] == 0


class TestPatternMatching:
    def test_match_empty(self) -> None:
        r = Empty()
        match r:
            case Empty():
                matched = "empty"
            case _:
                matched = "other"
        assert matched == "empty"

    def test_match_epsilon(self) -> None:
        r = Epsilon()
        match r:
            case Epsilon():
                matched = "epsilon"
            case _:
                matched = "other"
        assert matched == "epsilon"

    def test_match_symbol_extracts_value(self) -> None:
        r: Symbol[str] = Symbol("x")
        match r:
            case Symbol(value):
                captured = value
            case _:
                captured = None
        assert captured == "x"

    def test_match_union_extracts_children(self) -> None:
        r: Union[str] = Union(Symbol("a"), Symbol("b"))
        match r:
            case Union(left, right):
                assert left == Symbol("a")
                assert right == Symbol("b")
            case _:
                pytest.fail("union not matched")

    def test_match_concat_extracts_children(self) -> None:
        r: Concat[str] = Concat(Symbol("a"), Symbol("b"))
        match r:
            case Concat(left, right):
                assert left == Symbol("a")
                assert right == Symbol("b")
            case _:
                pytest.fail("concat not matched")

    def test_match_star_extracts_inner(self) -> None:
        r: Star[str] = Star(Symbol("a"))
        match r:
            case Star(inner):
                assert inner == Symbol("a")
            case _:
                pytest.fail("star not matched")

    def test_exhaustive_traversal(self) -> None:
        # Build tree: (a|b)*c
        tree = Concat(Star(Union(Symbol("a"), Symbol("b"))), Symbol("c"))

        def node_count(r: object) -> int:
            match r:
                case Empty() | Epsilon():
                    return 1
                case Symbol(_):
                    return 1
                case Union(l, rr) | Concat(l, rr):
                    return 1 + node_count(l) + node_count(rr)
                case Star(inner):
                    return 1 + node_count(inner)
                case _:
                    return 0

        # Concat=1, Star=1, Union=1, Symbol(a)=1, Symbol(b)=1, Symbol(c)=1 → 6
        assert node_count(tree) == 6
