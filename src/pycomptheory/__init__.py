"""pycomptheory – computational theory toolkit.

Quick start
-----------
>>> from pycomptheory import DFA, NFA, EpsilonNFA
>>> from pycomptheory import RegEx, Empty, Epsilon, Symbol, Union, Concat, Star

Build automata, convert them, and traverse the resulting regular expression:

>>> # DFA that accepts strings over {a, b} ending with 'a'
>>> dfa = DFA(
...     states=frozenset({"q0", "q1"}),
...     alphabet=frozenset({"a", "b"}),
...     transition={("q0", "a"): "q1", ("q0", "b"): "q0",
...                 ("q1", "a"): "q1", ("q1", "b"): "q0"},
...     start="q0",
...     accept=frozenset({"q1"}),
... )
>>> regex = dfa.to_regex()
>>> match regex:
...     case Star(_): print("Kleene star at root")
...     case Union(_, _): print("union at root")
...     case _: print(type(regex).__name__)
"""

from pycomptheory.automata.dfa import DFA
from pycomptheory.automata.enfa import EpsilonNFA
from pycomptheory.automata.fta import FTA
from pycomptheory.automata.nfa import NFA
from pycomptheory.automata.tree import Tree
from pycomptheory.regex.ast import (
    Concat,
    Empty,
    Epsilon,
    RegEx,
    Star,
    Symbol,
    Union,
)
from pycomptheory.regex.rte import (
    RTE,
    RTEConstant,
    RTEConstructor,
    RTEEmpty,
    RTEProduct,
    RTEStar,
    RTEStateVar,
    RTEUnion,
    alphabetic_width,
    contains_constant,
    rte_repr,
    substitute_var,
    to_fta,
)

__all__ = [
    # Automata
    "DFA",
    "NFA",
    "EpsilonNFA",
    "FTA",
    "Tree",
    # RegEx AST nodes
    "RegEx",
    "Empty",
    "Epsilon",
    "Symbol",
    "Union",
    "Concat",
    "Star",
    # Regular tree expression nodes and helpers
    "RTE",
    "RTEEmpty",
    "RTEConstant",
    "RTEConstructor",
    "RTEUnion",
    "RTEProduct",
    "RTEStar",
    "RTEStateVar",
    "rte_repr",
    "alphabetic_width",
    "substitute_var",
    "contains_constant",
    "to_fta",
]
