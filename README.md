# pycomptheory

A Python 3.12 library for computational theory: build DFAs, NFAs, ε-NFAs, and Finite Tree Automata (FTA), then convert them to regular expressions or regular tree expressions represented as traversable ASTs.

## Features

- **DFA** — deterministic finite automaton with GNFA state-elimination → RegEx
- **NFA** — nondeterministic finite automaton (powerset construction → DFA)
- **ε-NFA** — NFA with epsilon transitions (ε-closure elimination → NFA)
- **RegEx AST** — regular expressions as immutable, pattern-matchable trees (`Empty`, `Epsilon`, `Symbol`, `Union`, `Concat`, `Star`)
- **FTA** — bottom-up (non-)deterministic finite tree automaton over a ranked alphabet; supports `run`, `accepts`, `determinize` (subset construction), `minimize` (Hopcroft partition refinement), `to_rte` (state elimination → regular tree expression)
- **RTE AST** — regular tree expressions as immutable, pattern-matchable trees (`RTEEmpty`, `RTEConstant`, `RTEConstructor`, `RTEUnion`, `RTEProduct`, `RTEStar`); `to_fta` converts back via the partial-derivative construction
- States and alphabet symbols can be **any hashable type** (strings, ints, tuples, enums, …)
- Full type annotations, checked with **pyright strict**

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync --dev
```

## Usage

```python
from pycomptheory import DFA, NFA, EpsilonNFA
from pycomptheory import RegEx, Empty, Epsilon, Symbol, Union, Concat, Star
from pycomptheory import FTA, Tree
from pycomptheory import (
    RTE, RTEEmpty, RTEConstant, RTEConstructor,
    RTEUnion, RTEProduct, RTEStar,
    rte_repr, alphabetic_width, contains_constant,
    substitute_var, to_fta,
)
```

---

### Word automata (DFA / NFA / ε-NFA)

```python
# DFA accepting (a|b)*a  — strings over {a,b} ending with 'a'
dfa = DFA(
    states=frozenset({"q0", "q1"}),
    alphabet=frozenset({"a", "b"}),
    start="q0",
    transition={
        ("q0", "a"): "q1", ("q0", "b"): "q0",
        ("q1", "a"): "q1", ("q1", "b"): "q0",
    },
    accept=frozenset({"q1"}),
)

dfa.accepts(list("bba"))   # True
dfa.accepts(list("bb"))    # False
```

```python
# NFA — transition maps (state, symbol) to a frozenset of states
nfa = NFA(
    states=frozenset({0, 1}),
    alphabet=frozenset({"a", "b"}),
    start=0,
    transition={
        (0, "a"): frozenset({0, 1}),
        (0, "b"): frozenset({0}),
    },
    accept=frozenset({1}),
)
```

```python
# ε-NFA — use None as the epsilon symbol in the transition dict
enfa = EpsilonNFA(
    states=frozenset({0, 1, 2}),
    alphabet=frozenset({"a", "b"}),
    start=0,
    transition={
        (0, "a"): frozenset({1}),
        (1, None): frozenset({2}),    # ε-transition
        (2, "b"): frozenset({2}),
    },
    accept=frozenset({2}),
)
```

#### Converting word automata to regular expressions

Every word automaton exposes a `to_regex()` method returning a `RegEx[A]` tree:

```
EpsilonNFA  →(ε-closure)→  NFA  →(powerset)→  DFA  →(GNFA state elimination)→  RegEx
```

```python
regex = dfa.to_regex()

def to_str(r: RegEx[str]) -> str:
    match r:
        case Empty():    return "∅"
        case Epsilon():  return "ε"
        case Symbol(v):  return v
        case Union(l, r):  return f"({to_str(l)}|{to_str(r)})"
        case Concat(l, r): return f"{to_str(l)}{to_str(r)}"
        case Star(i):    return f"({to_str(i)})*"

print(to_str(regex))   # e.g. ((b)*(a))((a|b))*
```

---

### Tree automata (FTA)

An FTA operates on **ranked terms** (ground trees) over a ranked alphabet Σ — a mapping from symbols to their arity.

```python
from frozendict import frozendict

# Ranked alphabet: 'a' and 'b' are leaves (arity 0), 'f' is binary (arity 2)
sigma = frozendict({"a": 0, "b": 0, "f": 2})

# FTA accepting trees of the form f(a, a)
fta = FTA(
    states={0, 1, 2},
    sigma=sigma,
    final={2},
    delta=[
        ("a", 0, ()),       # a → state 0
        ("b", 1, ()),       # b → state 1
        ("f", 2, (0, 0)),   # f(state0, state0) → state 2
    ],
)

fta.accepts(Tree("f", (Tree("a"), Tree("a"))))  # True
fta.accepts(Tree("f", (Tree("a"), Tree("b"))))  # False
```

#### Determinization and minimization

```python
dfta  = fta.determinize()   # bottom-up subset construction → DFTA
mfta  = dfta.minimize()     # Hopcroft partition refinement  → minimal DFTA
```

#### Converting FTA ↔ regular tree expressions

```python
# FTA → RTE (state elimination with Arden's lemma)
expr = fta.to_rte()
print(rte_repr(expr))
# e.g. 'f'(RTEConstant('a'), RTEConstant('a'))

# RTE → FTA (partial-derivative construction)
fta2 = to_fta(expr, sigma)
```

#### Building RTEs directly

```python
from pycomptheory import RTEConstructor, RTEConstant, RTEUnion

# Trees that are either f(a,a) or f(b,b)
expr = RTEUnion(
    RTEConstructor("f", (RTEConstant("a"), RTEConstant("a"))),
    RTEConstructor("f", (RTEConstant("b"), RTEConstant("b"))),
)

fta = to_fta(expr, frozendict({"a": 0, "b": 0, "f": 2}))
```

---

## Type checking

```bash
uv run pyright src/
```

## Running tests

```bash
uv run pytest
```

## Author

Nicola Gentile — nicola.gentile.developer@gmail.com
