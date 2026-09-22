"""Safe expression language shared with the JavaScript engine (SPEC.md §3).

Hand-written recursive-descent parser; no ``eval``. An expression is parsed
once into a tree of closures and evaluated many times with different
variable bindings.

Grammar (lowest to highest precedence):

    ternary  := or ( '?' ternary ':' ternary )?
    or       := and ( ('or' | '||') and )*
    and      := not ( ('and' | '&&') not )*
    not      := ('not' | '!') not | compare
    compare  := sum ( ('<' | '<=' | '>' | '>=' | '==' | '!=') sum )?
    sum      := product ( ('+' | '-') product )*
    product  := unary ( ('*' | '/' | '%') unary )*
    unary    := '-' unary | power
    power    := atom ( '^' unary )?
    atom     := number | identifier | identifier '(' args ')' | '(' expr ')'
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

Env = Dict[str, float]


class ExprError(ValueError):
    """Raised for a syntax error or an evaluation error."""


def _round_half_away(a: float) -> float:
    return math.floor(a + 0.5) if a >= 0 else -math.floor(-a + 0.5)


def _check_finite(name: str, v: float) -> float:
    if not math.isfinite(v):
        raise ExprError(f"{name} produced a non-finite value ({v})")
    return v


def _fn_log(a: float) -> float:
    if a <= 0:
        raise ExprError(f"log of non-positive number ({a})")
    return math.log(a)


def _fn_sqrt(a: float) -> float:
    if a < 0:
        raise ExprError(f"sqrt of negative number ({a})")
    return math.sqrt(a)


def _fn_exp(a: float) -> float:
    try:
        return _check_finite("exp", math.exp(a))
    except OverflowError as e:
        raise ExprError(f"exp({a}) overflows") from e


FUNCTIONS: Dict[str, Callable[..., float]] = {
    "min": lambda *a: min(a),
    "max": lambda *a: max(a),
    "abs": abs,
    "floor": math.floor,
    "ceil": math.ceil,
    "round": _round_half_away,
    "sqrt": _fn_sqrt,
    "exp": _fn_exp,
    "log": _fn_log,
    "pow": lambda a, b: _pow(a, b),
    "if": lambda c, a, b: a if c != 0 else b,
}
FUNCTION_ARITY = {"min": None, "max": None, "abs": 1, "floor": 1, "ceil": 1,
                  "round": 1, "sqrt": 1, "exp": 1, "log": 1, "pow": 2, "if": 3}
RESERVED = set(FUNCTIONS) | {"t", "T", "and", "or", "not"}


def _pow(a: float, b: float) -> float:
    try:
        v = a ** b
    except (OverflowError, ZeroDivisionError) as e:
        raise ExprError(f"pow({a}, {b}) failed: {e}") from e
    if isinstance(v, complex):
        raise ExprError(f"pow({a}, {b}) is not a real number")
    return _check_finite("pow", v)


_TOKEN_RE = re.compile(
    r"\s*(?:(?P<num>\d+\.\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?|\d+(?:[eE][+-]?\d+)?)"
    r"|(?P<id>[A-Za-z_][A-Za-z0-9_]*)"
    r"|(?P<op><=|>=|==|!=|&&|\|\||[-+*/%^()<>?:,!]))"
)


@dataclass
class _Tok:
    kind: str  # 'num', 'id', 'op', 'end'
    text: str
    pos: int


def tokenize(src: str) -> List[_Tok]:
    toks: List[_Tok] = []
    pos = 0
    n = len(src)
    while pos < n:
        if src[pos].isspace():
            pos += 1
            continue
        m = _TOKEN_RE.match(src, pos)
        if not m:
            raise ExprError(f"unexpected character {src[pos]!r} at position {pos} in {src!r}")
        kind = m.lastgroup
        text = m.group(kind)
        toks.append(_Tok(kind, text, m.start(kind)))
        pos = m.end()
    toks.append(_Tok("end", "", n))
    return toks


Node = Callable[[Env], float]


class _Parser:
    def __init__(self, src: str, variables: Sequence[str]):
        self.src = src
        self.toks = tokenize(src)
        self.i = 0
        self.variables = set(variables)
        self.used: set = set()

    def peek(self) -> _Tok:
        return self.toks[self.i]

    def take(self, text: str | None = None) -> _Tok:
        tok = self.toks[self.i]
        if text is not None and tok.text != text:
            raise ExprError(f"expected {text!r} at position {tok.pos} in {self.src!r}, found {tok.text!r}")
        self.i += 1
        return tok

    def accept(self, *texts: str) -> _Tok | None:
        tok = self.toks[self.i]
        if tok.kind in ("op", "id") and tok.text in texts:
            self.i += 1
            return tok
        return None

    def parse(self) -> Node:
        node = self.ternary()
        if self.peek().kind != "end":
            tok = self.peek()
            raise ExprError(f"unexpected {tok.text!r} at position {tok.pos} in {self.src!r}")
        return node

    def ternary(self) -> Node:
        cond = self.or_()
        if self.accept("?"):
            a = self.ternary()
            self.take(":")
            b = self.ternary()
            return lambda env: a(env) if cond(env) != 0 else b(env)
        return cond

    def or_(self) -> Node:
        left = self.and_()
        while self.accept("or", "||"):
            right = self.and_()
            left = (lambda l, r: lambda env: 1.0 if (l(env) != 0 or r(env) != 0) else 0.0)(left, right)
        return left

    def and_(self) -> Node:
        left = self.not_()
        while self.accept("and", "&&"):
            right = self.not_()
            left = (lambda l, r: lambda env: 1.0 if (l(env) != 0 and r(env) != 0) else 0.0)(left, right)
        return left

    def not_(self) -> Node:
        if self.accept("not", "!"):
            inner = self.not_()
            return lambda env: 0.0 if inner(env) != 0 else 1.0
        return self.compare()

    def compare(self) -> Node:
        left = self.sum_()
        tok = self.accept("<", "<=", ">", ">=", "==", "!=")
        if tok is None:
            return left
        right = self.sum_()
        op = tok.text
        cmp = {
            "<": lambda a, b: a < b, "<=": lambda a, b: a <= b,
            ">": lambda a, b: a > b, ">=": lambda a, b: a >= b,
            "==": lambda a, b: a == b, "!=": lambda a, b: a != b,
        }[op]
        return lambda env: 1.0 if cmp(left(env), right(env)) else 0.0

    def sum_(self) -> Node:
        left = self.product()
        while True:
            tok = self.accept("+", "-")
            if tok is None:
                return left
            right = self.product()
            if tok.text == "+":
                left = (lambda l, r: lambda env: l(env) + r(env))(left, right)
            else:
                left = (lambda l, r: lambda env: l(env) - r(env))(left, right)

    def product(self) -> Node:
        left = self.unary()
        while True:
            tok = self.accept("*", "/", "%")
            if tok is None:
                return left
            right = self.unary()
            if tok.text == "*":
                left = (lambda l, r: lambda env: l(env) * r(env))(left, right)
            elif tok.text == "/":
                def div(env, l=left, r=right):
                    d = r(env)
                    if d == 0:
                        raise ExprError("division by zero")
                    return l(env) / d
                left = div
            else:
                def mod(env, l=left, r=right):
                    d = r(env)
                    if d == 0:
                        raise ExprError("modulo by zero")
                    return math.fmod(l(env), d)
                left = mod

    def unary(self) -> Node:
        if self.accept("-"):
            inner = self.unary()
            return lambda env: -inner(env)
        return self.power()

    def power(self) -> Node:
        base = self.atom()
        if self.accept("^"):
            exp = self.unary()
            return lambda env: _pow(base(env), exp(env))
        return base

    def atom(self) -> Node:
        tok = self.peek()
        if tok.kind == "num":
            self.i += 1
            v = float(tok.text)
            return lambda env, v=v: v
        if tok.kind == "id":
            self.i += 1
            name = tok.text
            if self.accept("("):
                if name not in FUNCTIONS:
                    raise ExprError(f"unknown function {name!r} at position {tok.pos} in {self.src!r}")
                args: List[Node] = []
                if not self.accept(")"):
                    args.append(self.ternary())
                    while self.accept(","):
                        args.append(self.ternary())
                    self.take(")")
                arity = FUNCTION_ARITY[name]
                if arity is None:
                    if len(args) < 1:
                        raise ExprError(f"{name}() needs at least one argument in {self.src!r}")
                elif len(args) != arity:
                    raise ExprError(f"{name}() takes {arity} argument(s), got {len(args)} in {self.src!r}")
                fn = FUNCTIONS[name]
                return lambda env: fn(*[a(env) for a in args])
            if name in ("and", "or", "not"):
                raise ExprError(f"unexpected keyword {name!r} at position {tok.pos} in {self.src!r}")
            if name not in self.variables and name not in ("t", "T"):
                raise ExprError(f"unknown variable {name!r} in {self.src!r}; known: t, T, "
                                + ", ".join(sorted(self.variables)))
            self.used.add(name)
            return lambda env: env[name]
        if tok.kind == "op" and tok.text == "(":
            self.i += 1
            inner = self.ternary()
            self.take(")")
            return inner
        raise ExprError(f"unexpected {tok.text!r} at position {tok.pos} in {self.src!r}"
                        if tok.kind != "end" else f"unexpected end of expression in {self.src!r}")


class Expression:
    """A parsed expression. ``Expression("0.4*x", ["x"])(dict(x=5, t=1, T=10))``."""

    def __init__(self, source, variables: Sequence[str]):
        if isinstance(source, bool):
            raise ExprError("boolean is not a valid expression")
        if isinstance(source, (int, float)):
            source = repr(source)
        if not isinstance(source, str):
            raise ExprError(f"expression must be a string or number, got {type(source).__name__}")
        if source.strip() == "":
            raise ExprError("empty expression")
        self.source = source
        parser = _Parser(source, variables)
        self._node = parser.parse()
        #: identifiers the expression reads (state variable names, 't', 'T')
        self.depends_on = frozenset(parser.used)

    def __call__(self, env: Env) -> float:
        v = self._node(env)
        if isinstance(v, bool):
            v = float(v)
        return _check_finite(self.source, float(v))

    def __repr__(self) -> str:
        return f"Expression({self.source!r})"


def is_valid_identifier(name: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name)) and name not in RESERVED
