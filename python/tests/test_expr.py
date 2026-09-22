"""Tests for the expression language, SPEC.md §3 (dsvm/expr.py)."""

from __future__ import annotations

import math

import pytest

from dsvm.expr import Expression, ExprError, is_valid_identifier


def ev(source, variables=(), **env):
    """Parse `source` with the given variable names and evaluate with `env`."""
    return Expression(source, list(variables))(dict(env))


# --------------------------------------------------------------------------- numbers

@pytest.mark.parametrize("src,expected", [
    ("0", 0.0),
    ("1", 1.0),
    ("0.4", 0.4),
    ("2e-3", 0.002),
    ("2E-3", 0.002),
    (".5", 0.5),
    ("1.5e2", 150.0),
    ("10", 10.0),
])
def test_numbers(src, expected):
    assert ev(src) == pytest.approx(expected)


def test_int_and_float_inputs_are_accepted():
    assert ev(2) == 2.0
    assert ev(2.5) == 2.5
    assert ev(0) == 0.0


def test_bool_input_is_rejected():
    with pytest.raises(ExprError):
        Expression(True, [])
    with pytest.raises(ExprError):
        Expression(False, [])


# --------------------------------------------------------------------------- precedence

def test_precedence_add_mul():
    assert ev("2+3*4") == 14.0


def test_precedence_unary_minus_binds_looser_than_power():
    # SPEC.md §3 worked example: -2^2 == -4 (power binds tighter than unary minus)
    assert ev("-2^2") == -4.0


def test_power_is_right_associative():
    # 2^3^2 == 2^(3^2) == 2^9 == 512, not (2^3)^2 == 64
    assert ev("2^3^2") == 512.0


def test_parens_override_precedence():
    assert ev("(2+3)*4") == 20.0


# --------------------------------------------------------------------------- comparisons

@pytest.mark.parametrize("src,expected", [
    ("1<2", 1.0), ("2<1", 0.0),
    ("1<=1", 1.0), ("2<=1", 0.0),
    ("2>1", 1.0), ("1>2", 0.0),
    ("1>=1", 1.0), ("1>=2", 0.0),
    ("1==1", 1.0), ("1==2", 0.0),
    ("1!=2", 1.0), ("1!=1", 0.0),
])
def test_comparisons_return_1_or_0(src, expected):
    assert ev(src) == expected


# --------------------------------------------------------------------------- boolean ops

@pytest.mark.parametrize("src,expected", [
    ("1 and 1", 1.0), ("1 and 0", 0.0), ("0 and 0", 0.0),
    ("1 or 0", 1.0), ("0 or 0", 0.0), ("0 or 1", 1.0),
    ("not 0", 1.0), ("not 1", 0.0),
    ("1 && 1", 1.0), ("1 && 0", 0.0),
    ("1 || 0", 1.0), ("0 || 0", 0.0),
    ("!0", 1.0), ("!1", 0.0),
])
def test_boolean_ops_both_spellings(src, expected):
    assert ev(src) == expected


def test_and_or_treat_nonzero_as_true():
    assert ev("5 and 3") == 1.0
    assert ev("0 or 7") == 1.0


# --------------------------------------------------------------------------- ternary

def test_ternary_basic():
    assert ev("1 ? 10 : 20") == 10.0
    assert ev("0 ? 10 : 20") == 20.0


def test_ternary_nested():
    assert ev("1 ? (0 ? 1 : 2) : 3") == 2.0
    assert ev("0 ? 1 : (1 ? 5 : 6)") == 5.0
    # right-nesting without parens: cond ? a : cond2 ? b : c
    assert ev("0 ? 1 : 1 ? 5 : 6") == 5.0


# --------------------------------------------------------------------------- functions

def test_min_max_variadic():
    assert ev("min(3,1,2)") == 1.0
    assert ev("max(3,1,2)") == 3.0
    assert ev("min(5)") == 5.0
    assert ev("max(5)") == 5.0


def test_abs():
    assert ev("abs(-3)") == 3.0
    assert ev("abs(3)") == 3.0


def test_floor_ceil():
    assert ev("floor(2.7)") == 2.0
    assert ev("floor(-2.1)") == -3.0
    assert ev("ceil(2.1)") == 3.0
    assert ev("ceil(-2.7)") == -2.0


def test_round_half_away_from_zero():
    assert ev("round(2.5)") == 3.0
    assert ev("round(-2.5)") == -3.0
    assert ev("round(2.4)") == 2.0
    assert ev("round(-2.4)") == -2.0


def test_sqrt():
    assert ev("sqrt(4)") == 2.0
    assert ev("sqrt(0)") == 0.0


def test_exp():
    assert ev("exp(0)") == 1.0


def test_exp_below_true_overflow_threshold_is_finite():
    assert ev("exp(700)") == pytest.approx(math.exp(700))


def test_log_natural():
    assert ev("log(1)") == 0.0
    assert ev(f"log({math.e})") == pytest.approx(1.0)


def test_pow():
    assert ev("pow(2,3)") == 8.0
    assert ev("pow(2,0)") == 1.0


def test_if():
    assert ev("if(1, 10, 20)") == 10.0
    assert ev("if(0, 10, 20)") == 20.0
    assert ev("if(1<2, 100, 200)") == 100.0


# --------------------------------------------------------------------------- variables

def test_variables_t_and_T():
    assert ev("t", (), t=5.0, T=10.0) == 5.0
    assert ev("T", (), t=5.0, T=10.0) == 10.0
    assert ev("t + T", (), t=1.0, T=20.0) == 21.0


def test_state_variable():
    assert ev("x", ("x",), x=7.0) == 7.0
    assert ev("x*2 + 1", ("x",), x=3.0) == 7.0


def test_depends_on_state_and_time():
    e = Expression("x + t * T", ["x"])
    assert e.depends_on == frozenset({"x", "t", "T"})


def test_depends_on_only_used_names():
    e = Expression("x + 1", ["x", "y"])
    assert e.depends_on == frozenset({"x"})


def test_depends_on_none_for_pure_number():
    e = Expression("0.4", ["x"])
    assert e.depends_on == frozenset()


# --------------------------------------------------------------------------- errors

def test_unknown_variable_error():
    with pytest.raises(ExprError, match="unknown variable"):
        Expression("y", ["x"])


def test_unknown_function_error():
    with pytest.raises(ExprError, match="unknown function"):
        Expression("foo(1)", [])


@pytest.mark.parametrize("src", [
    "abs(1,2)",       # too many args
    "abs()",          # too few args
    "pow(1)",         # too few args
    "pow(1,2,3)",     # too many args
    "if(1,2)",        # too few args
    "sqrt()",         # too few args
    "min()",          # variadic but needs >= 1
])
def test_wrong_arity_error(src):
    with pytest.raises(ExprError):
        Expression(src, [])


def test_division_by_zero():
    with pytest.raises(ExprError, match="division by zero"):
        ev("5/0")


def test_modulo_by_zero():
    with pytest.raises(ExprError, match="modulo by zero"):
        ev("5%0")


def test_log_of_zero_or_negative():
    with pytest.raises(ExprError, match="log"):
        ev("log(0)")
    with pytest.raises(ExprError, match="log"):
        ev("log(-1)")


def test_sqrt_of_negative():
    with pytest.raises(ExprError, match="sqrt"):
        ev("sqrt(-1)")


def test_empty_string_error():
    with pytest.raises(ExprError, match="empty"):
        Expression("", [])
    with pytest.raises(ExprError, match="empty"):
        Expression("   ", [])


def test_trailing_garbage_error():
    with pytest.raises(ExprError):
        Expression("1 1", [])
    with pytest.raises(ExprError):
        Expression("1 + 2 3", [])


def test_unbalanced_parens_error():
    with pytest.raises(ExprError):
        Expression("(1+2", [])
    with pytest.raises(ExprError):
        Expression("1+2)", [])


def test_boolean_input_error_message():
    with pytest.raises(ExprError, match="boolean"):
        Expression(True, [])


# --------------------------------------------------------------------------- is_valid_identifier

@pytest.mark.parametrize("name", ["t", "T", "min", "max", "abs", "and", "or", "not", "1x", ""])
def test_is_valid_identifier_rejects(name):
    assert is_valid_identifier(name) is False


@pytest.mark.parametrize("name", ["x_1", "x", "energy", "_private", "x1"])
def test_is_valid_identifier_accepts(name):
    assert is_valid_identifier(name) is True
