# DSVM tool: model specification and semantics

This document is the contract that the JavaScript engine (`docs/engine.js`,
used by the web app) and the Python engine (`python/dsvm/engine.py`) both
implement. A model saved by one runs unchanged in the other and must produce
the same fitness matrix `F` and decision matrix `D` to within floating-point
rounding (`|ΔF| < 1e-9`; `D` identical). Monte Carlo output agrees
statistically, not bit for bit, because the two languages use different
random number generators.

Notation follows Mangel & Clark (1988) *Dynamic Modeling in Behavioral
Ecology* and Clark & Mangel (2000) *Dynamic State Variable Models in Ecology*.

## 1. Model components

### 1.1 Time

Discrete periods `t = 1, 2, …, T`. `T` is the horizon (integer, `T ≥ 2`).
Decisions are made at `t = 1 … T-1`. No decision is made at `t = T`; the
terminal fitness function is evaluated there.

### 1.2 State variables

One or more named state variables `x_1 … x_N`. Each is an integer on a
closed range `[min, max]`. Fields:

| field | type | meaning |
|---|---|---|
| `name` | identifier | used in expressions; letters, digits, `_`; must not start with a digit; must not be `t`, `T`, or a function name (§3) |
| `min` | integer | lowest value on the grid. When `lethal_min` is true this is the critical threshold `x_crit` |
| `max` | integer | highest value on the grid (capacity `C`); `max > min` |
| `lethal_min` | boolean | if true, an individual whose value is **at or below** `min` is dead (Mangel & Clark's `x ≤ x_crit`). If false, `min` is only a floor (e.g. gut contents cannot go below 0) |
| `initial` | integer | starting value for Monte Carlo individuals; must lie in `[min, max]` and not be lethal |

The state space is the full Cartesian product of the variable ranges. A state
is **dead** if any variable with `lethal_min = true` is at or below its `min`.
Dead states have `F = 0` at every `t`, including `t = T`, whatever the terminal
fitness expression says.

### 1.3 Actions

An ordered list of one or more behavioural choices `i = 1 … M`. Fields:

| field | type | meaning |
|---|---|---|
| `name` | string | label |
| `mortality` | expression | probability `β_i(x, t)` that the individual dies this period **before** the outcome is realised (predation risk in Mangel & Clark). Must evaluate to a value in `[0, 1]` at every live state and every decision time |
| `reward` | expression, optional | per-period fitness increment `R_i(x, t)` accrued when the action is chosen, default `0`. Mike Heithaus asked for terminal fitness only "for now"; this field exists so per-period rewards (eggs laid, offspring fledged) can be added without changing the format. Leave it at `0` for a pure terminal-fitness model |
| `outcomes` | list | one or more stochastic outcomes, §1.4 |

### 1.4 Outcomes

Each outcome `j` of action `i` has:

| field | type | meaning |
|---|---|---|
| `label` | string, optional | e.g. "found food" |
| `prob` | expression | probability `λ_ij(x, t)` of this outcome given the individual survived the mortality draw. For each action, at every live state and decision time, the outcome probabilities must sum to 1 within `1e-9`; the engine reports an error otherwise (it never renormalises silently) |
| `change` | map name → expression | net change `Δ_ij(x, t)` applied to each state variable (gain minus cost). Variables not listed change by 0 |

### 1.5 Terminal fitness

One expression `φ(x)` evaluated at `t = T` for every live state. It may use
`T` and the state variable names. Mangel & Clark's simplest choice is survival:
`1` (every live state has fitness 1, dead states 0 by §1.2).

## 2. Dynamic programming equation

For every live state `x` and `t = T-1, T-2, …, 1`:

```
F(x, t) = max over i of  V_i(x, t)

V_i(x, t) = R_i(x, t) + (1 − β_i(x, t)) · Σ_j λ_ij(x, t) · F( x'_ij , t+1 )

x'_ij = chop( x + Δ_ij(x, t) ; min, max )        applied per variable
F(x, T) = φ(x)                                    for live x
F(x, t) = 0                                       for dead x, all t
D(x, t) = the i achieving the max                 for t = 1 … T-1
```

Rules:

- `chop(v; min, max)` = `min(max(v, min), max)`. Values are clamped, never
  wrapped or discarded.
- If an outcome expression produces a non-integer change, it is rounded to
  the nearest integer (`round half away from zero`) **before** chopping. The
  reference examples use integer changes, so no rounding occurs there. The
  web app and the CLI warn when rounding did occur for a model.
- If `x'_ij` is dead, `F(x'_ij, t+1) = 0` (follows from §1.2; stated here so
  no implementation special-cases it differently).
- Ties: when two or more actions have equal `V_i` (compared exactly, after
  computing in double precision), the action listed first wins. Both engines
  compute `V_i` with the same operation order: `expected = Σ_j (λ_ij · F)`
  accumulated in outcome order, then `V = R + (1 − β) · expected`.
- Expressions are evaluated at the **current** state `x` and time `t`, never
  at `x'`.
- `D(x, T)` is undefined (no decision at `T`); stored as `-1` in both
  engines and exported as an empty cell.
- `D` for dead states is likewise `-1`.

## 3. Expression language

Parameters may be plain numbers (`0.004`) or expressions in the state
variable names, `t`, and `T`. Both engines implement the same grammar with a
hand-written recursive-descent parser; neither uses `eval`.

```
expr     := ternary
ternary  := or ( '?' ternary ':' ternary )?
or       := and ( ('or' | '||') and )*
and      := not ( ('and' | '&&') not )*
not      := ('not' | '!') not | compare
compare  := sum ( ('<' | '<=' | '>' | '>=' | '==' | '!=') sum )?
sum      := product ( ('+' | '-') product )*
product  := unary ( ('*' | '/' | '%') unary )*
unary    := '-' unary | power
power    := atom ( '^' unary )?                 (right associative)
atom     := number | identifier | identifier '(' args ')' | '(' expr ')'
```

- Booleans are numbers: comparisons return `1` or `0`; `and`/`or`/`not` treat
  non-zero as true and return `1`/`0`; the ternary tests non-zero.
- Functions: `min(a, b, …)`, `max(a, b, …)`, `abs(a)`, `floor(a)`, `ceil(a)`,
  `round(a)` (half away from zero), `sqrt(a)`, `exp(a)`, `log(a)` (natural),
  `pow(a, b)`, `if(cond, a, b)`.
- Division by zero, `log` of a non-positive number, `sqrt` of a negative
  number, an unknown identifier, or a probability outside `[0, 1]` is an
  **error**, reported with the expression text and the state and time at
  which it occurred. Errors abort the solve; nothing is silently substituted.
- Numbers: decimal with optional fraction and exponent (`1`, `0.4`, `2e-3`).
- Identifiers are case-sensitive.

## 4. Forward Monte Carlo simulation

Individual-based, as requested. Inputs: number of individuals `n`, a random
seed, and the initial state (default: each variable's `initial`). For each
individual and each `t = 1 … T-1`, given it is alive at state `x`:

1. Look up `i = D(x, t)`.
2. Draw `u₁ ~ U(0,1)`. If `u₁ < β_i(x, t)`, the individual dies of
   **mortality risk** (predation) at `t`; stop.
3. Draw outcome `j` with probabilities `λ_ij(x, t)` (inverse-CDF on a single
   `u₂ ~ U(0,1)`, outcomes in listed order).
4. Set `x ← chop(x + Δ_ij(x, t))`. If `x` is dead, the individual dies of
   **starvation** (lethal threshold) at `t`; stop.
5. Record `x` as the state at `t+1`, and `i` as the action taken at `t`.

Outputs, per time step `t = 1 … T`: proportion alive, cumulative proportion
dead by each cause, mean of each state variable among the living, and the
frequency of each action among individuals that made a decision at `t`. Also
the full state history for spaghetti plots, and the terminal fitness realised:
mean of `φ(x_T)` over survivors × survivors / `n` (expected fitness of the
cohort), which converges to `F(x_initial, 1)` and is the internal check that
the simulation matches the solve.

Order of draws matters for reproducibility within one language: `u₁` then
`u₂`, one individual at a time, in individual order, then the next `t`.

## 5. Exact cohort projection (verification)

Alongside the Monte Carlo the engines compute the exact distribution of
states over time by propagating the probability mass under the optimal
policy (a Markov chain over the state grid plus two absorbing death states).
This is not a user-requested feature. It exists to check the Monte Carlo:
the web app overlays the exact survival curve on the simulated one, and the
Python tests assert the simulated survival lies within a binomial confidence
band of the exact value. It also gives `F(x_initial, 1)` back exactly when
`R ≡ 0` and `φ` is survival, which is the cross-check between §2 and §4.

## 6. Model file format

JSON, UTF-8. Extension `.dsvm.json`. Example: `examples/patch_selection.dsvm.json`.

```json
{
  "format": "dsvm-model",
  "version": 1,
  "name": "Patch selection under predation risk",
  "description": "free text, may cite the source",
  "horizon": 20,
  "state_variables": [
    {"name": "x", "label": "Energy reserves", "min": 3, "max": 10,
     "lethal_min": true, "initial": 8}
  ],
  "actions": [
    {"name": "Patch 1 (safe)", "mortality": "0", "reward": "0",
     "outcomes": [
       {"label": "no food", "prob": "1", "change": {"x": "-1"}}
     ]}
  ],
  "terminal_fitness": "1",
  "simulation": {"individuals": 1000, "seed": 42}
}
```

All `prob`, `mortality`, `reward`, `change` values, and `terminal_fitness`
are strings holding expressions (§3). Numbers are accepted where a string is
expected and treated as their decimal text. `label`, `description`,
`reward`, and `simulation` are optional. Unknown keys are preserved on
round-trip and ignored by the engines.

## 7. Exported matrices

CSV, one row per state, columns `<var names…>, t=1, t=2, …, t=T`.

- `F` values are written with 10 significant digits.
- `D` cells hold the action **name**; dead states and `t = T` are empty.
- The simulation summary CSV has one row per `t` with the per-step outputs
  of §4.

## 8. Out of scope (deliberately)

Continuous states with interpolation, per-period discounting, stochastic
environments with hidden states, multiple decision makers, and per-action
time costs are not implemented. Each is a documented extension point rather
than a silent assumption: the format has a `version` field so they can be
added later without breaking saved models.
