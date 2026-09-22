"""Dynamic state variable model engine (SPEC.md §1, §2, §4, §5).

Backward induction over an N-dimensional integer state grid, individual-based
forward Monte Carlo, and an exact cohort projection used to verify the Monte
Carlo. Mirrors ``docs/engine.js``; both must give identical F and D.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .expr import Expression, ExprError, is_valid_identifier

FORMAT = "dsvm-model"
FORMAT_VERSION = 1
PROB_TOL = 1e-9
MAX_TABLE_CELLS = 50_000_000  # guard against runaway memory on huge grids

CAUSE_ALIVE = 0
CAUSE_MORTALITY = 1   # died of the action's mortality risk (predation)
CAUSE_STARVATION = 2  # a lethal state variable reached its threshold
CAUSE_NAMES = {CAUSE_ALIVE: "alive", CAUSE_MORTALITY: "mortality", CAUSE_STARVATION: "starvation"}


class ModelError(ValueError):
    """A structural or numerical problem with the model. Always says where."""


def _expr(value, variables, what: str) -> Expression:
    try:
        return Expression(value, variables)
    except ExprError as e:
        raise ModelError(f"{what}: {e}") from e


def _round_half_away(a: float) -> int:
    return int(math.floor(a + 0.5)) if a >= 0 else -int(math.floor(-a + 0.5))


@dataclass
class StateVariable:
    name: str
    min: int
    max: int
    lethal_min: bool = True
    initial: Optional[int] = None
    label: str = ""

    def __post_init__(self):
        if not is_valid_identifier(self.name):
            raise ModelError(f"state variable name {self.name!r} is not a valid identifier "
                             "(letters, digits, underscore; not starting with a digit; not t, T, or a function name)")
        for f in ("min", "max"):
            v = getattr(self, f)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or v != int(v):
                raise ModelError(f"state variable {self.name!r}: {f} must be an integer, got {v!r}")
            setattr(self, f, int(v))
        if self.max <= self.min:
            raise ModelError(f"state variable {self.name!r}: max ({self.max}) must exceed min ({self.min})")
        if self.initial is None:
            self.initial = self.max
        if self.initial != int(self.initial):
            raise ModelError(f"state variable {self.name!r}: initial must be an integer")
        self.initial = int(self.initial)
        if not (self.min <= self.initial <= self.max):
            raise ModelError(f"state variable {self.name!r}: initial ({self.initial}) is outside [{self.min}, {self.max}]")
        if self.lethal_min and self.initial <= self.min:
            raise ModelError(f"state variable {self.name!r}: initial ({self.initial}) is at or below the lethal threshold ({self.min})")
        if not self.label:
            self.label = self.name

    @property
    def values(self) -> range:
        return range(self.min, self.max + 1)

    def to_dict(self) -> dict:
        return {"name": self.name, "label": self.label, "min": self.min, "max": self.max,
                "lethal_min": self.lethal_min, "initial": self.initial}


@dataclass
class Outcome:
    prob: Expression
    change: Dict[str, Expression]
    label: str = ""

    def to_dict(self) -> dict:
        return {"label": self.label, "prob": self.prob.source,
                "change": {k: v.source for k, v in self.change.items()}}


@dataclass
class Action:
    name: str
    mortality: Expression
    outcomes: List[Outcome]
    reward: Expression = None  # type: ignore[assignment]

    def to_dict(self) -> dict:
        return {"name": self.name, "mortality": self.mortality.source,
                "reward": self.reward.source, "outcomes": [o.to_dict() for o in self.outcomes]}


@dataclass
class Solution:
    F: np.ndarray            # shape (S, T+1); column 0 unused (NaN)
    D: np.ndarray            # shape (S, T+1) int; -1 where undefined
    rounding_warnings: List[str] = field(default_factory=list)


@dataclass
class Simulation:
    n: int
    seed: int
    alive: np.ndarray            # (n, T+1) bool, column 0 unused; alive[:, t] = alive entering period t
    states: np.ndarray           # (n, T+1, N) int; -1 where not alive/defined
    actions: np.ndarray          # (n, T+1) int; action taken at t, -1 if none
    cause: np.ndarray            # (n,) int, CAUSE_*
    death_time: np.ndarray       # (n,) int; t at which death occurred, -1 if survived
    prop_alive: np.ndarray       # (T+1,) proportion alive entering period t
    cum_mortality: np.ndarray    # (T+1,) cumulative proportion dead of mortality risk by entry to t
    cum_starvation: np.ndarray   # (T+1,)
    mean_state: np.ndarray       # (T+1, N) mean state among the living (NaN if none)
    action_freq: np.ndarray      # (T+1, A) proportion of decision-makers at t choosing each action
    realised_fitness: float      # mean over individuals of accrued reward + terminal fitness


@dataclass
class Projection:
    prop_alive: np.ndarray       # (T+1,) exact proportion alive entering period t
    cum_mortality: np.ndarray
    cum_starvation: np.ndarray
    state_dist: np.ndarray       # (T+1, S) probability mass on each state
    expected_fitness: float      # exact expected accrued reward + terminal fitness


class Model:
    """A fully specified DSVM. Build with ``Model.from_dict`` or ``load``."""

    def __init__(self, state_variables: Sequence[StateVariable], actions: Sequence[Action],
                 terminal_fitness: Expression, horizon: int, name: str = "", description: str = "",
                 simulation: Optional[dict] = None, extra: Optional[dict] = None):
        if len(state_variables) == 0:
            raise ModelError("at least one state variable is required")
        names = [v.name for v in state_variables]
        if len(set(names)) != len(names):
            raise ModelError(f"state variable names must be unique: {names}")
        if len(actions) == 0:
            raise ModelError("at least one action is required")
        if isinstance(horizon, bool) or horizon != int(horizon) or int(horizon) < 2:
            raise ModelError(f"horizon T must be an integer ≥ 2, got {horizon!r}")
        for a in actions:
            if len(a.outcomes) == 0:
                raise ModelError(f"action {a.name!r} has no outcomes")
        self.state_variables = list(state_variables)
        self.var_names = names
        self.actions = list(actions)
        self.terminal_fitness = terminal_fitness
        self.T = int(horizon)
        self.name = name
        self.description = description
        self.simulation = dict(simulation or {})
        self.extra = dict(extra or {})

        self.states: List[Tuple[int, ...]] = list(itertools.product(*[v.values for v in self.state_variables]))
        self.state_index = {s: i for i, s in enumerate(self.states)}
        self.S = len(self.states)
        self.dead = np.array([self.is_dead(s) for s in self.states], dtype=bool)
        self._table = None  # lazily built transition table

    # ----------------------------------------------------------------- basics
    def is_dead(self, state: Sequence[int]) -> bool:
        return any(v.lethal_min and x <= v.min for x, v in zip(state, self.state_variables))

    def chop(self, state: Sequence[int], change: Sequence[int]) -> Tuple[int, ...]:
        return tuple(min(max(x + d, v.min), v.max) for x, d, v in zip(state, change, self.state_variables))

    def env(self, state: Sequence[int], t: int) -> dict:
        e = dict(zip(self.var_names, (float(x) for x in state)))
        e["t"] = float(t)
        e["T"] = float(self.T)
        return e

    @property
    def initial_state(self) -> Tuple[int, ...]:
        return tuple(v.initial for v in self.state_variables)

    # ------------------------------------------------------------ construction
    @classmethod
    def from_dict(cls, d: dict) -> "Model":
        if not isinstance(d, dict):
            raise ModelError("model must be a JSON object")
        if d.get("format", FORMAT) != FORMAT:
            raise ModelError(f"unknown format {d.get('format')!r}; expected {FORMAT!r}")
        if int(d.get("version", FORMAT_VERSION)) > FORMAT_VERSION:
            raise ModelError(f"model version {d.get('version')} is newer than this engine supports ({FORMAT_VERSION})")
        svs_raw = d.get("state_variables")
        if not isinstance(svs_raw, list) or not svs_raw:
            raise ModelError("'state_variables' must be a non-empty list")
        svs = []
        for i, sv in enumerate(svs_raw):
            if not isinstance(sv, dict) or "name" not in sv:
                raise ModelError(f"state variable #{i + 1} must be an object with a 'name'")
            for req in ("min", "max"):
                if req not in sv:
                    raise ModelError(f"state variable {sv['name']!r} is missing {req!r}")
            svs.append(StateVariable(name=str(sv["name"]), min=sv["min"], max=sv["max"],
                                     lethal_min=bool(sv.get("lethal_min", True)),
                                     initial=sv.get("initial"), label=str(sv.get("label", ""))))
        names = [v.name for v in svs]
        acts_raw = d.get("actions")
        if not isinstance(acts_raw, list) or not acts_raw:
            raise ModelError("'actions' must be a non-empty list")
        acts = []
        for i, a in enumerate(acts_raw):
            if not isinstance(a, dict):
                raise ModelError(f"action #{i + 1} must be an object")
            aname = str(a.get("name", f"Action {i + 1}"))
            if "mortality" not in a:
                raise ModelError(f"action {aname!r} is missing 'mortality'")
            outs_raw = a.get("outcomes")
            if not isinstance(outs_raw, list) or not outs_raw:
                raise ModelError(f"action {aname!r}: 'outcomes' must be a non-empty list")
            outs = []
            for j, o in enumerate(outs_raw):
                if not isinstance(o, dict) or "prob" not in o:
                    raise ModelError(f"action {aname!r} outcome #{j + 1} must be an object with 'prob'")
                change_raw = o.get("change", {}) or {}
                if not isinstance(change_raw, dict):
                    raise ModelError(f"action {aname!r} outcome #{j + 1}: 'change' must be an object")
                for k in change_raw:
                    if k not in names:
                        raise ModelError(f"action {aname!r} outcome #{j + 1}: 'change' refers to unknown state variable {k!r}")
                change = {k: _expr(v, names, f"action {aname!r} outcome #{j + 1} change of {k!r}")
                          for k, v in change_raw.items()}
                outs.append(Outcome(prob=_expr(o["prob"], names, f"action {aname!r} outcome #{j + 1} probability"),
                                    change=change, label=str(o.get("label", ""))))
            acts.append(Action(name=aname,
                               mortality=_expr(a["mortality"], names, f"action {aname!r} mortality"),
                               reward=_expr(a.get("reward", "0"), names, f"action {aname!r} reward"),
                               outcomes=outs))
        if "terminal_fitness" not in d:
            raise ModelError("'terminal_fitness' is required")
        if "horizon" not in d:
            raise ModelError("'horizon' (T) is required")
        known = {"format", "version", "name", "description", "horizon", "state_variables",
                 "actions", "terminal_fitness", "simulation"}
        extra = {k: v for k, v in d.items() if k not in known}
        return cls(svs, acts, _expr(d["terminal_fitness"], names, "terminal fitness"), d["horizon"],
                   name=str(d.get("name", "")), description=str(d.get("description", "")),
                   simulation=d.get("simulation") or {}, extra=extra)

    def to_dict(self) -> dict:
        out = {"format": FORMAT, "version": FORMAT_VERSION, "name": self.name,
               "description": self.description, "horizon": self.T,
               "state_variables": [v.to_dict() for v in self.state_variables],
               "actions": [a.to_dict() for a in self.actions],
               "terminal_fitness": self.terminal_fitness.source,
               "simulation": dict(self.simulation)}
        out.update(self.extra)
        return out

    # ------------------------------------------------------- transition table
    def _build_table(self):
        """Evaluate every expression once per (state, t, action, outcome).

        Expressions that do not mention ``t`` are evaluated at one ``t`` and
        broadcast. Raises ModelError with location on any invalid value.
        """
        S, T, A = self.S, self.T, len(self.actions)
        J = max(len(a.outcomes) for a in self.actions)
        N = len(self.state_variables)
        if S * (T + 1) * A * J > MAX_TABLE_CELLS:
            raise ModelError(f"model too large: {S} states × {T} periods × {A} actions × {J} outcomes; "
                             "reduce the state ranges or the horizon")
        beta = np.zeros((S, T + 1, A))
        reward = np.zeros((S, T + 1, A))
        prob = np.zeros((S, T + 1, A, J))
        nxt = np.full((S, T + 1, A, J), -1, dtype=np.int64)
        rounding: List[str] = []
        rounding_seen = set()
        decision_ts = range(1, T)  # decisions at t = 1 … T-1

        def uses_t(ex: Expression) -> bool:
            return "t" in ex.depends_on

        for a_idx, act in enumerate(self.actions):
            any_t = uses_t(act.mortality) or uses_t(act.reward) or any(
                uses_t(o.prob) or any(uses_t(c) for c in o.change.values()) for o in act.outcomes)
            ts = list(decision_ts) if any_t else [1]
            for s_idx, state in enumerate(self.states):
                if self.dead[s_idx]:
                    continue
                for t in ts:
                    env = self.env(state, t)
                    where = f"action {act.name!r} at state {dict(zip(self.var_names, state))}, t={t}"
                    try:
                        b = act.mortality(env)
                        if not (0.0 <= b <= 1.0):
                            raise ModelError(f"mortality {act.mortality.source!r} = {b} is outside [0, 1] for {where}")
                        r = act.reward(env)
                        psum = 0.0
                        for j, o in enumerate(act.outcomes):
                            p = o.prob(env)
                            if not (0.0 <= p <= 1.0):
                                raise ModelError(f"outcome probability {o.prob.source!r} = {p} is outside [0, 1] for {where}")
                            psum += p
                            change = []
                            for v in self.state_variables:
                                ex = o.change.get(v.name)
                                if ex is None:
                                    change.append(0)
                                    continue
                                raw = ex(env)
                                rd = _round_half_away(raw)
                                if rd != raw and ex.source not in rounding_seen:
                                    rounding_seen.add(ex.source)
                                    rounding.append(f"change {ex.source!r} of {v.name!r} gave {raw} at {where}; "
                                                    f"rounded to {rd}")
                                change.append(rd)
                            ns = self.chop(state, change)
                            prob[s_idx, t, a_idx, j] = p
                            nxt[s_idx, t, a_idx, j] = self.state_index[ns]
                        if abs(psum - 1.0) > PROB_TOL:
                            raise ModelError(f"outcome probabilities sum to {psum!r}, not 1, for {where}")
                        beta[s_idx, t, a_idx] = b
                        reward[s_idx, t, a_idx] = r
                    except ExprError as e:
                        raise ModelError(f"{e} ({where})") from e
                if not any_t:
                    for t in decision_ts:
                        beta[s_idx, t, a_idx] = beta[s_idx, 1, a_idx]
                        reward[s_idx, t, a_idx] = reward[s_idx, 1, a_idx]
                        prob[s_idx, t, a_idx, :] = prob[s_idx, 1, a_idx, :]
                        nxt[s_idx, t, a_idx, :] = nxt[s_idx, 1, a_idx, :]
        # terminal fitness
        phi = np.zeros(S)
        for s_idx, state in enumerate(self.states):
            if self.dead[s_idx]:
                continue
            try:
                phi[s_idx] = self.terminal_fitness(self.env(state, T))
            except ExprError as e:
                raise ModelError(f"terminal fitness {self.terminal_fitness.source!r}: {e} at state "
                                 f"{dict(zip(self.var_names, state))}") from e
        self._table = dict(beta=beta, reward=reward, prob=prob, nxt=nxt, phi=phi, rounding=rounding, J=J)
        return self._table

    @property
    def table(self):
        return self._table if self._table is not None else self._build_table()

    # ------------------------------------------------------------------ solve
    def solve(self) -> Solution:
        """Backward induction (SPEC.md §2)."""
        tb = self.table
        S, T, A = self.S, self.T, len(self.actions)
        F = np.full((S, T + 1), np.nan)
        D = np.full((S, T + 1), -1, dtype=np.int64)
        F[:, T] = np.where(self.dead, 0.0, tb["phi"])
        beta, reward, prob, nxt = tb["beta"], tb["reward"], tb["prob"], tb["nxt"]
        for t in range(T - 1, 0, -1):
            Fnext = F[:, t + 1]
            for s_idx in range(S):
                if self.dead[s_idx]:
                    F[s_idx, t] = 0.0
                    continue
                best = -math.inf
                best_a = 0
                for a in range(A):
                    expected = 0.0
                    for j in range(len(self.actions[a].outcomes)):
                        expected += prob[s_idx, t, a, j] * Fnext[nxt[s_idx, t, a, j]]
                    v = reward[s_idx, t, a] + (1.0 - beta[s_idx, t, a]) * expected
                    if v > best:  # strict: first listed action wins ties
                        best = v
                        best_a = a
                F[s_idx, t] = best
                D[s_idx, t] = best_a
        return Solution(F=F, D=D, rounding_warnings=list(tb["rounding"]))

    # --------------------------------------------------------------- simulate
    def simulate(self, solution: Solution, n: Optional[int] = None, seed: Optional[int] = None,
                 initial: Optional[Sequence[int]] = None) -> Simulation:
        """Individual-based forward Monte Carlo (SPEC.md §4)."""
        if n is None:
            n = int(self.simulation.get("individuals", 1000))
        if seed is None:
            seed = int(self.simulation.get("seed", 42))
        if n < 1:
            raise ModelError("number of individuals must be ≥ 1")
        init = tuple(int(x) for x in (initial if initial is not None else self.initial_state))
        if init not in self.state_index:
            raise ModelError(f"initial state {init} is not on the state grid")
        if self.is_dead(init):
            raise ModelError(f"initial state {init} is dead")
        tb = self.table
        beta, reward, prob, nxt, phi = tb["beta"], tb["reward"], tb["prob"], tb["nxt"], tb["phi"]
        S, T, A, N = self.S, self.T, len(self.actions), len(self.state_variables)
        D = solution.D
        rng = np.random.default_rng(seed)
        states_arr = np.array(self.states, dtype=np.int64)

        cur = np.full(n, self.state_index[init], dtype=np.int64)
        alive = np.zeros((n, T + 1), dtype=bool)
        alive[:, 1] = True
        st_hist = np.full((n, T + 1, N), -1, dtype=np.int64)
        st_hist[:, 1, :] = states_arr[cur]
        act_hist = np.full((n, T + 1), -1, dtype=np.int64)
        cause = np.zeros(n, dtype=np.int64)
        death_time = np.full(n, -1, dtype=np.int64)
        accrued = np.zeros(n)

        for t in range(1, T):
            u = rng.random((n, 2))  # u[i,0] mortality draw, u[i,1] outcome draw, individual order
            for i in range(n):
                if not alive[i, t]:
                    continue
                s = cur[i]
                a = D[s, t]
                act_hist[i, t] = a
                accrued[i] += reward[s, t, a]
                if u[i, 0] < beta[s, t, a]:
                    cause[i] = CAUSE_MORTALITY
                    death_time[i] = t
                    continue
                # inverse-CDF over outcomes in listed order
                c = 0.0
                j_pick = len(self.actions[a].outcomes) - 1
                for j in range(len(self.actions[a].outcomes)):
                    c += prob[s, t, a, j]
                    if u[i, 1] < c:
                        j_pick = j
                        break
                ns = nxt[s, t, a, j_pick]
                if self.dead[ns]:
                    cause[i] = CAUSE_STARVATION
                    death_time[i] = t
                    continue
                cur[i] = ns
                alive[i, t + 1] = True
                st_hist[i, t + 1, :] = states_arr[ns]

        terminal = np.where(alive[:, T], phi[cur], 0.0)
        realised = float(np.mean(accrued + terminal))

        prop_alive = alive.mean(axis=0)
        prop_alive[0] = np.nan
        cum_m = np.full(T + 1, np.nan)
        cum_s = np.full(T + 1, np.nan)
        for t in range(1, T + 1):
            died_before = (death_time >= 0) & (death_time < t)
            cum_m[t] = np.mean(died_before & (cause == CAUSE_MORTALITY))
            cum_s[t] = np.mean(died_before & (cause == CAUSE_STARVATION))
        mean_state = np.full((T + 1, N), np.nan)
        for t in range(1, T + 1):
            if alive[:, t].any():
                mean_state[t] = st_hist[alive[:, t], t, :].mean(axis=0)
        action_freq = np.full((T + 1, A), np.nan)
        for t in range(1, T):
            deciders = act_hist[:, t] >= 0
            if deciders.any():
                action_freq[t] = np.bincount(act_hist[deciders, t], minlength=A) / deciders.sum()

        return Simulation(n=n, seed=seed, alive=alive, states=st_hist, actions=act_hist, cause=cause,
                          death_time=death_time, prop_alive=prop_alive, cum_mortality=cum_m,
                          cum_starvation=cum_s, mean_state=mean_state, action_freq=action_freq,
                          realised_fitness=realised)

    # ---------------------------------------------------------------- project
    def project(self, solution: Solution, initial: Optional[Sequence[int]] = None) -> Projection:
        """Exact cohort projection under the optimal policy (SPEC.md §5)."""
        init = tuple(int(x) for x in (initial if initial is not None else self.initial_state))
        if init not in self.state_index or self.is_dead(init):
            raise ModelError(f"initial state {init} is not a live state on the grid")
        tb = self.table
        beta, reward, prob, nxt, phi = tb["beta"], tb["reward"], tb["prob"], tb["nxt"], tb["phi"]
        S, T = self.S, self.T
        D = solution.D
        dist = np.zeros((T + 1, S))
        dist[1, self.state_index[init]] = 1.0
        cum_m = np.full(T + 1, np.nan)
        cum_s = np.full(T + 1, np.nan)
        cum_m[1] = 0.0
        cum_s[1] = 0.0
        expected = 0.0
        for t in range(1, T):
            m_add = 0.0
            s_add = 0.0
            for s in range(S):
                mass = dist[t, s]
                if mass == 0.0 or self.dead[s]:
                    continue
                a = D[s, t]
                expected += mass * reward[s, t, a]
                b = beta[s, t, a]
                m_add += mass * b
                surv = mass * (1.0 - b)
                for j in range(len(self.actions[a].outcomes)):
                    p = prob[s, t, a, j]
                    if p == 0.0:
                        continue
                    ns = nxt[s, t, a, j]
                    if self.dead[ns]:
                        s_add += surv * p
                    else:
                        dist[t + 1, ns] += surv * p
            cum_m[t + 1] = cum_m[t] + m_add
            cum_s[t + 1] = cum_s[t] + s_add
        expected += float(np.dot(dist[T], np.where(self.dead, 0.0, phi)))
        prop_alive = dist.sum(axis=1)
        prop_alive[0] = np.nan
        return Projection(prop_alive=prop_alive, cum_mortality=cum_m, cum_starvation=cum_s,
                          state_dist=dist, expected_fitness=expected)
