// Dynamic state variable model engine (SPEC.md §1, §2, §4, §5).
// Mirrors python/dsvm/engine.py. Both must give identical F and D for the
// same model file; python/tests/test_crosscheck_js.py enforces this.

import { Expression, ExprError, isValidIdentifier } from "./expr.js";

export const FORMAT = "dsvm-model";
export const FORMAT_VERSION = 1;
export const PROB_TOL = 1e-9;
export const MAX_TABLE_CELLS = 50_000_000;

export const CAUSE_ALIVE = 0;
export const CAUSE_MORTALITY = 1;   // died of the action's mortality risk (predation)
export const CAUSE_STARVATION = 2;  // a lethal state variable reached its threshold
export const CAUSE_NAMES = { 0: "alive", 1: "mortality", 2: "starvation" };

export class ModelError extends Error {
  constructor(msg) { super(msg); this.name = "ModelError"; }
}

function roundHalfAway(a) {
  return a >= 0 ? Math.floor(a + 0.5) : -Math.floor(-a + 0.5);
}

function isInt(v) {
  return typeof v === "number" && Number.isFinite(v) && v === Math.trunc(v);
}

function expr(value, variables, what) {
  try {
    return new Expression(value, variables);
  } catch (e) {
    if (e instanceof ExprError) throw new ModelError(`${what}: ${e.message}`);
    throw e;
  }
}

function fmtState(names, state) {
  return "{" + names.map((n, k) => `${n}: ${state[k]}`).join(", ") + "}";
}

// ---------------------------------------------------------------- RNG
// xoshiro128** seeded from a 32-bit integer via splitmix32. Deterministic
// for a given seed across browsers and node.
export class RNG {
  constructor(seed) {
    let s = (Number(seed) >>> 0) || 0;
    const next = () => {
      s = (s + 0x9e3779b9) >>> 0;
      let z = s;
      z = Math.imul(z ^ (z >>> 16), 0x85ebca6b) >>> 0;
      z = Math.imul(z ^ (z >>> 13), 0xc2b2ae35) >>> 0;
      return (z ^ (z >>> 16)) >>> 0;
    };
    this.a = next(); this.b = next(); this.c = next(); this.d = next();
    if ((this.a | this.b | this.c | this.d) === 0) this.a = 1;
  }
  /** uniform on [0, 1) with 32 bits of resolution */
  random() {
    const t = this.b << 9;
    let r = this.b * 5;
    r = ((r << 7) | (r >>> 25)) * 9;
    this.c ^= this.a; this.d ^= this.b; this.b ^= this.c; this.a ^= this.d;
    this.c ^= t;
    this.d = (this.d << 11) | (this.d >>> 21);
    return (r >>> 0) / 4294967296;
  }
}

// ------------------------------------------------------- state variables
export class StateVariable {
  constructor({ name, min, max, lethal_min = true, initial = null, label = "" }) {
    if (typeof name !== "string" || !isValidIdentifier(name)) {
      throw new ModelError(`state variable name '${name}' is not a valid identifier (letters, digits, underscore; not starting with a digit; not t, T, or a function name)`);
    }
    if (!isInt(min)) throw new ModelError(`state variable '${name}': min must be an integer, got ${JSON.stringify(min)}`);
    if (!isInt(max)) throw new ModelError(`state variable '${name}': max must be an integer, got ${JSON.stringify(max)}`);
    if (max <= min) throw new ModelError(`state variable '${name}': max (${max}) must exceed min (${min})`);
    this.name = name;
    this.min = min;
    this.max = max;
    this.lethalMin = Boolean(lethal_min);
    if (initial === null || initial === undefined) initial = max;
    if (!isInt(initial)) throw new ModelError(`state variable '${name}': initial must be an integer`);
    if (initial < min || initial > max) throw new ModelError(`state variable '${name}': initial (${initial}) is outside [${min}, ${max}]`);
    if (this.lethalMin && initial <= min) throw new ModelError(`state variable '${name}': initial (${initial}) is at or below the lethal threshold (${min})`);
    this.initial = initial;
    this.label = label || name;
  }
  get size() { return this.max - this.min + 1; }
  toObject() {
    return { name: this.name, label: this.label, min: this.min, max: this.max, lethal_min: this.lethalMin, initial: this.initial };
  }
}

// ------------------------------------------------------------------ model
export class Model {
  constructor(stateVariables, actions, terminalFitness, horizon, { name = "", description = "", simulation = {}, extra = {} } = {}) {
    if (stateVariables.length === 0) throw new ModelError("at least one state variable is required");
    const names = stateVariables.map((v) => v.name);
    if (new Set(names).size !== names.length) throw new ModelError(`state variable names must be unique: ${names.join(", ")}`);
    if (actions.length === 0) throw new ModelError("at least one action is required");
    if (!isInt(horizon) || horizon < 2) throw new ModelError(`horizon T must be an integer ≥ 2, got ${JSON.stringify(horizon)}`);
    for (const a of actions) if (a.outcomes.length === 0) throw new ModelError(`action '${a.name}' has no outcomes`);
    this.stateVariables = stateVariables;
    this.varNames = names;
    this.actions = actions;
    this.terminalFitness = terminalFitness;
    this.T = horizon;
    this.name = name;
    this.description = description;
    this.simulation = { ...simulation };
    this.extra = { ...extra };

    // Full Cartesian product, first variable outermost (same order as Python itertools.product).
    const N = stateVariables.length;
    let S = 1;
    for (const v of stateVariables) S *= v.size;
    this.S = S;
    this.states = new Array(S);
    for (let s = 0; s < S; s++) {
      const st = new Array(N);
      let rem = s;
      for (let k = N - 1; k >= 0; k--) {
        const v = stateVariables[k];
        st[k] = v.min + (rem % v.size);
        rem = Math.floor(rem / v.size);
      }
      this.states[s] = st;
    }
    this.dead = new Uint8Array(S);
    for (let s = 0; s < S; s++) this.dead[s] = this.isDead(this.states[s]) ? 1 : 0;
    this._table = null;
  }

  static fromObject(d) {
    if (typeof d !== "object" || d === null || Array.isArray(d)) throw new ModelError("model must be a JSON object");
    if ((d.format ?? FORMAT) !== FORMAT) throw new ModelError(`unknown format '${d.format}'; expected '${FORMAT}'`);
    if (Number(d.version ?? FORMAT_VERSION) > FORMAT_VERSION) throw new ModelError(`model version ${d.version} is newer than this engine supports (${FORMAT_VERSION})`);
    if (!Array.isArray(d.state_variables) || d.state_variables.length === 0) throw new ModelError("'state_variables' must be a non-empty list");
    const svs = d.state_variables.map((sv, i) => {
      if (typeof sv !== "object" || sv === null || !("name" in sv)) throw new ModelError(`state variable #${i + 1} must be an object with a 'name'`);
      for (const req of ["min", "max"]) if (!(req in sv)) throw new ModelError(`state variable '${sv.name}' is missing '${req}'`);
      return new StateVariable({ name: String(sv.name), min: sv.min, max: sv.max, lethal_min: sv.lethal_min ?? true, initial: sv.initial ?? null, label: String(sv.label ?? "") });
    });
    const names = svs.map((v) => v.name);
    if (!Array.isArray(d.actions) || d.actions.length === 0) throw new ModelError("'actions' must be a non-empty list");
    const acts = d.actions.map((a, i) => {
      if (typeof a !== "object" || a === null) throw new ModelError(`action #${i + 1} must be an object`);
      const aname = String(a.name ?? `Action ${i + 1}`);
      if (!("mortality" in a)) throw new ModelError(`action '${aname}' is missing 'mortality'`);
      if (!Array.isArray(a.outcomes) || a.outcomes.length === 0) throw new ModelError(`action '${aname}': 'outcomes' must be a non-empty list`);
      const outs = a.outcomes.map((o, j) => {
        if (typeof o !== "object" || o === null || !("prob" in o)) throw new ModelError(`action '${aname}' outcome #${j + 1} must be an object with 'prob'`);
        const changeRaw = o.change ?? {};
        if (typeof changeRaw !== "object" || Array.isArray(changeRaw)) throw new ModelError(`action '${aname}' outcome #${j + 1}: 'change' must be an object`);
        const change = {};
        for (const k of Object.keys(changeRaw)) {
          if (!names.includes(k)) throw new ModelError(`action '${aname}' outcome #${j + 1}: 'change' refers to unknown state variable '${k}'`);
          change[k] = expr(changeRaw[k], names, `action '${aname}' outcome #${j + 1} change of '${k}'`);
        }
        return { label: String(o.label ?? ""), prob: expr(o.prob, names, `action '${aname}' outcome #${j + 1} probability`), change };
      });
      return {
        name: aname,
        mortality: expr(a.mortality, names, `action '${aname}' mortality`),
        reward: expr(a.reward ?? "0", names, `action '${aname}' reward`),
        outcomes: outs,
      };
    });
    if (!("terminal_fitness" in d)) throw new ModelError("'terminal_fitness' is required");
    if (!("horizon" in d)) throw new ModelError("'horizon' (T) is required");
    const known = new Set(["format", "version", "name", "description", "horizon", "state_variables", "actions", "terminal_fitness", "simulation"]);
    const extra = {};
    for (const k of Object.keys(d)) if (!known.has(k)) extra[k] = d[k];
    return new Model(svs, acts, expr(d.terminal_fitness, names, "terminal fitness"), d.horizon,
      { name: String(d.name ?? ""), description: String(d.description ?? ""), simulation: d.simulation ?? {}, extra });
  }

  toObject() {
    return {
      format: FORMAT, version: FORMAT_VERSION, name: this.name, description: this.description, horizon: this.T,
      state_variables: this.stateVariables.map((v) => v.toObject()),
      actions: this.actions.map((a) => ({
        name: a.name, mortality: a.mortality.source, reward: a.reward.source,
        outcomes: a.outcomes.map((o) => ({
          label: o.label, prob: o.prob.source,
          change: Object.fromEntries(Object.entries(o.change).map(([k, e]) => [k, e.source])),
        })),
      })),
      terminal_fitness: this.terminalFitness.source,
      simulation: { ...this.simulation },
      ...this.extra,
    };
  }

  // ------------------------------------------------------------ basics
  isDead(state) {
    for (let k = 0; k < this.stateVariables.length; k++) {
      const v = this.stateVariables[k];
      if (v.lethalMin && state[k] <= v.min) return true;
    }
    return false;
  }
  stateIndex(state) {
    let idx = 0;
    for (let k = 0; k < this.stateVariables.length; k++) {
      const v = this.stateVariables[k];
      const x = state[k];
      if (!isInt(x) || x < v.min || x > v.max) return -1;
      idx = idx * v.size + (x - v.min);
    }
    return idx;
  }
  chop(state, change) {
    return state.map((x, k) => {
      const v = this.stateVariables[k];
      return Math.min(Math.max(x + change[k], v.min), v.max);
    });
  }
  env(state, t) {
    const e = { t, T: this.T };
    this.varNames.forEach((n, k) => { e[n] = state[k]; });
    return e;
  }
  get initialState() { return this.stateVariables.map((v) => v.initial); }

  // --------------------------------------------------- transition table
  // Flat typed arrays indexed [s][t][a] and [s][t][a][j]:
  //   beta/reward: ((s * (T+1)) + t) * A + a
  //   prob/nxt:    (((s * (T+1)) + t) * A + a) * J + j
  get table() { return this._table ?? this._buildTable(); }

  _buildTable() {
    const S = this.S, T = this.T, A = this.actions.length;
    const J = Math.max(...this.actions.map((a) => a.outcomes.length));
    if (S * (T + 1) * A * J > MAX_TABLE_CELLS) {
      throw new ModelError(`model too large: ${S} states × ${T} periods × ${A} actions × ${J} outcomes; reduce the state ranges or the horizon`);
    }
    const beta = new Float64Array(S * (T + 1) * A);
    const reward = new Float64Array(S * (T + 1) * A);
    const prob = new Float64Array(S * (T + 1) * A * J);
    const nxt = new Int32Array(S * (T + 1) * A * J).fill(-1);
    const rounding = [];
    const roundingSeen = new Set();
    const usesT = (ex) => ex.dependsOn.has("t");

    for (let a = 0; a < A; a++) {
      const act = this.actions[a];
      const anyT = usesT(act.mortality) || usesT(act.reward) ||
        act.outcomes.some((o) => usesT(o.prob) || Object.values(o.change).some(usesT));
      const ts = anyT ? Array.from({ length: T - 1 }, (_, i) => i + 1) : [1];
      for (let s = 0; s < S; s++) {
        if (this.dead[s]) continue;
        const state = this.states[s];
        for (const t of ts) {
          const env = this.env(state, t);
          const where = `action '${act.name}' at state ${fmtState(this.varNames, state)}, t=${t}`;
          try {
            const b = act.mortality.evaluate(env);
            if (!(b >= 0 && b <= 1)) throw new ModelError(`mortality '${act.mortality.source}' = ${b} is outside [0, 1] for ${where}`);
            const r = act.reward.evaluate(env);
            let psum = 0;
            const base = ((s * (T + 1)) + t) * A + a;
            for (let j = 0; j < act.outcomes.length; j++) {
              const o = act.outcomes[j];
              const p = o.prob.evaluate(env);
              if (!(p >= 0 && p <= 1)) throw new ModelError(`outcome probability '${o.prob.source}' = ${p} is outside [0, 1] for ${where}`);
              psum += p;
              const change = this.stateVariables.map((v) => {
                const ex = o.change[v.name];
                if (!ex) return 0;
                const raw = ex.evaluate(env);
                const rd = roundHalfAway(raw);
                if (rd !== raw && !roundingSeen.has(ex.source)) {
                  roundingSeen.add(ex.source);
                  rounding.push(`change '${ex.source}' of '${v.name}' gave ${raw} at ${where}; rounded to ${rd}`);
                }
                return rd;
              });
              const ns = this.stateIndex(this.chop(state, change));
              prob[base * J + j] = p;
              nxt[base * J + j] = ns;
            }
            if (Math.abs(psum - 1) > PROB_TOL) throw new ModelError(`outcome probabilities sum to ${psum}, not 1, for ${where}`);
            beta[base] = b;
            reward[base] = r;
          } catch (e) {
            if (e instanceof ExprError) throw new ModelError(`${e.message} (${where})`);
            throw e;
          }
        }
        if (!anyT) {
          const src = ((s * (T + 1)) + 1) * A + a;
          for (let t = 2; t < T; t++) {
            const dst = ((s * (T + 1)) + t) * A + a;
            beta[dst] = beta[src];
            reward[dst] = reward[src];
            for (let j = 0; j < J; j++) { prob[dst * J + j] = prob[src * J + j]; nxt[dst * J + j] = nxt[src * J + j]; }
          }
        }
      }
    }
    const phi = new Float64Array(S);
    for (let s = 0; s < S; s++) {
      if (this.dead[s]) continue;
      try {
        phi[s] = this.terminalFitness.evaluate(this.env(this.states[s], T));
      } catch (e) {
        if (e instanceof ExprError) throw new ModelError(`terminal fitness '${this.terminalFitness.source}': ${e.message} at state ${fmtState(this.varNames, this.states[s])}`);
        throw e;
      }
    }
    this._table = { beta, reward, prob, nxt, phi, rounding, J, A, T, S };
    return this._table;
  }

  // ------------------------------------------------------------------ solve
  /** Backward induction (SPEC §2). Returns {F, D, roundingWarnings}; F[s][t] Float64Array rows, D[s][t] Int32Array rows (-1 undefined). */
  solve() {
    const tb = this.table;
    const { S, T, A, J } = tb;
    const F = Array.from({ length: S }, () => new Float64Array(T + 1).fill(NaN));
    const D = Array.from({ length: S }, () => new Int32Array(T + 1).fill(-1));
    for (let s = 0; s < S; s++) F[s][T] = this.dead[s] ? 0 : tb.phi[s];
    for (let t = T - 1; t >= 1; t--) {
      for (let s = 0; s < S; s++) {
        if (this.dead[s]) { F[s][t] = 0; continue; }
        let best = -Infinity, bestA = 0;
        for (let a = 0; a < A; a++) {
          const base = ((s * (T + 1)) + t) * A + a;
          let expected = 0;
          const nOut = this.actions[a].outcomes.length;
          for (let j = 0; j < nOut; j++) {
            expected += tb.prob[base * J + j] * F[tb.nxt[base * J + j]][t + 1];
          }
          const v = tb.reward[base] + (1 - tb.beta[base]) * expected;
          if (v > best) { best = v; bestA = a; }  // strict: first listed action wins ties
        }
        F[s][t] = best;
        D[s][t] = bestA;
      }
    }
    return { F, D, roundingWarnings: [...tb.rounding] };
  }

  // --------------------------------------------------------------- simulate
  /** Individual-based forward Monte Carlo (SPEC §4). */
  simulate(solution, { n = null, seed = null, initial = null } = {}) {
    if (n === null) n = Number(this.simulation.individuals ?? 1000);
    if (seed === null) seed = Number(this.simulation.seed ?? 42);
    if (!isInt(n) || n < 1) throw new ModelError("number of individuals must be an integer ≥ 1");
    const init = (initial ?? this.initialState).map(Number);
    const initIdx = this.stateIndex(init);
    if (initIdx < 0) throw new ModelError(`initial state ${fmtState(this.varNames, init)} is not on the state grid`);
    if (this.dead[initIdx]) throw new ModelError(`initial state ${fmtState(this.varNames, init)} is dead`);
    const tb = this.table;
    const { S, T, A, J } = tb;
    const N = this.stateVariables.length;
    const D = solution.D;
    const rng = new RNG(seed);

    const cur = new Int32Array(n).fill(initIdx);
    const alive = new Uint8Array(n * (T + 1));
    const states = new Int32Array(n * (T + 1) * N).fill(-1);
    const actions = new Int32Array(n * (T + 1)).fill(-1);
    const cause = new Int32Array(n);
    const deathTime = new Int32Array(n).fill(-1);
    const accrued = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      alive[i * (T + 1) + 1] = 1;
      for (let k = 0; k < N; k++) states[(i * (T + 1) + 1) * N + k] = init[k];
    }
    for (let t = 1; t < T; t++) {
      for (let i = 0; i < n; i++) {
        if (!alive[i * (T + 1) + t]) continue;
        const s = cur[i];
        const a = D[s][t];
        const base = ((s * (T + 1)) + t) * A + a;
        actions[i * (T + 1) + t] = a;
        accrued[i] += tb.reward[base];
        const u1 = rng.random();
        const u2 = rng.random();
        if (u1 < tb.beta[base]) { cause[i] = CAUSE_MORTALITY; deathTime[i] = t; continue; }
        const nOut = this.actions[a].outcomes.length;
        let c = 0, jPick = nOut - 1;
        for (let j = 0; j < nOut; j++) {
          c += tb.prob[base * J + j];
          if (u2 < c) { jPick = j; break; }
        }
        const ns = tb.nxt[base * J + jPick];
        if (this.dead[ns]) { cause[i] = CAUSE_STARVATION; deathTime[i] = t; continue; }
        cur[i] = ns;
        alive[i * (T + 1) + t + 1] = 1;
        const st = this.states[ns];
        for (let k = 0; k < N; k++) states[(i * (T + 1) + t + 1) * N + k] = st[k];
      }
    }
    let fitSum = 0;
    for (let i = 0; i < n; i++) fitSum += accrued[i] + (alive[i * (T + 1) + T] ? tb.phi[cur[i]] : 0);
    const realisedFitness = fitSum / n;

    const propAlive = new Float64Array(T + 1).fill(NaN);
    const cumMortality = new Float64Array(T + 1).fill(NaN);
    const cumStarvation = new Float64Array(T + 1).fill(NaN);
    const meanState = Array.from({ length: T + 1 }, () => new Float64Array(N).fill(NaN));
    const actionFreq = Array.from({ length: T + 1 }, () => new Float64Array(A).fill(NaN));
    for (let t = 1; t <= T; t++) {
      let na = 0, nm = 0, nsv = 0;
      const sums = new Float64Array(N);
      const counts = new Float64Array(A);
      let deciders = 0;
      for (let i = 0; i < n; i++) {
        if (alive[i * (T + 1) + t]) {
          na++;
          for (let k = 0; k < N; k++) sums[k] += states[(i * (T + 1) + t) * N + k];
        }
        if (deathTime[i] >= 0 && deathTime[i] < t) {
          if (cause[i] === CAUSE_MORTALITY) nm++; else nsv++;
        }
        const a = actions[i * (T + 1) + t];
        if (a >= 0) { deciders++; counts[a]++; }
      }
      propAlive[t] = na / n;
      cumMortality[t] = nm / n;
      cumStarvation[t] = nsv / n;
      if (na > 0) for (let k = 0; k < N; k++) meanState[t][k] = sums[k] / na;
      if (deciders > 0) for (let a = 0; a < A; a++) actionFreq[t][a] = counts[a] / deciders;
    }
    return { n, seed, T, N, alive, states, actions, cause, deathTime, propAlive, cumMortality, cumStarvation, meanState, actionFreq, realisedFitness,
      /** helpers */
      isAlive: (i, t) => alive[i * (T + 1) + t] === 1,
      stateOf: (i, t) => Array.from(states.subarray((i * (T + 1) + t) * N, (i * (T + 1) + t + 1) * N)),
      actionOf: (i, t) => actions[i * (T + 1) + t],
    };
  }

  // ---------------------------------------------------------------- project
  /** Exact cohort projection under the optimal policy (SPEC §5). */
  project(solution, { initial = null } = {}) {
    const init = (initial ?? this.initialState).map(Number);
    const initIdx = this.stateIndex(init);
    if (initIdx < 0 || this.dead[initIdx]) throw new ModelError(`initial state ${fmtState(this.varNames, init)} is not a live state on the grid`);
    const tb = this.table;
    const { S, T, A, J } = tb;
    const D = solution.D;
    const dist = Array.from({ length: T + 1 }, () => new Float64Array(S));
    dist[1][initIdx] = 1;
    const cumMortality = new Float64Array(T + 1).fill(NaN);
    const cumStarvation = new Float64Array(T + 1).fill(NaN);
    cumMortality[1] = 0; cumStarvation[1] = 0;
    let expected = 0;
    for (let t = 1; t < T; t++) {
      let mAdd = 0, sAdd = 0;
      for (let s = 0; s < S; s++) {
        const mass = dist[t][s];
        if (mass === 0 || this.dead[s]) continue;
        const a = D[s][t];
        const base = ((s * (T + 1)) + t) * A + a;
        expected += mass * tb.reward[base];
        const b = tb.beta[base];
        mAdd += mass * b;
        const surv = mass * (1 - b);
        const nOut = this.actions[a].outcomes.length;
        for (let j = 0; j < nOut; j++) {
          const p = tb.prob[base * J + j];
          if (p === 0) continue;
          const ns = tb.nxt[base * J + j];
          if (this.dead[ns]) sAdd += surv * p; else dist[t + 1][ns] += surv * p;
        }
      }
      cumMortality[t + 1] = cumMortality[t] + mAdd;
      cumStarvation[t + 1] = cumStarvation[t] + sAdd;
    }
    for (let s = 0; s < S; s++) if (!this.dead[s]) expected += dist[T][s] * tb.phi[s];
    const propAlive = new Float64Array(T + 1).fill(NaN);
    for (let t = 1; t <= T; t++) { let sum = 0; for (let s = 0; s < S; s++) sum += dist[t][s]; propAlive[t] = sum; }
    return { propAlive, cumMortality, cumStarvation, stateDist: dist, expectedFitness: expected };
  }
}
