// CSV export matching SPEC.md §7 and python/dsvm/io.py.
import { CAUSE_NAMES } from "./engine.js";

function fmt(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return "";
  return String(Number(Number(v).toPrecision(10)));
}
function csvCell(s) {
  s = String(s);
  return /[",\n\r]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
}
function csv(header, rows) {
  return [header, ...rows].map((r) => r.map(csvCell).join(",")).join("\r\n") + "\r\n";
}

export function fitnessCSV(model, sol) {
  const T = model.T;
  const header = [...model.varNames, ...Array.from({ length: T }, (_, i) => `t=${i + 1}`)];
  const rows = model.states.map((st, s) => [...st, ...Array.from({ length: T }, (_, i) => fmt(sol.F[s][i + 1]))]);
  return csv(header, rows);
}

export function decisionCSV(model, sol) {
  const T = model.T;
  const header = [...model.varNames, ...Array.from({ length: T }, (_, i) => `t=${i + 1}`)];
  const rows = model.states.map((st, s) => [...st, ...Array.from({ length: T }, (_, i) => {
    const a = sol.D[s][i + 1];
    return a >= 0 ? model.actions[a].name : "";
  })]);
  return csv(header, rows);
}

export function simulationCSV(model, sim, proj = null) {
  const T = model.T;
  const header = ["t", "prop_alive", "cum_dead_mortality", "cum_dead_starvation",
    ...model.varNames.map((v) => `mean_${v}`), ...model.actions.map((a) => `freq_${a.name}`)];
  if (proj) header.push("exact_prop_alive", "exact_cum_dead_mortality", "exact_cum_dead_starvation");
  const rows = [];
  for (let t = 1; t <= T; t++) {
    const r = [t, fmt(sim.propAlive[t]), fmt(sim.cumMortality[t]), fmt(sim.cumStarvation[t]),
      ...model.varNames.map((_, k) => fmt(sim.meanState[t][k])),
      ...model.actions.map((_, a) => fmt(sim.actionFreq[t][a]))];
    if (proj) r.push(fmt(proj.propAlive[t]), fmt(proj.cumMortality[t]), fmt(proj.cumStarvation[t]));
    rows.push(r);
  }
  return csv(header, rows);
}

export function individualsCSV(model, sim) {
  const T = model.T;
  const header = ["individual", "t", "alive", ...model.varNames, "action", "cause_of_death", "death_time"];
  const rows = [];
  for (let i = 0; i < sim.n; i++) {
    for (let t = 1; t <= T; t++) {
      if (!sim.isAlive(i, t)) continue;
      const a = sim.actionOf(i, t);
      rows.push([i + 1, t, 1, ...sim.stateOf(i, t), a >= 0 ? model.actions[a].name : "",
        CAUSE_NAMES[sim.cause[i]], sim.deathTime[i] >= 0 ? sim.deathTime[i] : ""]);
    }
  }
  return csv(header, rows);
}

export function modelJSON(model) {
  return JSON.stringify(model.toObject(), null, 2) + "\n";
}
