"""Load/save model files and export results as CSV (SPEC.md §6, §7)."""

from __future__ import annotations

import csv
import json
import os
from typing import Iterable, List

import numpy as np

from .engine import CAUSE_NAMES, Model, Projection, Simulation, Solution


def load(path: str) -> Model:
    with open(path, encoding="utf-8") as f:
        return Model.from_dict(json.load(f))


def loads(text: str) -> Model:
    return Model.from_dict(json.loads(text))


def save(model: Model, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(model.to_dict(), f, indent=2, ensure_ascii=False)
        f.write("\n")


def _write_csv(path: str, header: List[str], rows: Iterable[Iterable]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def _fmt(v: float) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    return f"{v:.10g}"


def fitness_rows(model: Model, sol: Solution):
    T = model.T
    header = list(model.var_names) + [f"t={t}" for t in range(1, T + 1)]
    rows = []
    for s_idx, state in enumerate(model.states):
        rows.append(list(state) + [_fmt(float(sol.F[s_idx, t])) for t in range(1, T + 1)])
    return header, rows


def decision_rows(model: Model, sol: Solution):
    T = model.T
    header = list(model.var_names) + [f"t={t}" for t in range(1, T + 1)]
    rows = []
    for s_idx, state in enumerate(model.states):
        cells = []
        for t in range(1, T + 1):
            a = int(sol.D[s_idx, t])
            cells.append(model.actions[a].name if a >= 0 else "")
        rows.append(list(state) + cells)
    return header, rows


def simulation_rows(model: Model, sim: Simulation, proj: Projection | None = None):
    T = model.T
    header = ["t", "prop_alive", "cum_dead_mortality", "cum_dead_starvation"]
    header += [f"mean_{v}" for v in model.var_names]
    header += [f"freq_{a.name}" for a in model.actions]
    if proj is not None:
        header += ["exact_prop_alive", "exact_cum_dead_mortality", "exact_cum_dead_starvation"]
    rows = []
    for t in range(1, T + 1):
        r = [t, _fmt(sim.prop_alive[t]), _fmt(sim.cum_mortality[t]), _fmt(sim.cum_starvation[t])]
        r += [_fmt(sim.mean_state[t, k]) for k in range(len(model.var_names))]
        r += [_fmt(sim.action_freq[t, a]) for a in range(len(model.actions))]
        if proj is not None:
            r += [_fmt(proj.prop_alive[t]), _fmt(proj.cum_mortality[t]), _fmt(proj.cum_starvation[t])]
        rows.append(r)
    return header, rows


def individual_rows(model: Model, sim: Simulation):
    """One row per individual per period it was alive: long format."""
    T, N = model.T, len(model.var_names)
    header = ["individual", "t", "alive"] + list(model.var_names) + ["action", "cause_of_death", "death_time"]
    rows = []
    for i in range(sim.n):
        for t in range(1, T + 1):
            if not sim.alive[i, t]:
                continue
            a = int(sim.actions[i, t])
            rows.append([i + 1, t, 1] + [int(sim.states[i, t, k]) for k in range(N)]
                        + [model.actions[a].name if a >= 0 else "",
                           CAUSE_NAMES[int(sim.cause[i])],
                           int(sim.death_time[i]) if sim.death_time[i] >= 0 else ""])
    return header, rows


def export_all(model: Model, sol: Solution, out_dir: str, sim: Simulation | None = None,
               proj: Projection | None = None, individuals: bool = False) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    written = []
    p = os.path.join(out_dir, "fitness_F.csv")
    _write_csv(p, *fitness_rows(model, sol))
    written.append(p)
    p = os.path.join(out_dir, "decision_D.csv")
    _write_csv(p, *decision_rows(model, sol))
    written.append(p)
    if sim is not None:
        p = os.path.join(out_dir, "simulation_summary.csv")
        _write_csv(p, *simulation_rows(model, sim, proj))
        written.append(p)
        if individuals:
            p = os.path.join(out_dir, "simulation_individuals.csv")
            _write_csv(p, *individual_rows(model, sim))
            written.append(p)
    p = os.path.join(out_dir, "model.dsvm.json")
    save(model, p)
    written.append(p)
    return written
