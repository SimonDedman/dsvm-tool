"""Tests for CSV/JSON export, SPEC.md §6, §7 (dsvm/io.py)."""

from __future__ import annotations

import csv
import json

import pytest

from dsvm.io import export_all, load


@pytest.fixture
def solved_model(examples_dir):
    m = load(str(examples_dir / "patch_selection.dsvm.json"))
    sol = m.solve()
    proj = m.project(sol)
    sim = m.simulate(sol, n=200, seed=1)
    return m, sol, sim, proj


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.reader(f))


def test_export_all_without_simulation_writes_only_F_D_and_model(solved_model, tmp_path):
    m, sol, sim, proj = solved_model
    files = export_all(m, sol, str(tmp_path))
    names = {p.split("/")[-1] for p in files}
    assert names == {"fitness_F.csv", "decision_D.csv", "model.dsvm.json"}
    for f in files:
        assert (tmp_path / f.split("/")[-1]).exists()


def test_export_all_with_simulation_writes_summary(solved_model, tmp_path):
    m, sol, sim, proj = solved_model
    files = export_all(m, sol, str(tmp_path), sim=sim, proj=proj, individuals=False)
    names = {p.split("/")[-1] for p in files}
    assert names == {"fitness_F.csv", "decision_D.csv", "simulation_summary.csv", "model.dsvm.json"}


def test_export_all_with_individuals_writes_individual_rows(solved_model, tmp_path):
    m, sol, sim, proj = solved_model
    files = export_all(m, sol, str(tmp_path), sim=sim, proj=proj, individuals=True)
    names = {p.split("/")[-1] for p in files}
    assert names == {"fitness_F.csv", "decision_D.csv", "simulation_summary.csv",
                      "simulation_individuals.csv", "model.dsvm.json"}


def test_fitness_csv_header_shape(solved_model, tmp_path):
    m, sol, sim, proj = solved_model
    export_all(m, sol, str(tmp_path))
    rows = read_csv(tmp_path / "fitness_F.csv")
    header = rows[0]
    assert header == list(m.var_names) + [f"t={t}" for t in range(1, m.T + 1)]
    assert len(rows) - 1 == m.S  # one data row per state


def test_decision_csv_header_shape(solved_model, tmp_path):
    m, sol, sim, proj = solved_model
    export_all(m, sol, str(tmp_path))
    rows = read_csv(tmp_path / "decision_D.csv")
    header = rows[0]
    assert header == list(m.var_names) + [f"t={t}" for t in range(1, m.T + 1)]
    assert len(rows) - 1 == m.S


def test_decision_csv_has_action_names(solved_model, tmp_path):
    m, sol, sim, proj = solved_model
    export_all(m, sol, str(tmp_path))
    rows = read_csv(tmp_path / "decision_D.csv")
    action_names = {a.name for a in m.actions}
    seen_names = set()
    for row in rows[1:]:
        for cell in row[1:]:
            if cell != "":
                seen_names.add(cell)
    assert seen_names <= action_names
    assert seen_names  # at least one non-empty decision cell


def test_decision_csv_empty_for_dead_state_and_t_equals_T(solved_model, tmp_path):
    m, sol, sim, proj = solved_model
    export_all(m, sol, str(tmp_path))
    rows = read_csv(tmp_path / "decision_D.csv")
    header = rows[0]
    t_T_col = header.index(f"t={m.T}")
    for s_idx, row in enumerate(rows[1:]):
        state = tuple(int(x) for x in row[:len(m.var_names)])
        assert m.state_index[state] == s_idx
        # t=T column is always empty (no decision at T)
        assert row[t_T_col] == ""
        if m.dead[s_idx]:
            # every t column is empty for a dead state
            for t in range(1, m.T + 1):
                col = header.index(f"t={t}")
                assert row[col] == ""


def test_fitness_csv_values_round_trip_within_tolerance(solved_model, tmp_path):
    m, sol, sim, proj = solved_model
    export_all(m, sol, str(tmp_path))
    rows = read_csv(tmp_path / "fitness_F.csv")
    n_vars = len(m.var_names)
    for s_idx, row in enumerate(rows[1:]):
        state = tuple(int(x) for x in row[:n_vars])
        assert m.state_index[state] == s_idx
        for t in range(1, m.T + 1):
            cell = row[n_vars + t - 1]
            expected = sol.F[s_idx, t]
            assert cell != ""  # F is defined at every t = 1..T, dead or alive
            assert float(cell) == pytest.approx(float(expected), abs=1e-9)


def test_model_json_round_trip_exported_file(solved_model, tmp_path):
    m, sol, sim, proj = solved_model
    export_all(m, sol, str(tmp_path))
    with open(tmp_path / "model.dsvm.json", encoding="utf-8") as f:
        d = json.load(f)
    assert d["format"] == "dsvm-model"
    assert d["horizon"] == m.T
    assert len(d["state_variables"]) == len(m.state_variables)
    assert len(d["actions"]) == len(m.actions)


def test_simulation_summary_csv_shape(solved_model, tmp_path):
    m, sol, sim, proj = solved_model
    export_all(m, sol, str(tmp_path), sim=sim, proj=proj)
    rows = read_csv(tmp_path / "simulation_summary.csv")
    assert len(rows) - 1 == m.T  # one row per t = 1..T
    header = rows[0]
    assert header[0] == "t"
    assert "prop_alive" in header
    assert "exact_prop_alive" in header  # present because proj was supplied


def test_simulation_summary_csv_without_projection(solved_model, tmp_path):
    m, sol, sim, proj = solved_model
    export_all(m, sol, str(tmp_path), sim=sim)
    rows = read_csv(tmp_path / "simulation_summary.csv")
    header = rows[0]
    assert "exact_prop_alive" not in header


def test_individual_rows_csv_shape(solved_model, tmp_path):
    m, sol, sim, proj = solved_model
    export_all(m, sol, str(tmp_path), sim=sim, proj=proj, individuals=True)
    rows = read_csv(tmp_path / "simulation_individuals.csv")
    header = rows[0]
    assert header[:3] == ["individual", "t", "alive"]
    # every remaining row belongs to an individual that was alive at that t
    for row in rows[1:]:
        assert row[2] == "1"
