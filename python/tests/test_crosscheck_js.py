"""Cross-check: the JavaScript engine (docs/engine.js) and the Python engine
must produce identical F and D for the same model (SPEC.md preamble).

Skipped when node is not installed.
"""

from __future__ import annotations

import copy
import json
import math
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
EXAMPLES = os.path.join(ROOT, "examples")
CROSSCHECK = os.path.join(ROOT, "docs", "tools", "crosscheck.mjs")
sys.path.insert(0, os.path.join(ROOT, "python"))

from dsvm import Model, ModelError  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


def run_js(model_dict: dict, tmp_path) -> dict:
    p = tmp_path / "model.json"
    p.write_text(json.dumps(model_dict))
    out = subprocess.run(["node", CROSSCHECK, str(p)], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def compare(model_dict: dict, tmp_path, ftol=1e-12):
    js = run_js(model_dict, tmp_path)
    assert "error" not in js, js.get("error")
    model = Model.from_dict(model_dict)
    sol = model.solve()
    assert [list(s) for s in model.states] == js["states"], "state ordering differs"
    F_js = np.array([[np.nan if v is None else v for v in row] for row in js["F"]])
    D_js = np.array(js["D"])
    T = model.T
    assert np.array_equal(D_js[:, 1:], sol.D[:, 1:]), "decision matrices differ"
    diff = np.nanmax(np.abs(F_js[:, 1:] - sol.F[:, 1:]))
    assert diff < ftol, f"fitness matrices differ by up to {diff}"
    proj = model.project(sol)
    assert math.isclose(js["projection"]["expectedFitness"], proj.expected_fitness, rel_tol=0, abs_tol=1e-9)
    assert np.allclose(np.array(js["projection"]["propAlive"][1:], dtype=float), proj.prop_alive[1:], atol=1e-9)
    # Monte Carlo agrees statistically (different RNGs): within 5 SE of the exact value
    p = proj.prop_alive[T]
    n = js["simulation"]["n"]
    se = math.sqrt(max(p * (1 - p), 1e-12) / n)
    assert abs(js["simulation"]["propAliveT"] - p) <= 5 * se + 1e-12
    assert len(js["roundingWarnings"]) == len(sol.rounding_warnings)
    return model, sol, js


@pytest.mark.parametrize("fname", sorted(f for f in os.listdir(EXAMPLES) if f.endswith(".json")))
def test_examples_match(fname, tmp_path):
    with open(os.path.join(EXAMPLES, fname)) as f:
        compare(json.load(f), tmp_path)


def base_model() -> dict:
    return {
        "format": "dsvm-model", "version": 1, "name": "cross-check", "horizon": 12,
        "state_variables": [{"name": "x", "min": 0, "max": 8, "lethal_min": True, "initial": 4}],
        "actions": [
            {"name": "A", "mortality": "0.01", "outcomes": [
                {"prob": "0.5", "change": {"x": "2"}}, {"prob": "0.5", "change": {"x": "-1"}}]},
            {"name": "B", "mortality": "0.05", "outcomes": [
                {"prob": "0.8", "change": {"x": "3"}}, {"prob": "0.2", "change": {"x": "-2"}}]},
        ],
        "terminal_fitness": "1",
        "simulation": {"individuals": 500, "seed": 3},
    }


def test_ties_first_action_wins(tmp_path):
    m = base_model()
    m["actions"][1] = copy.deepcopy(m["actions"][0])
    m["actions"][1]["name"] = "A-copy"
    model, sol, js = compare(m, tmp_path)
    live = ~model.dead
    assert (sol.D[live, 1:model.T] == 0).all()


def test_time_dependent_expressions(tmp_path):
    m = base_model()
    m["actions"][0]["mortality"] = "0.01 * t / T"
    m["actions"][1]["outcomes"][0]["prob"] = "0.8 - 0.3 * (t > T / 2)"
    m["actions"][1]["outcomes"][1]["prob"] = "1 - (0.8 - 0.3 * (t > T / 2))"
    m["terminal_fitness"] = "x >= 4 ? 1 : x / 4"
    compare(m, tmp_path)


def test_state_dependent_change_with_rounding(tmp_path):
    m = base_model()
    m["actions"][0]["outcomes"][0]["change"] = {"x": "0.5 * x"}
    model, sol, js = compare(m, tmp_path)
    assert sol.rounding_warnings and js["roundingWarnings"]


def test_three_state_variables_and_non_lethal_min(tmp_path):
    m = base_model()
    m["horizon"] = 6
    m["state_variables"] = [
        {"name": "energy", "min": 0, "max": 5, "lethal_min": True, "initial": 3},
        {"name": "gut", "min": 0, "max": 2, "lethal_min": False, "initial": 0},
        {"name": "eggs", "min": 0, "max": 3, "lethal_min": False, "initial": 0},
    ]
    m["actions"] = [
        {"name": "Forage", "mortality": "energy < 2 ? 0.05 : 0.02", "outcomes": [
            {"prob": "0.6 - 0.1 * gut", "change": {"energy": "min(gut, 1) - 1", "gut": "2 - min(gut, 1)"}},
            {"prob": "1 - (0.6 - 0.1 * gut)", "change": {"energy": "min(gut, 1) - 1", "gut": "-min(gut, 1)"}}]},
        {"name": "Lay", "mortality": "0.01", "reward": "energy >= 2 ? 1 : 0", "outcomes": [
            {"prob": "1", "change": {"energy": "-2", "eggs": "energy >= 2 ? 1 : 0"}}]},
    ]
    m["terminal_fitness"] = "eggs + 0.1 * energy"
    compare(m, tmp_path)


def test_invalid_model_errors_agree(tmp_path):
    m = base_model()
    m["actions"][0]["outcomes"][0]["prob"] = "0.6"  # sums to 1.1
    js = run_js(m, tmp_path)
    assert "error" in js and "sum to" in js["error"]
    with pytest.raises(ModelError, match="sum to"):
        Model.from_dict(m).solve()
