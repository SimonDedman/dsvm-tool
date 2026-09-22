"""Tests for the CLI, `python -m dsvm ...` (dsvm/cli.py)."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

EXAMPLE_FILES = [
    "patch_selection.dsvm.json",
    "reimer2019_two_patch.dsvm.json",
    "two_state_energy_gut.dsvm.json",
]


def run_cli(python_dir, *args):
    return subprocess.run(
        [sys.executable, "-m", "dsvm", *args],
        cwd=str(python_dir),
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.fixture
def python_dir():
    from pathlib import Path
    return Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("filename", EXAMPLE_FILES)
def test_validate_ok_on_examples(python_dir, examples_dir, filename):
    result = run_cli(python_dir, "validate", str(examples_dir / filename))
    assert result.returncode == 0, result.stderr
    assert "ok:" in result.stdout


@pytest.mark.parametrize("filename", EXAMPLE_FILES)
def test_show_prints_optimal_decision(python_dir, examples_dir, filename):
    result = run_cli(python_dir, "show", str(examples_dir / filename))
    assert result.returncode == 0, result.stderr
    assert "Optimal decision" in result.stdout


def test_run_writes_expected_files(python_dir, examples_dir, tmp_path):
    out_dir = tmp_path / "out"
    result = run_cli(python_dir, "run", str(examples_dir / "patch_selection.dsvm.json"),
                      "-o", str(out_dir), "-n", "200")
    assert result.returncode == 0, result.stderr
    assert (out_dir / "fitness_F.csv").exists()
    assert (out_dir / "decision_D.csv").exists()
    assert (out_dir / "simulation_summary.csv").exists()
    assert (out_dir / "model.dsvm.json").exists()
    assert not (out_dir / "simulation_individuals.csv").exists()


def test_run_with_per_individual_writes_individuals_file(python_dir, examples_dir, tmp_path):
    out_dir = tmp_path / "out_ind"
    result = run_cli(python_dir, "run", str(examples_dir / "patch_selection.dsvm.json"),
                      "-o", str(out_dir), "-n", "50", "--per-individual")
    assert result.returncode == 0, result.stderr
    assert (out_dir / "simulation_individuals.csv").exists()


def test_solve_writes_F_and_D_only(python_dir, examples_dir, tmp_path):
    out_dir = tmp_path / "out_solve"
    result = run_cli(python_dir, "solve", str(examples_dir / "patch_selection.dsvm.json"),
                      "-o", str(out_dir))
    assert result.returncode == 0, result.stderr
    assert (out_dir / "fitness_F.csv").exists()
    assert (out_dir / "decision_D.csv").exists()
    assert not (out_dir / "simulation_summary.csv").exists()


def test_validate_on_broken_model_is_nonzero_and_says_invalid(python_dir, tmp_path):
    broken = {
        "format": "dsvm-model", "version": 1, "name": "broken", "horizon": 3,
        "state_variables": [{"name": "x", "min": 0, "max": 3, "lethal_min": True, "initial": 2}],
        "actions": [{"name": "A", "mortality": "0", "reward": "0",
                     "outcomes": [{"prob": "0.5", "change": {"x": "0"}},
                                  {"prob": "0.4", "change": {"x": "0"}}]}],
        "terminal_fitness": "1",
    }
    p = tmp_path / "broken.dsvm.json"
    p.write_text(json.dumps(broken), encoding="utf-8")
    result = run_cli(python_dir, "validate", str(p))
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "invalid" in combined


def test_validate_on_missing_file_is_nonzero(python_dir, tmp_path):
    missing = tmp_path / "does_not_exist.dsvm.json"
    result = run_cli(python_dir, "validate", str(missing))
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "not found" in combined or "error" in combined
