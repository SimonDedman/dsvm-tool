"""Shared fixtures for the dsvm test suite."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest


@pytest.fixture
def examples_dir() -> Path:
    """Absolute path to the repo's examples/ directory, resolved relative to this file.

    Layout: <repo>/python/tests/conftest.py  ->  <repo>/examples/
    """
    return (Path(__file__).resolve().parent.parent.parent / "examples").resolve()


def minimal_model_dict() -> dict:
    """A minimal, valid dsvm-model dict. Tests mutate a deep copy of this."""
    return {
        "format": "dsvm-model",
        "version": 1,
        "name": "minimal test model",
        "description": "",
        "horizon": 3,
        "state_variables": [
            {"name": "x", "label": "x", "min": 0, "max": 3, "lethal_min": True, "initial": 2},
        ],
        "actions": [
            {
                "name": "A",
                "mortality": "0",
                "reward": "0",
                "outcomes": [
                    {"label": "stay", "prob": "1", "change": {"x": "0"}},
                ],
            },
        ],
        "terminal_fitness": "1",
        "simulation": {"individuals": 200, "seed": 1},
    }


@pytest.fixture
def minimal_model_factory():
    """Returns a callable that gives a fresh deep copy of the minimal model dict."""

    def _make():
        return copy.deepcopy(minimal_model_dict())

    return _make
