"""Tests for the DP engine, SPEC.md §1, §2, §4, §5 (dsvm/engine.py)."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from dsvm.engine import Model, ModelError
from dsvm.io import load

from conftest import minimal_model_dict


# --------------------------------------------------------------------------- patch_selection

def test_patch_selection_regression_values(examples_dir):
    m = load(str(examples_dir / "patch_selection.dsvm.json"))
    sol = m.solve()
    assert m.T == 20

    idx3 = m.state_index[(3,)]
    assert sol.F[idx3, 1] == pytest.approx(0.0, abs=1e-6)

    expected_F = {4: 0.513651, 5: 0.723863, 6: 0.811538, 7: 0.846181,
                  8: 0.865991, 9: 0.877955, 10: 0.885622}
    for x, f in expected_F.items():
        idx = m.state_index[(x,)]
        assert sol.F[idx, 1] == pytest.approx(f, abs=1e-6), f"x={x}"

    expected_D = {4: 2, 5: 2, 6: 2, 7: 2, 8: 1, 9: 1, 10: 0}
    for x, a in expected_D.items():
        idx = m.state_index[(x,)]
        assert int(sol.D[idx, 1]) == a, f"x={x}"


def test_patch_selection_dead_state_D_is_minus_one(examples_dir):
    m = load(str(examples_dir / "patch_selection.dsvm.json"))
    sol = m.solve()
    idx3 = m.state_index[(3,)]  # x=3 is at the lethal threshold
    assert m.dead[idx3]
    assert (sol.D[idx3, :] == -1).all()


def test_D_at_horizon_T_is_minus_one_for_all_states(examples_dir):
    m = load(str(examples_dir / "patch_selection.dsvm.json"))
    sol = m.solve()
    assert (sol.D[:, m.T] == -1).all()


# --------------------------------------------------------------------------- reimer2019_two_patch

def test_reimer_decisions_far_from_horizon(examples_dir):
    m = load(str(examples_dir / "reimer2019_two_patch.dsvm.json"))
    assert m.T == 100
    sol = m.solve()
    for x in (1, 2):
        idx = m.state_index[(x,)]
        assert int(sol.D[idx, 1]) == 1, f"x={x} should choose Patch 2 (index 1)"
    for x in (3, 4, 5):
        idx = m.state_index[(x,)]
        assert int(sol.D[idx, 1]) == 0, f"x={x} should choose Patch 1 (index 0)"


def test_reimer_dominant_eigenvalue(examples_dir):
    m = load(str(examples_dir / "reimer2019_two_patch.dsvm.json"))
    sol = m.solve()
    proj = m.project(sol)
    ratios = []
    for t in range(40, 61):
        ratios.append(proj.prop_alive[t + 1] / proj.prop_alive[t])
    for r in ratios:
        assert abs(r - 0.97) < 0.005


# --------------------------------------------------------------------------- Model.from_dict validation

def test_probabilities_not_summing_to_one_names_action():
    d = minimal_model_dict()
    d["actions"][0]["outcomes"] = [
        {"prob": "0.5", "change": {"x": "0"}},
        {"prob": "0.4", "change": {"x": "0"}},
    ]
    m = Model.from_dict(d)
    with pytest.raises(ModelError, match="'A'"):
        m.table


def test_mortality_greater_than_one_raises():
    d = minimal_model_dict()
    d["actions"][0]["mortality"] = "1.5"
    m = Model.from_dict(d)
    with pytest.raises(ModelError, match=r"outside \[0, 1\]"):
        m.table


def test_unknown_variable_in_mortality_raises_with_expression_text():
    d = minimal_model_dict()
    d["actions"][0]["mortality"] = "y"
    with pytest.raises(ModelError, match="'y'"):
        Model.from_dict(d)


def test_change_naming_unknown_variable_raises():
    d = minimal_model_dict()
    d["actions"][0]["outcomes"][0]["change"] = {"not_a_state_var": "0"}
    with pytest.raises(ModelError, match="unknown state variable"):
        Model.from_dict(d)


def test_duplicate_variable_names_raise():
    d = minimal_model_dict()
    d["state_variables"].append({"name": "x", "min": 0, "max": 2, "lethal_min": False, "initial": 0})
    with pytest.raises(ModelError, match="unique"):
        Model.from_dict(d)


def test_horizon_less_than_two_raises():
    d = minimal_model_dict()
    d["horizon"] = 1
    with pytest.raises(ModelError, match="horizon"):
        Model.from_dict(d)


def test_initial_at_lethal_threshold_raises():
    d = minimal_model_dict()
    d["state_variables"][0]["initial"] = d["state_variables"][0]["min"]
    with pytest.raises(ModelError, match="lethal threshold"):
        Model.from_dict(d)


def test_non_integer_change_gives_rounding_warning_and_completes():
    d = minimal_model_dict()
    d["actions"][0]["outcomes"][0]["change"] = {"x": "0.5"}
    m = Model.from_dict(d)
    sol = m.solve()
    assert len(sol.rounding_warnings) >= 1
    assert "0.5" in sol.rounding_warnings[0]


def test_lethal_min_false_at_min_is_not_dead():
    d = minimal_model_dict()
    d["state_variables"][0]["lethal_min"] = False
    d["state_variables"][0]["min"] = 0
    d["state_variables"][0]["initial"] = 0
    m = Model.from_dict(d)
    assert m.is_dead((0,)) is False


def test_lethal_min_true_at_min_is_dead():
    d = minimal_model_dict()  # min=0 default lethal_min True, but initial=2 elsewhere
    m = Model.from_dict(d)
    assert m.is_dead((d["state_variables"][0]["min"],)) is True


def test_ties_go_to_first_listed_action():
    d = minimal_model_dict()
    outcome_set = [
        {"prob": "0.5", "change": {"x": "-1"}},
        {"prob": "0.5", "change": {"x": "1"}},
    ]
    d["actions"] = [
        {"name": "A", "mortality": "0.1", "reward": "0", "outcomes": copy.deepcopy(outcome_set)},
        {"name": "B", "mortality": "0.1", "reward": "0", "outcomes": copy.deepcopy(outcome_set)},
    ]
    m = Model.from_dict(d)
    sol = m.solve()
    for s_idx, state in enumerate(m.states):
        if m.dead[s_idx]:
            continue
        for t in range(1, m.T):
            assert int(sol.D[s_idx, t]) == 0


# --------------------------------------------------------------------------- time-dependent expressions

def test_time_dependent_mortality_solves_and_differs_from_constant():
    d = minimal_model_dict()
    d["state_variables"][0]["lethal_min"] = False
    d["state_variables"][0]["min"] = 0
    d["state_variables"][0]["max"] = 1
    d["state_variables"][0]["initial"] = 1
    d["actions"][0]["outcomes"][0]["change"] = {"x": "0"}
    d["horizon"] = 5

    d_rising = copy.deepcopy(d)
    d_rising["actions"][0]["mortality"] = "t / T"
    m_rising = Model.from_dict(d_rising)
    sol_rising = m_rising.solve()  # must not raise

    d_const = copy.deepcopy(d)
    d_const["actions"][0]["mortality"] = "0.2"  # equals t/T at t=1 for T=5
    m_const = Model.from_dict(d_const)
    sol_const = m_const.solve()

    idx = m_rising.state_index[(1,)]
    assert sol_rising.F[idx, 1] != pytest.approx(sol_const.F[idx, 1], abs=1e-9)


def test_time_dependent_survival_product_hand_check():
    """1 state, 1 action, T=3, mortality = t/T, only outcome 'stay' (change 0), phi=1.

    F(x,1) must equal the product of (1 - beta(t)) over t=1,2.
    """
    d = minimal_model_dict()
    d["state_variables"][0]["lethal_min"] = False
    d["state_variables"][0]["min"] = 0
    d["state_variables"][0]["max"] = 1
    d["state_variables"][0]["initial"] = 1
    d["horizon"] = 3
    d["actions"][0]["mortality"] = "t / T"
    d["actions"][0]["outcomes"][0]["change"] = {"x": "0"}
    d["terminal_fitness"] = "1"
    m = Model.from_dict(d)
    sol = m.solve()
    idx = m.state_index[(1,)]

    beta = lambda t: t / m.T
    expected = (1 - beta(1)) * (1 - beta(2))
    assert sol.F[idx, 1] == pytest.approx(expected, abs=1e-12)


# --------------------------------------------------------------------------- hand-computed backward step

def test_hand_computed_backward_step():
    """1 state var x in [0,3], lethal at 0, T=2, one action:
    mortality 0.1, outcomes 0.5/change -1 and 0.5/change +1, phi=1.

    F(1,1) = 0.9*(0.5*0 + 0.5*1) = 0.45
    F(2,1) = 0.9*(0.5*1 + 0.5*1) = 0.9
    F(3,1) = 0.9*(0.5*1 + 0.5*1) = 0.9   (x+1 chopped to 3, stays alive)
    """
    d = {
        "format": "dsvm-model", "version": 1, "name": "hand", "horizon": 2,
        "state_variables": [{"name": "x", "min": 0, "max": 3, "lethal_min": True, "initial": 2}],
        "actions": [
            {"name": "A", "mortality": "0.1", "reward": "0", "outcomes": [
                {"prob": "0.5", "change": {"x": "-1"}},
                {"prob": "0.5", "change": {"x": "1"}},
            ]},
        ],
        "terminal_fitness": "1",
    }
    m = Model.from_dict(d)
    sol = m.solve()
    assert sol.F[m.state_index[(1,)], 1] == pytest.approx(0.45, abs=1e-12)
    assert sol.F[m.state_index[(2,)], 1] == pytest.approx(0.9, abs=1e-12)
    assert sol.F[m.state_index[(3,)], 1] == pytest.approx(0.9, abs=1e-12)


# --------------------------------------------------------------------------- project()

@pytest.mark.parametrize("filename", [
    "patch_selection.dsvm.json",
    "reimer2019_two_patch.dsvm.json",
    "two_state_energy_gut.dsvm.json",
])
def test_project_expected_fitness_matches_F_initial(examples_dir, filename):
    m = load(str(examples_dir / filename))
    sol = m.solve()
    proj = m.project(sol)
    init_idx = m.state_index[m.initial_state]
    assert proj.expected_fitness == pytest.approx(sol.F[init_idx, 1], abs=1e-9)


@pytest.mark.parametrize("filename", [
    "patch_selection.dsvm.json",
    "reimer2019_two_patch.dsvm.json",
    "two_state_energy_gut.dsvm.json",
])
def test_project_cum_probabilities_sum_to_one(examples_dir, filename):
    m = load(str(examples_dir / filename))
    sol = m.solve()
    proj = m.project(sol)
    for t in range(1, m.T + 1):
        total = proj.cum_mortality[t] + proj.cum_starvation[t] + proj.prop_alive[t]
        assert total == pytest.approx(1.0, abs=1e-9), f"t={t}"


# --------------------------------------------------------------------------- simulate()

SIM_N = 8000
SIM_SEED = 20260921


@pytest.mark.parametrize("filename", [
    "patch_selection.dsvm.json",
    "reimer2019_two_patch.dsvm.json",
    "two_state_energy_gut.dsvm.json",
])
class TestSimulate:
    def test_prop_alive_matches_projection_within_4se(self, examples_dir, filename):
        m = load(str(examples_dir / filename))
        sol = m.solve()
        proj = m.project(sol)
        sim = m.simulate(sol, n=SIM_N, seed=SIM_SEED)
        p = proj.prop_alive[m.T]
        se = (p * (1 - p) / SIM_N) ** 0.5
        assert abs(sim.prop_alive[m.T] - p) <= 4 * se

    def test_same_seed_gives_identical_arrays(self, examples_dir, filename):
        m = load(str(examples_dir / filename))
        sol = m.solve()
        sim1 = m.simulate(sol, n=500, seed=7)
        sim2 = m.simulate(sol, n=500, seed=7)
        assert np.array_equal(sim1.alive, sim2.alive)
        assert np.array_equal(sim1.states, sim2.states)

    def test_different_seed_gives_different_arrays(self, examples_dir, filename):
        m = load(str(examples_dir / filename))
        sol = m.solve()
        sim1 = m.simulate(sol, n=500, seed=7)
        sim2 = m.simulate(sol, n=500, seed=8)
        assert not np.array_equal(sim1.alive, sim2.alive) or not np.array_equal(sim1.states, sim2.states)

    def test_realised_fitness_matches_expected_within_4se(self, examples_dir, filename):
        m = load(str(examples_dir / filename))
        sol = m.solve()
        sim = m.simulate(sol, n=SIM_N, seed=SIM_SEED)
        p = sol.F[m.state_index[m.initial_state], 1]
        se = (p * (1 - p) / SIM_N) ** 0.5
        assert abs(sim.realised_fitness - p) <= 4 * se

    def test_alive_at_t1_all_true(self, examples_dir, filename):
        m = load(str(examples_dir / filename))
        sol = m.solve()
        sim = m.simulate(sol, n=200, seed=1)
        assert sim.alive[:, 1].all()

    def test_actions_at_T_are_minus_one(self, examples_dir, filename):
        m = load(str(examples_dir / filename))
        sol = m.solve()
        sim = m.simulate(sol, n=200, seed=1)
        assert (sim.actions[:, m.T] == -1).all()

    def test_cause_codes_in_valid_set(self, examples_dir, filename):
        m = load(str(examples_dir / filename))
        sol = m.solve()
        sim = m.simulate(sol, n=200, seed=1)
        assert set(np.unique(sim.cause)).issubset({0, 1, 2})

    def test_death_time_at_least_one_when_not_alive_cause(self, examples_dir, filename):
        m = load(str(examples_dir / filename))
        sol = m.solve()
        sim = m.simulate(sol, n=200, seed=1)
        died = sim.cause != 0
        if died.any():
            assert (sim.death_time[died] >= 1).all()

    def test_mean_state_at_t1_equals_initial(self, examples_dir, filename):
        m = load(str(examples_dir / filename))
        sol = m.solve()
        sim = m.simulate(sol, n=200, seed=1)
        assert np.allclose(sim.mean_state[1], np.array(m.initial_state, dtype=float))


# --------------------------------------------------------------------------- round trip

@pytest.mark.parametrize("filename", [
    "patch_selection.dsvm.json",
    "reimer2019_two_patch.dsvm.json",
    "two_state_energy_gut.dsvm.json",
])
def test_round_trip_gives_identical_F_and_D(examples_dir, filename):
    m = load(str(examples_dir / filename))
    sol = m.solve()
    m2 = Model.from_dict(m.to_dict())
    sol2 = m2.solve()
    assert np.allclose(np.nan_to_num(sol.F), np.nan_to_num(sol2.F))
    assert np.array_equal(sol.D, sol2.D)


def test_unknown_top_level_keys_survive_round_trip():
    d = minimal_model_dict()
    d["some_unknown_key"] = {"nested": [1, 2, 3]}
    m = Model.from_dict(d)
    out = m.to_dict()
    assert out.get("some_unknown_key") == {"nested": [1, 2, 3]}


# --------------------------------------------------------------------------- reward model

def test_per_period_reward_model():
    """One action, reward '1', mortality 0, one outcome change 0, phi=0, T=5.

    F(x,1) = 4 (rewards accrue at t=1..4). project() must agree.
    """
    d = minimal_model_dict()
    d["horizon"] = 5
    d["actions"][0]["mortality"] = "0"
    d["actions"][0]["reward"] = "1"
    d["actions"][0]["outcomes"][0]["change"] = {"x": "0"}
    d["terminal_fitness"] = "0"
    m = Model.from_dict(d)
    sol = m.solve()
    idx = m.state_index[(m.initial_state)]
    assert sol.F[idx, 1] == pytest.approx(4.0, abs=1e-12)
    proj = m.project(sol)
    assert proj.expected_fitness == pytest.approx(4.0, abs=1e-9)
