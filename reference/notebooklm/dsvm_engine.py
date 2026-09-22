# Verbatim copy of the engine NotebookLM produced for Mike Heithaus
# (email "Some AI code n stuff", 2026-09-16). Kept for provenance and for
# regression comparison. Not used by the tool itself. See ../README.md.

import itertools
import numpy as np
import pandas as pd

class StateVariable:
  """Defines a discrete integer state variable bounded by [x_min, x_max].

  x_min represents the critical starvation/depletion threshold (death if <=
  x_min).
  """

  def __init__(self, name, x_min, x_max, initial_val=None):
    self.name = name
    self.x_min = int(x_min)
    self.x_max = int(x_max)
    self.initial_val = (
        int(initial_val) if initial_val is not None else int(x_max)
    )
    self.values = list(range(self.x_min, self.x_max + 1))

class ActionOutcome:
  """Defines a stochastic outcome for a given behavioral choice.

  prob: probability of outcome occurrence (lambda_ij). delta_state: dictionary
  mapping state variable names to net state changes.
  """

  def __init__(self, prob, delta_state):
    self.prob = float(prob)
    self.delta_state = delta_state # e.g., {'Energy_Reserves': +4}

class Action:
  """Defines a behavioral choice (e.g., foraging patch, clutch size, activity).

  predation_prob: probability of mortality per time step (beta_i). outcomes:
  list of ActionOutcome objects.
  """

  def __init__(self, name, predation_prob, outcomes):
    self.name = name
    self.predation_prob = float(predation_prob)
    self.outcomes = outcomes

class DSVMSolver:
  """Solves the Dynamic Programming Equation (Backward Iteration) and executes

  Forward Individual-Based Monte Carlo Simulations based on Mangel & Clark
  (1988).
  """

  def __init__(
      self, state_vars, time_horizon, actions, terminal_fitness_fn=None
  ):
    self.state_vars = state_vars
    self.var_names = [v.name for v in state_vars]
    self.time_horizon = int(time_horizon) # T
    self.actions = actions

    # Generate full N-dimensional state grid
    ranges = [v.values for v in state_vars]
    self.all_states = list(itertools.product(*ranges))
    self.state_to_idx = {s: i for i, s in enumerate(self.all_states)}
    self.num_states = len(self.all_states)

    # Fitness matrix F[state_idx, t] and Decision matrix D[state_idx, t]
    self.F = np.zeros((self.num_states, self.time_horizon + 1))
    self.D = np.zeros((self.num_states, self.time_horizon + 1), dtype=int)

    # Terminal fitness function phi(X)
    if terminal_fitness_fn is not None:
      self.terminal_fitness_fn = terminal_fitness_fn
    else:

      def default_phi(state_dict):
        for v in self.state_vars:
          if state_dict[v.name] <= v.x_min:
            return 0.0
        return 1.0

      self.terminal_fitness_fn = default_phi

  def is_dead(self, state_tuple):
    """Returns True if any state variable is at or below its critical threshold."""
    for val, v_obj in zip(state_tuple, self.state_vars):
      if val <= v_obj.x_min:
        return True
    return False

  def chop_state(self, state_tuple, delta_dict):
    """Applies net state increments and clamps values into [x_min, x_max]."""
    new_vals = []
    for val, v_obj in zip(state_tuple, self.state_vars):
      change = delta_dict.get(v_obj.name, 0)
      new_val = val + change
      new_val = max(v_obj.x_min, min(v_obj.x_max, new_val))
      new_vals.append(new_val)
    return tuple(new_vals)

  def solve_backward(self):
    """Executes Backward Dynamic Programming iteration from t = T down to t = 1."""
    T = self.time_horizon

    # 1. Terminal Fitness Evaluation at t = T
    for s_idx, state_tuple in enumerate(self.all_states):
      if self.is_dead(state_tuple):
        self.F[s_idx, T] = 0.0
      else:
        state_dict = dict(zip(self.var_names, state_tuple))
        self.F[s_idx, T] = self.terminal_fitness_fn(state_dict)

    # 2. Backward Recursion: t = T-1 down to 1
    for t in range(T - 1, 0, -1):
      for s_idx, state_tuple in enumerate(self.all_states):
        if self.is_dead(state_tuple):
          self.F[s_idx, t] = 0.0
          self.D[s_idx, t] = 0
          continue

        best_fitness = -1.0
        best_action_idx = 0

        for a_idx, action in enumerate(self.actions):
          beta = action.predation_prob
          expected_future_fitness = 0.0

          for outcome in action.outcomes:
            next_state = self.chop_state(state_tuple, outcome.delta_state)
            if self.is_dead(next_state):
              next_f = 0.0
            else:
              next_s_idx = self.state_to_idx[next_state]
              next_f = self.F[next_s_idx, t + 1]

            expected_future_fitness += outcome.prob * next_f

          total_val = (1.0 - beta) * expected_future_fitness

          if total_val > best_fitness:
            best_fitness = total_val
            best_action_idx = a_idx

        self.F[s_idx, t] = best_fitness
        self.D[s_idx, t] = best_action_idx

    return self.F, self.D

  def run_monte_carlo(self, num_individuals=1000, initial_states=None, seed=42):
    """Runs an Individual-Based Forward Monte Carlo stochastic tracking simulation."""
    np.random.seed(seed)
    T = self.time_horizon

    if initial_states is None:
      init_tuple = tuple(v.initial_val for v in self.state_vars)
      pop_states = [list(init_tuple) for _ in range(num_individuals)]
    else:
      pop_states = [list(s) for s in initial_states]

    alive = [True] * num_individuals
    cause_of_death = [None] * num_individuals

    history_states = np.zeros(
        (num_individuals, T + 1, len(self.state_vars)), dtype=int
    )
    history_actions = np.zeros((num_individuals, T + 1), dtype=int)
    history_alive = np.ones((num_individuals, T + 1), dtype=bool)

    for i in range(num_individuals):
      history_states[i, 1] = pop_states[i]

    for t in range(1, T):
      for i in range(num_individuals):
        if not alive[i]:
          history_alive[i, t + 1] = False
          continue

        curr_state = tuple(pop_states[i])

        if self.is_dead(curr_state):
          alive[i] = False
          cause_of_death[i] = "starvation"
          history_alive[i, t + 1] = False
          continue

        s_idx = self.state_to_idx[curr_state]
        opt_a_idx = self.D[s_idx, t]
        action = self.actions[opt_a_idx]
        history_actions[i, t] = opt_a_idx

        # 1. Predation test
        if np.random.rand() < action.predation_prob:
          alive[i] = False
          cause_of_death[i] = "predation"
          history_alive[i, t + 1] = False
          continue

        # 2. Foraging outcome selection
        probs = [o.prob for o in action.outcomes]
        outcome_idx = np.random.choice(len(action.outcomes), p=probs)
        outcome = action.outcomes[outcome_idx]

        # Update state
        next_state = self.chop_state(curr_state, outcome.delta_state)
        pop_states[i] = list(next_state)
        history_states[i, t + 1] = next_state

        if self.is_dead(next_state):
          alive[i] = False
          cause_of_death[i] = "starvation"
          history_alive[i, t + 1] = False
        else:
          history_alive[i, t + 1] = True

    survival_rate = np.mean(history_alive, axis=0)[1:]
    starvation_count = sum(1 for c in cause_of_death if c == "starvation")
    predation_count = sum(1 for c in cause_of_death if c == "predation")

    return {
        "history_states": history_states,
        "history_actions": history_actions,
        "history_alive": history_alive,
        "survival_rate": survival_rate,
        "starvation_rate": starvation_count / num_individuals,
        "predation_rate": predation_count / num_individuals,
        "cause_of_death": cause_of_death,
    }
