# Verbatim copy of the Streamlit GUI NotebookLM produced for Mike Heithaus
# (email "Some AI code n stuff", 2026-09-16). Kept for provenance only; the
# tool's GUI is the static web app in ../../docs/. See ../README.md.

from dsvm_engine import Action, ActionOutcome, DSVMSolver, StateVariable
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="DSVM Tool - Behavioral Ecology", layout="wide", page_icon="🐾"
)

st.title(
    "🐾 Dynamic State Variable Modeling Tool (Mangel & Clark Framework)"
)
st.markdown(
    "Automated Dynamic Programming & Stochastic Monte Carlo Simulation for"
    " Behavioral Ecology."
)

# Sidebar Configuration
st.sidebar.header("1. Model Setup & State Variables")
num_vars = st.sidebar.number_input(
    "Number of State Variables", min_value=1, max_value=3, value=1, step=1
)

state_vars = []
for v_i in range(int(num_vars)):
  st.sidebar.subheader(f"State Variable {v_i+1}")
  v_name = st.sidebar.text_input(
      f"Variable Name #{v_i+1}",
      value="Energy_Reserves" if v_i == 0 else f"State_{v_i+1}",
  )
  v_min = st.sidebar.number_input(
      f"Critical Threshold (Death if <= x)",
      value=3 if v_i == 0 else 0,
      key=f"min_{v_i}",
  )
  v_max = st.sidebar.number_input(
      f"Capacity Limit (C)", value=10 if v_i == 0 else 10, key=f"max_{v_i}"
  )
  v_init = st.sidebar.number_input(
      f"Initial Value (for Monte Carlo)",
      value=8 if v_i == 0 else 5,
      key=f"init_{v_i}",
  )
  state_vars.append(StateVariable(v_name, v_min, v_max, v_init))

time_horizon = st.sidebar.number_input(
    "Time Horizon (T)", min_value=2, max_value=100, value=10, step=1
)

# Main Area: Define Actions
st.header("2. Define Behavioral Actions & Outcomes")
num_actions = st.number_input(
    "Number of Actions/Choices", min_value=1, max_value=6, value=3, step=1
)

actions = []
preset_names = [
    "Patch 1 (Safe)",
    "Patch 2 (Moderate)",
    "Patch 3 (Risky)",
    "Choice 4",
    "Choice 5",
    "Choice 6",
]
preset_betas = [0.0, 0.004, 0.020, 0.0, 0.0, 0.0]

for a_i in range(int(num_actions)):
  with st.expander(f"Action {a_i+1}: {preset_names[a_i]}", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
      act_name = st.text_input(
          f"Action {a_i+1} Name", value=preset_names[a_i], key=f"aname_{a_i}"
      )
      act_beta = st.number_input(
          f"Predation Risk (beta_{a_i+1})",
          min_value=0.0,
          max_value=1.0,
          value=preset_betas[a_i],
          format="%.4f",
          key=f"abeta_{a_i}",
      )
    with col2:
      num_outcomes = st.number_input(
          f"Number of Outcomes for Action {a_i+1}",
          min_value=1,
          max_value=4,
          value=2 if a_i > 0 else 1,
          key=f"nout_{a_i}",
      )

    outcomes = []
    for o_i in range(int(num_outcomes)):
      st.markdown(f"**Outcome {o_i+1} for {act_name}**")
      ocol1, ocol2 = st.columns(2)
      with ocol1:
        default_prob = (
            1.0
            if a_i == 0
            else (
                0.4
                if o_i == 0 and a_i == 1
                else (0.6 if o_i == 0 else (0.6 if a_i == 1 else 0.4))
            )
        )
        oprob = st.number_input(
            f"Probability (lambda)",
            min_value=0.0,
            max_value=1.0,
            value=default_prob,
            key=f"oprob_{a_i}_{o_i}",
        )

      delta_dict = {}
      with ocol2:
        for v in state_vars:
          default_delta = (
              -1
              if a_i == 0
              else (
                  +2
                  if o_i == 0 and a_i == 1
                  else (+4 if o_i == 0 and a_i == 2 else -1)
              )
          )
          delta_val = st.number_input(
              f"Delta {v.name}",
              value=default_delta,
              key=f"odelta_{a_i}_{o_i}_{v.name}",
          )
          delta_dict[v.name] = delta_val

      outcomes.append(ActionOutcome(oprob, delta_dict))

    actions.append(Action(act_name, act_beta, outcomes))

# Run Solver Button
st.header("3. Backward Dynamic Programming Solution")

if st.button("🚀 Solve Dynamic Programming Equations", type="primary"):
  solver = DSVMSolver(state_vars, time_horizon, actions)
  F_matrix, D_matrix = solver.solve_backward()
  st.session_state["solver"] = solver
  st.success("Backward Dynamic Programming recursion complete!")

if "solver" in st.session_state:
  solver = st.session_state["solver"]
  T = solver.time_horizon

  st.subheader("Optimal Decision Matrix & Fitness Values")

  rows = []
  for s_idx, state_tuple in enumerate(solver.all_states):
    row_dict = dict(zip(solver.var_names, state_tuple))
    for t in range(1, T + 1):
      row_dict[f"F(t={t})"] = round(solver.F[s_idx, t], 4)
      row_dict[f"Choice(t={t})"] = actions[solver.D[s_idx, t]].name
    rows.append(row_dict)

  df_res = pd.DataFrame(rows)
  st.dataframe(df_res, use_container_width=True)

  if len(state_vars) == 1:
    var_name = state_vars[0].name
    state_vals = [s[0] for s in solver.all_states]
    dec_grid = solver.D[:, 1 : T + 1]

    fig = px.imshow(
        dec_grid,
        x=[f"t={t}" for t in range(1, T + 1)],
        y=state_vals,
        labels=dict(x="Time Step (t)", y=var_name, color="Action Index"),
        title=f"Optimal Decision Matrix d*({var_name}, t)",
        color_continuous_scale="Viridis",
    )
    fig.update_layout(yaxis=dict(tickmode="linear"))
    st.plotly_chart(fig, use_container_width=True)

  st.header("4. Forward Individual-Based Monte Carlo Simulation")
  num_sim = st.number_input(
      "Number of Simulated Organisms",
      min_value=100,
      max_value=10000,
      value=1000,
      step=100,
  )

  if st.button("🎲 Run Monte Carlo Simulation"):
    sim_res = solver.run_monte_carlo(num_individuals=int(num_sim))

    col_s1, col_s2, col_s3 = st.columns(3)
    col_s1.metric(
        "Final Survival Rate", f"{sim_res['survival_rate'][-1]*100:.1f}%"
    )
    col_s2.metric(
        "Starvation Mortality", f"{sim_res['starvation_rate']*100:.1f}%"
    )
    col_s3.metric(
        "Predation Mortality", f"{sim_res['predation_rate']*100:.1f}%"
    )

    t_steps = list(range(1, T + 1))
    fig_surv = px.line(
        x=t_steps,
        y=sim_res["survival_rate"],
        labels={"x": "Time Step (t)", "y": "Proportion Alive"},
        title="Population Survival Trajectory Over Time",
    )
    st.plotly_chart(fig_surv, use_container_width=True)
