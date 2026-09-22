// Dynamic State Variable Model builder: page controller.
// Holds one plain model object (SPEC.md section 6) as the source of truth,
// renders the form from it, and hands it to the engine to solve and simulate.

import { Model, ModelError } from "./engine.js";
import { fitnessCSV, decisionCSV, simulationCSV, individualsCSV, modelJSON } from "./export.js";
import { blankModel, normaliseModel, renderForm, updateDerived, esc } from "./ui-model.js";
import { renderMatrixView, renderSimulation, downloadText, alertBox } from "./ui-results.js";

const STORAGE_KEY = "dsvm-tool.model.v1";
const DEFAULT_EXAMPLE = "patch_selection.dsvm.json";
const SIM_WARN = 20000;
const SIM_MAX = 100000;

const state = {
  model: blankModel(),   // plain object, edited by the form
  compiled: null,        // engine Model from the last successful solve
  sol: null,
  sim: null,
  proj: null,
  stale: false,          // the form changed since the last solve
};

const $ = (id) => document.getElementById(id);

// ------------------------------------------------------------- persistence

function save() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state.model));
  } catch (e) {
    note(`The browser refused to save your work locally (${e && e.name ? e.name : "storage error"}). Use Save model to keep a copy.`);
  }
}

function restore() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    return normaliseModel(JSON.parse(raw));
  } catch (e) {
    return null;
  }
}

let noteTimer = null;
function note(text) {
  const el = $("toolbarNote");
  el.textContent = text;
  if (noteTimer) clearTimeout(noteTimer);
  if (text) noteTimer = setTimeout(() => { el.textContent = ""; }, 9000);
}

// ------------------------------------------------------------- re-rendering

const hooks = {
  structural() { rebuild(); },
  value() { updateDerived(state.model); markStale(); save(); },
};

function rebuild() {
  renderForm(state.model, hooks);
  renderSimInitial();
  markStale();
  save();
}

function setModel(m, message) {
  $("exampleSelect").value = "";
  state.model = normaliseModel(m);
  state.compiled = state.sol = state.sim = state.proj = null;
  $("solveResults").classList.add("hidden");
  $("simResults").classList.add("hidden");
  $("solveAlerts").replaceChildren();
  $("simAlerts").replaceChildren();
  $("solveStatus").textContent = "";
  $("simStatus").textContent = "";
  lockSimulation(true);
  renderForm(state.model, hooks);
  renderSimInitial();
  state.stale = false;
  save();
  if (message) note(message);
}

function markStale() {
  if (!state.sol || state.stale) return;
  state.stale = true;
  const box = alertBox("warn", "The model has changed since it was solved",
    "The matrices and simulation below came from the previous version of the model. Solve again to bring them up to date.");
  box.id = "staleAlert";
  const alerts = $("solveAlerts");
  if (!$("staleAlert")) alerts.prepend(box);
  lockSimulation(true);
}

// ------------------------------------------------------- simulation inputs

function renderSimInitial() {
  const wrap = $("simInitial");
  wrap.replaceChildren();
  for (const v of state.model.state_variables) {
    const label = document.createElement("label");
    label.className = "field";
    label.innerHTML = `<span>Initial ${esc(v.name)}${v.label && v.label !== v.name ? ` (${esc(v.label)})` : ""}</span>`;
    const input = document.createElement("input");
    input.type = "number";
    input.step = "1";
    input.value = String(v.initial);
    input.dataset.var = v.name;
    label.appendChild(input);
    wrap.appendChild(label);
  }
}

function initialFromForm() {
  return [...$("simInitial").querySelectorAll("input")].map((i) => Number(i.value));
}

function lockSimulation(locked) {
  $("simControls").disabled = locked;
  $("simLocked").classList.toggle("hidden", !locked);
}

// -------------------------------------------------------------- the solve

function runSolve() {
  const alerts = $("solveAlerts");
  alerts.replaceChildren();
  $("solveResults").classList.add("hidden");
  const btn = $("solveBtn");
  btn.disabled = true;
  $("solveStatus").textContent = "Solving...";

  setTimeout(() => {
    try {
      const model = Model.fromObject(JSON.parse(JSON.stringify(state.model)));
      const t0 = performance.now();
      const sol = model.solve();
      const ms = performance.now() - t0;

      state.compiled = model;
      state.sol = sol;
      state.sim = state.proj = null;
      state.stale = false;

      $("solveStatus").innerHTML =
        `Solved <strong>${model.S.toLocaleString("en-GB")}</strong> states over <strong>${model.T}</strong> periods in <strong>${ms < 1 ? "under 1" : Math.round(ms)}</strong> ms.`;

      if (sol.roundingWarnings.length) {
        alerts.appendChild(alertBox("warn",
          `${sol.roundingWarnings.length} change expression${sol.roundingWarnings.length === 1 ? "" : "s"} produced a non-integer value and ${sol.roundingWarnings.length === 1 ? "was" : "were"} rounded`,
          sol.roundingWarnings));
      }

      // Unhide first: Plotly measures the container, and a display:none
      // parent would give it a width of zero and a fallback-sized plot.
      $("solveResults").classList.remove("hidden");
      renderMatrixView($("decisionView"), model, sol, "D");
      renderMatrixView($("fitnessView"), model, sol, "F");

      $("simN").value = String(state.model.simulation.individuals ?? 1000);
      $("simSeed").value = String(state.model.simulation.seed ?? 42);
      lockSimulation(false);
      $("simResults").classList.add("hidden");
      $("simAlerts").replaceChildren();
      $("simStatus").textContent = "";
    } catch (e) {
      $("solveStatus").textContent = "";
      const isModel = e instanceof ModelError;
      alerts.appendChild(alertBox("error",
        isModel ? "The model could not be solved" : "Something went wrong while solving",
        e && e.message ? e.message : String(e)));
      if (!isModel && e && e.stack) console.error(e);
      lockSimulation(true);
      alerts.scrollIntoView({ behavior: "smooth", block: "center" });
    } finally {
      btn.disabled = false;
    }
  }, 0);
}

// ----------------------------------------------------------- the simulation

function runSimulation() {
  const alerts = $("simAlerts");
  alerts.replaceChildren();
  if (!state.compiled || !state.sol) return;

  const n = Number($("simN").value);
  const seed = Number($("simSeed").value);
  if (!Number.isInteger(n) || n < 1) {
    alerts.appendChild(alertBox("error", "Number of individuals", "Enter a whole number of 1 or more."));
    return;
  }
  if (n > SIM_MAX) {
    alerts.appendChild(alertBox("error", "Number of individuals",
      `${n.toLocaleString("en-GB")} is above the limit of ${SIM_MAX.toLocaleString("en-GB")} this page will run. Reduce it and try again.`));
    return;
  }
  if (!Number.isFinite(seed)) {
    alerts.appendChild(alertBox("error", "Random seed", "Enter a number. The same seed always gives the same run."));
    return;
  }

  const btn = $("simBtn");
  btn.disabled = true;
  $("simStatus").textContent = "Running...";

  setTimeout(() => {
    try {
      const initial = initialFromForm();
      const t0 = performance.now();
      const sim = state.compiled.simulate(state.sol, { n, seed, initial });
      const proj = state.compiled.project(state.sol, { initial });
      const ms = performance.now() - t0;
      state.sim = sim;
      state.proj = proj;
      state.simInitial = initial;

      $("simStatus").innerHTML = `Simulated <strong>${n.toLocaleString("en-GB")}</strong> individuals in <strong>${ms < 1 ? "under 1" : Math.round(ms)}</strong> ms.`;
      $("simResults").classList.remove("hidden");
      renderSimulation(state.compiled, state.sol, sim, proj, initial, {
        headline: $("simHeadline"),
        survival: $("plotSurvival"),
        meanState: $("plotMeanState"),
        behaviour: $("plotBehaviour"),
        traj: $("plotTraj"),
        trajVar: $("trajVar"),
      });

      state.model.simulation = { ...state.model.simulation, individuals: n, seed };
      save();
    } catch (e) {
      $("simStatus").textContent = "";
      alerts.appendChild(alertBox("error",
        e instanceof ModelError ? "The simulation could not run" : "Something went wrong while simulating",
        e && e.message ? e.message : String(e)));
      if (!(e instanceof ModelError)) console.error(e);
      $("simResults").classList.add("hidden");
    } finally {
      btn.disabled = false;
    }
  }, 0);
}

// ----------------------------------------------------------------- toolbar

function fileStem(name) {
  const cleaned = String(name || "model").trim().replace(/[^A-Za-z0-9._ -]+/g, "").replace(/\s+/g, "_");
  return cleaned || "model";
}

async function loadExample(file) {
  try {
    const res = await fetch(`examples/${file}`, { cache: "no-cache" });
    if (!res.ok) throw new Error(`the server returned ${res.status} ${res.statusText}`);
    const data = await res.json();
    setModel(data, `Loaded ${file}.`);
    $("exampleSelect").value = file;
  } catch (e) {
    $("exampleSelect").value = "";
    $("solveAlerts").replaceChildren(alertBox("error", `Could not load the example ${file}`,
      `${e && e.message ? e.message : String(e)}\n\nExamples are fetched over HTTP. If this page was opened from a file:// path, serve the docs folder instead, or use the hosted copy.`));
  }
}

function wireToolbar() {
  $("exampleSelect").onchange = (e) => {
    const file = e.target.value;
    if (file) loadExample(file);
  };

  $("openFile").onchange = (e) => {
    const file = e.target.files && e.target.files[0];
    e.target.value = "";
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        setModel(JSON.parse(String(reader.result)), `Opened ${file.name}.`);
      } catch (err) {
        $("solveAlerts").replaceChildren(alertBox("error", `${file.name} is not a readable model file`,
          `${err && err.message ? err.message : String(err)}\n\nThe file must be JSON in the format described in SPEC.md section 6.`));
      }
    };
    reader.onerror = () => {
      $("solveAlerts").replaceChildren(alertBox("error", `${file.name} could not be read`, String(reader.error)));
    };
    reader.readAsText(file);
  };

  $("saveBtn").onclick = () => {
    try {
      const model = Model.fromObject(JSON.parse(JSON.stringify(state.model)));
      downloadText(`${fileStem(state.model.name)}.dsvm.json`, modelJSON(model), "application/json");
      note("Model saved.");
    } catch (e) {
      // The model does not yet validate. Save exactly what is on the form
      // rather than refusing, so no work is lost, and say so.
      downloadText(`${fileStem(state.model.name)}.dsvm.json`, JSON.stringify(state.model, null, 2) + "\n", "application/json");
      $("solveAlerts").replaceChildren(alertBox("warn", "Saved, but the model is not yet valid",
        `${e && e.message ? e.message : String(e)}\n\nThe file holds exactly what is on the form, so nothing is lost. Fix the problem above and save again.`));
    }
  };

  $("resetBtn").onclick = () => {
    if (!window.confirm("Clear the form and start from a blank model? Your current model will be lost unless you have saved it.")) return;
    setModel(blankModel(), "Reset to a blank model.");
  };

  $("solveBtn").onclick = runSolve;
  $("simBtn").onclick = runSimulation;

  $("dlFitness").onclick = () => downloadText("fitness_F.csv", fitnessCSV(state.compiled, state.sol));
  $("dlDecision").onclick = () => downloadText("decision_D.csv", decisionCSV(state.compiled, state.sol));
  $("dlSimSummary").onclick = () => downloadText("simulation_summary.csv", simulationCSV(state.compiled, state.sim, state.proj));
  $("dlSimIndividuals").onclick = () => downloadText("simulation_individuals.csv", individualsCSV(state.compiled, state.sim));

  const warnN = () => {
    const n = Number($("simN").value);
    const box = $("simWarn");
    if (Number.isFinite(n) && n > SIM_WARN && n <= SIM_MAX) {
      box.innerHTML = `<strong>Large cohort</strong>${n.toLocaleString("en-GB")} individuals will take noticeably longer to run and to plot. The page will not respond while it runs.`;
      box.classList.remove("hidden");
    } else {
      box.classList.add("hidden");
    }
  };
  $("simN").oninput = warnN;
}

// -------------------------------------------------------------- decoration

function drawLattice() {
  const g = document.querySelector(".lattice-dots");
  if (!g) return;
  const on = new Set(["1,3", "2,2", "3,3", "4,1", "5,0", "6,1", "7,0"]);
  const ns = "http://www.w3.org/2000/svg";
  for (let c = 0; c < 8; c++) {
    for (let r = 0; r < 4; r++) {
      const dot = document.createElementNS(ns, "circle");
      dot.setAttribute("cx", String(8 + c * 16.6));
      dot.setAttribute("cy", String(12 + r * 12));
      dot.setAttribute("r", on.has(`${c},${r}`) ? "2.6" : "1.6");
      if (on.has(`${c},${r}`)) dot.setAttribute("class", "on");
      g.appendChild(dot);
    }
  }
}

// -------------------------------------------------------------------- boot

async function boot() {
  drawLattice();
  wireToolbar();
  lockSimulation(true);
  const saved = restore();
  if (saved) {
    setModel(saved, "Restored the model you were last working on.");
    return;
  }
  try {
    const res = await fetch(`examples/${DEFAULT_EXAMPLE}`, { cache: "no-cache" });
    if (!res.ok) throw new Error(`the server returned ${res.status} ${res.statusText}`);
    setModel(await res.json(), "Loaded the patch selection example to start from.");
    $("exampleSelect").value = DEFAULT_EXAMPLE;
  } catch (e) {
    setModel(blankModel(), "");
    $("solveAlerts").replaceChildren(alertBox("warn", "The starting example could not be fetched",
      `${e && e.message ? e.message : String(e)}\n\nA blank model has been loaded instead. Examples are fetched over HTTP: if this page was opened from a file:// path, serve the docs folder with a local server, or use the hosted copy.`));
  }
}

if (typeof document !== "undefined" && document.getElementById("modelName")) {
  boot();
}

export { state };
