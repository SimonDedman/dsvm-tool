// Sections 7 and 8: rendering the solved matrices and the simulation output.
// Everything here reads the engine's own objects; nothing is recomputed.

import { CAUSE_MORTALITY, CAUSE_STARVATION } from "./engine.js";
import * as plots from "./plots.js";
import { esc } from "./ui-model.js";

const MAX_TABLE_CELLS = 50000;
const MAX_TRAJECTORIES = 50;

const labelOf = (v) => (v.label && v.label !== v.name ? `${v.name} (${v.label})` : v.name);

function el(tag, attrs = {}, children = []) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") n.className = v;
    else if (k === "html") n.innerHTML = v;
    else if (k === "text") n.textContent = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else if (v !== false && v !== null && v !== undefined) n.setAttribute(k, v);
  }
  for (const c of [].concat(children)) if (c) n.appendChild(c);
  return n;
}

function selectOf(options, selected, onchange) {
  const s = el("select", { onchange: (e) => onchange(e.target.value) });
  for (const o of options) {
    const opt = el("option", { value: String(o.value), text: o.label });
    if (String(o.value) === String(selected)) opt.selected = true;
    s.appendChild(opt);
  }
  return s;
}

const inlineField = (label, control) =>
  el("label", { class: "field inline" }, [el("span", { text: label }), control]);

export function downloadText(filename, text, mime = "text/csv;charset=utf-8") {
  const blob = new Blob([text], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = el("a", { href: url, download: filename });
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 4000);
}

// -------------------------------------------------- matrix views (7b, 7c)

/**
 * Builds the heatmap plus its axis controls and the table underneath.
 * kind is "D" for the decision matrix or "F" for the fitness matrix.
 */
export function renderMatrixView(container, model, sol, kind) {
  container.replaceChildren();
  const N = model.stateVariables.length;
  const T = model.T;

  const view = {
    yVar: 0,
    xVar: N > 1 ? 1 : -1,   // -1 means time is on the horizontal axis
    t: Math.min(1, T),
    fixed: model.stateVariables.map((v) => v.initial),
  };

  const controls = el("div", { class: "view-controls" });
  const plotDiv = el("div", { class: "plot" });
  const swatches = el("div", { class: "swatches" });
  const tableWrap = el("div");

  function rebuildControls() {
    controls.replaceChildren();
    if (N === 1) {
      controls.appendChild(el("span", {
        class: "metric",
        html: `Rows are <strong>${esc(model.stateVariables[0].name)}</strong> in ascending order, columns are time <i>t</i>.`,
      }));
      return;
    }
    const varOpts = model.stateVariables.map((v, k) => ({ value: k, label: labelOf(v) }));
    controls.appendChild(inlineField("Vertical axis", selectOf(varOpts, view.yVar, (v) => {
      const k = Number(v);
      if (k === view.xVar) view.xVar = view.yVar;
      view.yVar = k;
      rebuildControls(); redraw();
    })));
    controls.appendChild(inlineField("Horizontal axis", selectOf(varOpts, view.xVar, (v) => {
      const k = Number(v);
      if (k === view.yVar) view.yVar = view.xVar;
      view.xVar = k;
      rebuildControls(); redraw();
    })));

    const tMax = kind === "D" ? T - 1 : T;
    const tRange = el("input", { type: "range", min: 1, max: tMax, step: 1, value: Math.min(view.t, tMax) });
    const tOut = el("output", { class: "metric", text: String(Math.min(view.t, tMax)) });
    view.t = Math.min(view.t, tMax);
    tRange.oninput = () => { view.t = Number(tRange.value); tOut.textContent = tRange.value; redraw(); };
    controls.appendChild(inlineField("Time t", el("span", {}, [tRange, tOut])));

    for (let k = 0; k < N; k++) {
      if (k === view.yVar || k === view.xVar) continue;
      const v = model.stateVariables[k];
      const opts = [];
      for (let x = v.min; x <= v.max; x++) opts.push({ value: x, label: String(x) });
      controls.appendChild(inlineField(`${v.name} fixed at`, selectOf(opts, view.fixed[k], (val) => {
        view.fixed[k] = Number(val);
        redraw();
      })));
    }
  }

  function cellState(yv, xv) {
    const st = view.fixed.slice();
    st[view.yVar] = yv;
    if (view.xVar >= 0) st[view.xVar] = xv;
    return st;
  }

  function redraw() {
    const yVarObj = model.stateVariables[view.yVar];
    const yVals = [];
    for (let x = yVarObj.min; x <= yVarObj.max; x++) yVals.push(x);

    let xVals, xTitle, timeOnX;
    if (view.xVar < 0) {
      timeOnX = true;
      xVals = Array.from({ length: kind === "D" ? T - 1 : T }, (_, i) => i + 1);
      xTitle = "Time period t";
    } else {
      timeOnX = false;
      const xv = model.stateVariables[view.xVar];
      xVals = [];
      for (let x = xv.min; x <= xv.max; x++) xVals.push(x);
      xTitle = labelOf(xv);
    }

    const z = [], text = [];
    for (const yv of yVals) {
      const zr = [], tr = [];
      for (const xv of xVals) {
        const t = timeOnX ? xv : view.t;
        const st = cellState(yv, timeOnX ? null : xv);
        const s = model.stateIndex(st);
        const where = model.varNames.map((n, k) => `${esc(n)} = ${st[k]}`).join(", ") + `, t = ${t}`;
        if (s < 0 || model.dead[s]) {
          zr.push(null);
          tr.push(`${where}<br>dead state`);
        } else if (kind === "D") {
          const a = sol.D[s][t];
          zr.push(a >= 0 ? a : null);
          tr.push(`${where}<br>${a >= 0 ? esc(model.actions[a].name) : "no decision at t = T"}`);
        } else {
          const f = sol.F[s][t];
          zr.push(Number.isFinite(f) ? f : null);
          tr.push(`${where}<br>F = ${Number.isFinite(f) ? f.toFixed(6) : "not defined"}`);
        }
      }
      z.push(zr); text.push(tr);
    }

    const yTitle = labelOf(yVarObj);
    if (kind === "D") {
      plots.decisionHeatmap(plotDiv, {
        z, x: xVals, y: yVals, xTitle, yTitle, text,
        actionNames: model.actions.map((a) => a.name),
      });
      swatches.innerHTML = model.actions.map((a, i) =>
        `<span><i style="background:${plots.actionColour(i)}"></i>${esc(a.name)}</span>`).join("") +
        '<span class="plain">Blank cells are dead states, or t = T where no decision is taken.</span>';
    } else {
      plots.fitnessHeatmap(plotDiv, { z, x: xVals, y: yVals, xTitle, yTitle, text, colourTitle: "F" });
      swatches.innerHTML = '<span class="plain">Blank cells are dead states, where F = 0 by definition.</span>';
    }
  }

  rebuildControls();
  container.append(controls, plotDiv, swatches, tableWrap);
  redraw();
  tableWrap.appendChild(matrixTable(model, sol, kind));
  return { redraw };
}

function matrixTable(model, sol, kind) {
  const T = model.T, S = model.S;
  const frag = document.createDocumentFragment();
  const cells = S * T;
  let rowLimit = S;
  if (cells > MAX_TABLE_CELLS) {
    rowLimit = Math.max(1, Math.floor(MAX_TABLE_CELLS / T));
    frag.appendChild(el("div", {
      class: "alert warn",
      html: `<strong>Table truncated</strong>This model has ${S.toLocaleString("en-GB")} states and ${T} periods, which is ${cells.toLocaleString("en-GB")} cells: too many to draw. The first ${rowLimit.toLocaleString("en-GB")} states are shown. Download the CSV below for the whole matrix.`,
    }));
  }

  const table = el("table");
  const thead = el("thead");
  const hrow = el("tr");
  for (const n of model.varNames) hrow.appendChild(el("th", { text: n }));
  for (let t = 1; t <= T; t++) hrow.appendChild(el("th", { text: `t = ${t}` }));
  thead.appendChild(hrow);
  table.appendChild(thead);

  const tbody = el("tbody");
  for (let s = 0; s < rowLimit; s++) {
    const row = el("tr");
    model.states[s].forEach((x, k) => row.appendChild(el(k === 0 ? "th" : "td", { class: "num", text: String(x) })));
    const isDead = model.dead[s] === 1;
    for (let t = 1; t <= T; t++) {
      if (isDead) { row.appendChild(el("td", { class: "dead", text: "" })); continue; }
      if (kind === "D") {
        const a = sol.D[s][t];
        row.appendChild(el("td", { text: a >= 0 ? model.actions[a].name : "" }));
      } else {
        const f = sol.F[s][t];
        row.appendChild(el("td", { class: "num", text: Number.isFinite(f) ? f.toFixed(4) : "" }));
      }
    }
    tbody.appendChild(row);
  }
  table.appendChild(tbody);
  frag.appendChild(el("div", { class: "table-scroll" }, [table]));
  return frag;
}

// ------------------------------------------------- simulation output (8)

const pct = (v) => `${(v * 100).toFixed(1)}%`;
const f4 = (v) => (Number.isFinite(v) ? v.toFixed(4) : "not defined");

export function renderSimulation(model, sol, sim, proj, initial, els) {
  const T = model.T;
  const initIdx = model.stateIndex(initial);
  const fInit = initIdx >= 0 ? sol.F[initIdx][1] : NaN;

  let nMort = 0, nStarve = 0;
  for (let i = 0; i < sim.n; i++) {
    if (sim.cause[i] === CAUSE_MORTALITY) nMort++;
    else if (sim.cause[i] === CAUSE_STARVATION) nStarve++;
  }

  const stateText = model.varNames.map((n, k) => `${n} = ${initial[k]}`).join(", ");
  const items = [
    ["Fitness from the solve, F(initial state, t = 1)", f4(fInit), `Starting at ${stateText}`],
    ["Expected fitness, exact projection", f4(proj.expectedFitness), "Markov propagation of the cohort under the optimal policy"],
    ["Mean fitness, simulated", f4(sim.realisedFitness), `${sim.n.toLocaleString("en-GB")} individuals, seed ${sim.seed}`],
    ["Survival to t = T, simulated", pct(sim.propAlive[T]), `${Math.round(sim.propAlive[T] * sim.n).toLocaleString("en-GB")} of ${sim.n.toLocaleString("en-GB")} alive at t = ${T}`],
    ["Survival to t = T, exact", pct(proj.propAlive[T]), "From the same projection"],
    ["Died of mortality risk", nMort.toLocaleString("en-GB"), `${pct(nMort / sim.n)} of the cohort`],
    ["Died of starvation", nStarve.toLocaleString("en-GB"), `${pct(nStarve / sim.n)} of the cohort`],
  ];
  els.headline.innerHTML = items.map(([k, v, note]) =>
    `<div><dt>${esc(k)}</dt><dd>${esc(v)}<small>${esc(note)}</small></dd></div>`).join("");

  const ts = Array.from({ length: T }, (_, i) => i + 1);
  const pick = (arr) => ts.map((t) => (Number.isFinite(arr[t]) ? arr[t] : null));

  plots.survivalPlot(els.survival, {
    ts,
    sim: { propAlive: pick(sim.propAlive), cumMortality: pick(sim.cumMortality), cumStarvation: pick(sim.cumStarvation) },
    exact: { propAlive: pick(proj.propAlive), cumMortality: pick(proj.cumMortality), cumStarvation: pick(proj.cumStarvation) },
  });

  plots.meanStatePlot(els.meanState, {
    ts,
    series: model.stateVariables.map((v, k) => ({
      name: labelOf(v),
      y: ts.map((t) => (Number.isFinite(sim.meanState[t][k]) ? sim.meanState[t][k] : null)),
    })),
    yTitle: "Mean value among individuals alive at t",
  });

  plots.behaviourPlot(els.behaviour, {
    ts,
    series: model.actions.map((a, i) => ({
      name: a.name,
      y: ts.map((t) => (Number.isFinite(sim.actionFreq[t][i]) ? sim.actionFreq[t][i] : 0)),
    })),
  });

  // trajectory variable chooser
  const sel = els.trajVar;
  const keep = sel.value;
  sel.replaceChildren();
  model.stateVariables.forEach((v, k) => sel.appendChild(el("option", { value: k, text: labelOf(v) })));
  sel.value = keep && Number(keep) < model.stateVariables.length ? keep : "0";
  const drawTraj = () => trajectories(model, sim, Number(sel.value), els.traj);
  sel.onchange = drawTraj;
  drawTraj();
}

/** Deterministic sample of individuals, so the same run always shows the same lines. */
function sampleIndices(n, k, seed) {
  const idx = Array.from({ length: n }, (_, i) => i);
  let s = (seed >>> 0) || 1;
  const rnd = () => {
    s ^= s << 13; s >>>= 0;
    s ^= s >>> 17;
    s ^= s << 5; s >>>= 0;
    return s / 4294967296;
  };
  for (let i = n - 1; i > 0 && i >= n - k; i--) {
    const j = Math.floor(rnd() * (i + 1));
    [idx[i], idx[j]] = [idx[j], idx[i]];
  }
  return idx.slice(Math.max(0, n - k)).sort((a, b) => a - b);
}

function trajectories(model, sim, varIndex, div) {
  const T = model.T;
  const chosen = sampleIndices(sim.n, Math.min(MAX_TRAJECTORIES, sim.n), sim.seed ^ 0x5bf03635);
  const lineX = [], lineY = [], deathX = [], deathY = [];
  for (const i of chosen) {
    for (let t = 1; t <= T; t++) {
      if (!sim.isAlive(i, t)) continue;
      lineX.push(t);
      lineY.push(sim.stateOf(i, t)[varIndex]);
    }
    lineX.push(null); lineY.push(null);
    const dt = sim.deathTime[i];
    if (dt >= 0 && sim.isAlive(i, dt)) {
      deathX.push(dt);
      deathY.push(sim.stateOf(i, dt)[varIndex]);
    }
  }
  plots.trajectoryPlot(div, {
    lineX, lineY, deathX, deathY,
    yTitle: labelOf(model.stateVariables[varIndex]),
    nShown: chosen.length, nTotal: sim.n,
  });
}

export function alertBox(kind, title, detail) {
  const box = el("div", { class: `alert ${kind}` });
  box.appendChild(el("strong", { text: title }));
  if (Array.isArray(detail)) {
    const ul = el("ul");
    for (const d of detail) ul.appendChild(el("li", { text: d }));
    box.appendChild(ul);
  } else if (detail) {
    box.appendChild(el("div", { class: "detail", text: detail }));
  }
  return box;
}
