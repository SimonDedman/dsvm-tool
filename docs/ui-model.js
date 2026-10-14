// Sections 1 to 5: the model form and the live rendering of the equation
// being solved. The plain model object (SPEC.md section 6) is the single
// source of truth; this module reads and writes it and nothing else.

import { Expression, ExprError, isValidIdentifier } from "./expr.js";
import { actionColour } from "./plots.js";

export const FORMAT = "dsvm-model";
export const FORMAT_VERSION = 1;
export const GRID_WARN = 20000;

const PLAIN_NUMBER = /^\s*[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?\s*$/;

export const isPlainNumber = (s) => PLAIN_NUMBER.test(String(s ?? ""));

export function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

export function blankModel() {
  return {
    format: FORMAT,
    version: FORMAT_VERSION,
    name: "",
    description: "",
    horizon: 20,
    state_variables: [
      { name: "x", label: "Energy reserves", min: 0, max: 10, lethal_min: true, initial: 5 },
    ],
    actions: [
      {
        name: "Action 1", mortality: "0", reward: "0",
        outcomes: [{ label: "", prob: "1", change: { x: "-1" } }],
      },
    ],
    terminal_fitness: "1",
    simulation: { individuals: 1000, seed: 42 },
  };
}

/** Fill in optional keys without discarding anything the file already held. */
export function normaliseModel(raw) {
  const m = { ...blankModel(), ...(raw && typeof raw === "object" ? raw : {}) };
  m.format = FORMAT;
  m.version = FORMAT_VERSION;
  m.name = String(m.name ?? "");
  m.description = String(m.description ?? "");
  m.terminal_fitness = String(m.terminal_fitness ?? "1");
  m.state_variables = (Array.isArray(m.state_variables) ? m.state_variables : []).map((v, i) => ({
    ...v,
    name: String(v?.name ?? `x${i + 1}`),
    label: String(v?.label ?? ""),
    min: Number(v?.min ?? 0),
    max: Number(v?.max ?? 10),
    lethal_min: v?.lethal_min === undefined ? true : Boolean(v.lethal_min),
    initial: Number(v?.initial ?? v?.max ?? 10),
  }));
  if (!m.state_variables.length) m.state_variables = blankModel().state_variables;
  m.actions = (Array.isArray(m.actions) ? m.actions : []).map((a, i) => ({
    ...a,
    name: String(a?.name ?? `Action ${i + 1}`),
    mortality: String(a?.mortality ?? "0"),
    reward: String(a?.reward ?? "0"),
    outcomes: (Array.isArray(a?.outcomes) ? a.outcomes : []).map((o) => ({
      ...o,
      label: String(o?.label ?? ""),
      prob: String(o?.prob ?? "1"),
      change: Object.fromEntries(Object.entries(o?.change ?? {}).map(([k, v]) => [k, String(v)])),
    })),
  }));
  if (!m.actions.length) m.actions = blankModel().actions;
  for (const a of m.actions) if (!a.outcomes.length) a.outcomes = [{ label: "", prob: "1", change: {} }];
  const sim = m.simulation && typeof m.simulation === "object" ? m.simulation : {};
  m.simulation = { individuals: Number(sim.individuals ?? 1000), seed: Number(sim.seed ?? 42), ...sim };
  m.simulation.individuals = Number(m.simulation.individuals);
  m.simulation.seed = Number(m.simulation.seed);
  return m;
}

export const varNamesOf = (m) => m.state_variables.map((v) => v.name);

export function gridSize(m) {
  let n = 1;
  for (const v of m.state_variables) {
    const size = Math.trunc(v.max) - Math.trunc(v.min) + 1;
    if (!Number.isFinite(size) || size < 2) return null;
    n *= size;
  }
  return n;
}

// ------------------------------------------------------------- DOM helpers

function el(tag, attrs = {}, children = []) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") n.className = v;
    else if (k === "html") n.innerHTML = v;
    else if (k === "text") n.textContent = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else if (v === true) n.setAttribute(k, "");
    else if (v !== false && v !== null && v !== undefined) n.setAttribute(k, v);
  }
  for (const c of [].concat(children)) if (c) n.appendChild(c);
  return n;
}

function field(labelHTML, control, errEl) {
  const lab = el("label", { class: "field" }, [el("span", { html: labelHTML }), control]);
  if (errEl) lab.appendChild(errEl);
  return lab;
}

function errSlot() { return el("em", { class: "err" }); }

function showErr(input, errEl, message) {
  if (message) {
    input.classList.add("invalid");
    errEl.textContent = message;
    errEl.classList.add("shown");
  } else {
    input.classList.remove("invalid");
    errEl.textContent = "";
    errEl.classList.remove("shown");
  }
}

// ---------------------------------------------------------------- the form

/**
 * Builds sections 1 to 4 from the model object.
 * hooks.structural() is called when the shape of the model changes and the
 * form must be rebuilt; hooks.value() when only a value changed.
 */
export function renderForm(m, hooks) {
  const value = () => hooks.value();
  const structural = () => hooks.structural();

  // ---- section 1
  const nameInput = document.getElementById("modelName");
  const descInput = document.getElementById("modelDesc");
  const horizonInput = document.getElementById("horizon");
  const horizonErr = document.getElementById("horizonErr");
  nameInput.value = m.name;
  descInput.value = m.description;
  horizonInput.value = m.horizon;
  nameInput.oninput = () => { m.name = nameInput.value; value(); };
  descInput.oninput = () => { m.description = descInput.value; value(); };
  horizonInput.oninput = () => {
    const raw = horizonInput.value.trim();
    const v = Number(raw);
    if (raw === "" || !Number.isInteger(v) || v < 2) {
      showErr(horizonInput, horizonErr, "The horizon T must be a whole number of 2 or more.");
    } else {
      showErr(horizonInput, horizonErr, "");
      m.horizon = v;
    }
    value();
  };

  // ---- section 2: state variables
  const varList = document.getElementById("varList");
  varList.replaceChildren();
  m.state_variables.forEach((v, k) => varList.appendChild(variableItem(m, v, k, hooks)));

  // ---- section 3: actions
  const actionList = document.getElementById("actionList");
  actionList.replaceChildren();
  m.actions.forEach((a, i) => actionList.appendChild(actionItem(m, a, i, hooks)));

  // ---- section 4: terminal fitness
  const tf = document.getElementById("terminalFitness");
  const tfErr = document.getElementById("terminalFitnessErr");
  tf.value = m.terminal_fitness;
  tf.oninput = () => {
    m.terminal_fitness = tf.value;
    showErr(tf, tfErr, exprProblem(tf.value, varNamesOf(m)));
    value();
  };
  showErr(tf, tfErr, exprProblem(tf.value, varNamesOf(m)));

  document.getElementById("addVar").onclick = () => {
    const used = new Set(varNamesOf(m));
    let n = 1, name = "y";
    while (used.has(name)) { n += 1; name = `y${n}`; }
    m.state_variables.push({ name, label: "", min: 0, max: 10, lethal_min: true, initial: 5 });
    structural();
  };
  document.getElementById("addAction").onclick = () => {
    m.actions.push({
      name: `Action ${m.actions.length + 1}`, mortality: "0", reward: "0",
      outcomes: [{ label: "", prob: "1", change: {} }],
    });
    structural();
  };

  updateDerived(m);
}

function exprProblem(text, names) {
  try {
    new Expression(text, names);
    return "";
  } catch (e) {
    if (e instanceof ExprError) return e.message;
    return String(e && e.message ? e.message : e);
  }
}

/** A text input holding an expression, validated as it is typed. */
function exprField(labelHTML, getter, setter, m, hooks) {
  const input = el("input", { type: "text", class: "mono", value: getter() });
  const err = errSlot();
  const check = () => showErr(input, err, exprProblem(input.value, varNamesOf(m)));
  input.oninput = () => { setter(input.value); check(); hooks.value(); };
  input.onblur = check;
  check();
  return field(labelHTML, input, err);
}

function numField(labelHTML, getter, setter, hooks, { min, max, integer = true } = {}) {
  const input = el("input", { type: "number", step: integer ? "1" : "any", value: getter() });
  if (min !== undefined) input.min = min;
  if (max !== undefined) input.max = max;
  const err = errSlot();
  input.oninput = () => {
    const raw = input.value.trim();
    const v = Number(raw);
    if (raw === "" || !Number.isFinite(v) || (integer && !Number.isInteger(v))) {
      showErr(input, err, integer ? "Enter a whole number." : "Enter a number.");
    } else {
      showErr(input, err, "");
      setter(v);
    }
    hooks.value();
  };
  return field(labelHTML, input, err);
}

function toolButton(label, title, onclick, disabled = false) {
  return el("button", {
    type: "button", class: "btn small", title, "aria-label": title,
    disabled: disabled || false, onclick,
  }, [document.createTextNode(label)]);
}

// ------------------------------------------------------------- a variable

function variableItem(m, v, k, hooks) {
  const head = el("div", { class: "item-head" }, [
    el("h4", { class: "item-title", html: `State variable <span>${esc(v.name)}</span>` }),
    el("div", { class: "item-tools" }, [
      toolButton("Remove", `Remove state variable ${v.name}`, () => {
        if (m.state_variables.length === 1) {
          window.alert("A model needs at least one state variable, so this one cannot be removed.");
          return;
        }
        const gone = m.state_variables[k].name;
        m.state_variables.splice(k, 1);
        for (const a of m.actions) for (const o of a.outcomes) delete o.change[gone];
        hooks.structural();
      }),
    ]),
  ]);

  const nameInput = el("input", { type: "text", value: v.name });
  const nameErr = errSlot();
  const checkName = () => {
    const proposed = nameInput.value.trim();
    if (!isValidIdentifier(proposed)) {
      showErr(nameInput, nameErr,
        "Use letters, digits, and underscores, not starting with a digit. The names t and T and the function names (min, max, abs, floor, ceil, round, sqrt, exp, log, pow, if) are reserved.");
      return false;
    }
    if (m.state_variables.some((o, i) => i !== k && o.name === proposed)) {
      showErr(nameInput, nameErr, `Another state variable is already called ${proposed}.`);
      return false;
    }
    showErr(nameInput, nameErr, "");
    return true;
  };
  nameInput.oninput = () => { checkName(); hooks.value(); };
  nameInput.onchange = () => {
    const proposed = nameInput.value.trim();
    if (!checkName() || proposed === v.name) return;
    const old = v.name;
    v.name = proposed;
    for (const a of m.actions) {
      for (const o of a.outcomes) {
        if (old in o.change) {
          const rebuilt = {};
          for (const [key, val] of Object.entries(o.change)) rebuilt[key === old ? proposed : key] = val;
          o.change = rebuilt;
        }
      }
    }
    hooks.structural();
  };
  checkName();

  const lethal = el("input", { type: "checkbox" });
  lethal.checked = Boolean(v.lethal_min);
  lethal.onchange = () => { v.lethal_min = lethal.checked; hooks.value(); };

  const body = el("div", {}, [
    el("div", { class: "grid2" }, [
      field("Name, as used in expressions", nameInput, nameErr),
      field("Label for plots and tables (optional)", (() => {
        const i = el("input", { type: "text", value: v.label, placeholder: v.name });
        i.oninput = () => { v.label = i.value; hooks.value(); };
        return i;
      })()),
    ]),
    el("div", { class: "grid3" }, [
      numField("Minimum", () => v.min, (x) => { v.min = x; }, hooks),
      numField("Maximum", () => v.max, (x) => { v.max = x; }, hooks),
      numField("Initial value for simulations", () => v.initial, (x) => { v.initial = x; }, hooks),
    ]),
    el("label", { class: "check" }, [
      lethal,
      el("span", { html: "Individual dies at or below the minimum (the critical threshold <i>x</i><sub>crit</sub>)" }),
    ]),
    el("p", { class: "hint range-note" }),
  ]);
  return el("div", { class: "item", "data-var": k }, [head, body]);
}

// --------------------------------------------------------------- an action

function actionItem(m, a, i, hooks) {
  const names = varNamesOf(m);
  const head = el("div", { class: "item-head" }, [
    el("h4", { class: "item-title", html: `Action ${i + 1} <span>${esc(a.name)}</span>` }),
    el("div", { class: "item-tools" }, [
      toolButton("Up", `Move ${a.name} earlier in the list`, () => {
        [m.actions[i - 1], m.actions[i]] = [m.actions[i], m.actions[i - 1]];
        hooks.structural();
      }, i === 0),
      toolButton("Down", `Move ${a.name} later in the list`, () => {
        [m.actions[i + 1], m.actions[i]] = [m.actions[i], m.actions[i + 1]];
        hooks.structural();
      }, i === m.actions.length - 1),
      toolButton("Remove", `Remove action ${a.name}`, () => {
        if (m.actions.length === 1) {
          window.alert("A model needs at least one action, so this one cannot be removed.");
          return;
        }
        m.actions.splice(i, 1);
        hooks.structural();
      }),
    ]),
  ]);

  const nameInput = el("input", { type: "text", value: a.name });
  nameInput.oninput = () => { a.name = nameInput.value; hooks.value(); };
  nameInput.onchange = () => hooks.structural();

  const advanced = el("details", { class: "advanced" }, [
    el("summary", { text: "Advanced: per-period reward" }),
    exprField(
      "Reward <i>R</i>, the fitness gained in the period this action is chosen",
      () => a.reward, (x) => { a.reward = x; }, m, hooks),
    el("p", {
      class: "hint",
      html: "Leave at <code>0</code> for a terminal-fitness-only model, as in Mangel &amp; Clark chapter 2. Use it for fitness accrued during the season, such as eggs laid or offspring fledged.",
    }),
  ]);

  const outcomes = el("div", { class: "outcomes" });
  a.outcomes.forEach((o, j) => outcomes.appendChild(outcomeItem(m, a, o, i, j, names, hooks)));

  const sumLine = el("span", { class: "probsum", "data-action": i });

  const body = el("div", {}, [
    el("div", { class: "grid2" }, [
      field("Action name", nameInput),
      exprField(
        "Mortality risk <i>&#946;</i>, the probability of dying this period before the outcome is realised, such as predation",
        () => a.mortality, (x) => { a.mortality = x; }, m, hooks),
    ]),
    advanced,
    el("h5", { text: "Outcomes", style: "font:600 13.5px/1.3 var(--sans);margin:12px 0 0;color:var(--ink-soft)" }),
    outcomes,
    el("div", { class: "row-actions" }, [
      el("button", {
        type: "button", class: "btn small",
        onclick: () => { a.outcomes.push({ label: "", prob: "0", change: {} }); hooks.structural(); },
      }, [document.createTextNode("Add outcome")]),
      sumLine,
    ]),
  ]);

  return el("div", { class: "item", "data-action": i }, [head, body]);
}

function outcomeItem(m, a, o, i, j, names, hooks) {
  const labelInput = el("input", { type: "text", value: o.label, placeholder: "e.g. found food" });
  labelInput.oninput = () => { o.label = labelInput.value; hooks.value(); };

  const changes = names.map((n) => {
    const v = m.state_variables.find((s) => s.name === n);
    return exprField(
      `Change in <code>${esc(n)}</code>${v && v.label && v.label !== n ? ` (${esc(v.label)})` : ""}, gain minus cost`,
      () => o.change[n] ?? "0",
      (x) => { if (String(x).trim() === "") delete o.change[n]; else o.change[n] = x; },
      m, hooks);
  });

  const head = el("div", { class: "outcome-head" }, [
    el("h5", { text: `Outcome ${j + 1}` }),
    el("div", { class: "item-tools" }, [
      toolButton("Remove", `Remove outcome ${j + 1} of ${a.name}`, () => {
        if (a.outcomes.length === 1) {
          window.alert(`Action "${a.name}" needs at least one outcome, so this one cannot be removed.`);
          return;
        }
        a.outcomes.splice(j, 1);
        hooks.structural();
      }),
    ]),
  ]);

  return el("div", { class: "outcome" }, [
    head,
    el("div", { class: "grid2" }, [
      field("Label (optional)", labelInput),
      exprField("Probability <i>&#955;</i> of this outcome, given the individual survived the mortality draw",
        () => o.prob, (x) => { o.prob = x; }, m, hooks),
    ]),
    el("div", { class: names.length > 2 ? "grid3" : "grid2" }, changes),
  ]);
}

// -------------------------------------------------- live derived readouts

export function updateDerived(m) {
  // grid size and per-variable range notes
  const sizeEl = document.getElementById("gridSize");
  const warnEl = document.getElementById("gridWarn");
  const n = gridSize(m);
  if (n === null) {
    sizeEl.innerHTML = "Grid size unavailable: every state variable needs a maximum greater than its minimum.";
    warnEl.classList.add("hidden");
  } else {
    const parts = m.state_variables.map((v) => `${esc(v.name)}: ${Math.trunc(v.max) - Math.trunc(v.min) + 1}`);
    sizeEl.innerHTML = `Grid: <strong>${n.toLocaleString("en-GB")}</strong> states (${parts.join(" &times; ")}).`;
    if (n > GRID_WARN) {
      warnEl.innerHTML = `<strong>Large state space</strong>${n.toLocaleString("en-GB")} states is above the 20,000 at which a solve starts to take noticeable time and memory. It will still run; narrow a variable's range if the page becomes slow.`;
      warnEl.classList.remove("hidden");
    } else {
      warnEl.classList.add("hidden");
    }
  }

  document.querySelectorAll("#varList .item").forEach((item, k) => {
    const v = m.state_variables[k];
    const note = item.querySelector(".range-note");
    if (!v || !note) return;
    const lo = Math.trunc(v.min), hi = Math.trunc(v.max);
    if (!(hi > lo)) {
      note.innerHTML = `<span style="color:var(--claret)">The maximum (${hi}) must be greater than the minimum (${lo}).</span>`;
      return;
    }
    const live = v.lethal_min ? `${lo + 1} to ${hi}` : `${lo} to ${hi}`;
    let msg = `${hi - lo + 1} values, ${lo} to ${hi}. Live states: ${live}.`;
    if (v.initial < lo || v.initial > hi) msg += ` The initial value ${v.initial} is outside that range.`;
    else if (v.lethal_min && v.initial <= lo) msg += ` The initial value ${v.initial} is at or below the lethal threshold.`;
    note.textContent = msg;
  });

  document.querySelectorAll("#actionList .item").forEach((item, i) => {
    const a = m.actions[i];
    const slot = item.querySelector(".probsum");
    if (!a || !slot) return;
    const title = item.querySelector(".item-title");
    if (title) title.innerHTML = `Action ${i + 1} <span>${esc(a.name)}</span>`;
    if (a.outcomes.every((o) => isPlainNumber(o.prob))) {
      const sum = a.outcomes.reduce((s, o) => s + Number(o.prob), 0);
      const shown = Number(sum.toPrecision(12));
      slot.textContent = `Outcome probabilities sum to ${shown}${Math.abs(sum - 1) > 1e-9 ? ", not 1" : ""}`;
      slot.classList.toggle("bad", Math.abs(sum - 1) > 1e-9);
      slot.classList.remove("prose");
    } else {
      slot.textContent = "Probabilities depend on the state or time; the solver checks the sum at every state and period.";
      slot.classList.remove("bad");
      slot.classList.add("prose");
    }
  });

  document.querySelectorAll("#varList .item-title").forEach((title, k) => {
    const v = m.state_variables[k];
    if (v) title.innerHTML = `State variable <span>${esc(v.name)}</span>`;
  });

  renderEquation(m);
}

// ----------------------------------------------- section 5: the equation

const iv = (s) => `<i class="v">${esc(s)}</i>`;
const src = (s) => `<span class="src">${esc(String(s).trim() === "" ? "(empty)" : s)}</span>`;

export function renderEquation(m) {
  const box = document.getElementById("equationBox");
  const tables = document.getElementById("equationTables");
  if (!box || !tables) return;
  const names = varNamesOf(m);
  const xs = names.map(iv).join(", ");
  const xsPrime = names.map((n) => `${iv(n)}′`).join(", ");
  const anames = m.actions.map((a) => `<span class="aname">${esc(a.name)}</span>`).join(", ");
  const lethal = m.state_variables.filter((v) => v.lethal_min);

  const lines = [];
  lines.push(`${iv("F")}(${xs}, ${iv("t")}) <span class="op">=</span> max over { ${anames} } of ${iv("V")}<sub>i</sub>(${xs}, ${iv("t")})`);
  lines.push(`${iv("V")}<sub>i</sub>(${xs}, ${iv("t")}) <span class="op">=</span> ${iv("R")}<sub>i</sub> <span class="op">+</span> (1 − ${iv("β")}<sub>i</sub>) <span class="op">·</span> <span class="big">Σ</span><sub>j</sub> ${iv("λ")}<sub>ij</sub> <span class="op">·</span> ${iv("F")}(${xsPrime}<sub>ij</sub>, ${iv("t")}+1)`);
  lines.push(`${iv("F")}(${xs}, ${iv("T")}) <span class="op">=</span> ${iv("φ")}(${xs}) <span class="op">=</span> ${src(m.terminal_fitness)}`);
  if (lethal.length) {
    lines.push(`${iv("F")} <span class="op">=</span> 0 at every ${iv("t")} where ${lethal.map((v) => `${iv(v.name)} ≤ ${Math.trunc(v.min)}`).join(" or ")}`);
  } else {
    lines.push(`<span class="note">No state variable is marked lethal, so no state is a death state and fitness is never forced to zero.</span>`);
  }
  for (const v of m.state_variables) {
    lines.push(`${iv(v.name)}′ <span class="op">=</span> chop( ${iv(v.name)} <span class="op">+</span> ${iv("Δ")}<sub>${esc(v.name)}</sub> ; ${Math.trunc(v.min)}, ${Math.trunc(v.max)} )`);
  }
  lines.push(`<span class="note">Solved backwards from ${iv("t")} = ${Math.trunc(m.horizon) - 1} to ${iv("t")} = 1, with ${iv("T")} = ${Math.trunc(m.horizon)}. ${iv("d")}*(${xs}, ${iv("t")}) is the action reaching the maximum; ties go to the action listed first.</span>`);

  box.innerHTML = lines.map((l) => `<div class="line">${l}</div>`).join("");

  const blocks = m.actions.map((a, i) => {
    const rows = a.outcomes.map((o, j) => `<tr>
      <td class="lft">${esc(o.label || `Outcome ${j + 1}`)}</td>
      <td class="lft">${src(o.prob)}</td>
      ${names.map((n) => `<td class="lft">${src(o.change[n] ?? "0")}</td>`).join("")}
    </tr>`).join("");
    return `<div class="eqblock">
      <h4><span class="eqswatch" style="background:${actionSwatch(i)}"></span>${esc(a.name)}</h4>
      <p class="eqmeta">mortality risk &#946;<sub>${i + 1}</sub> = ${src(a.mortality)} &nbsp;&nbsp; reward R<sub>${i + 1}</sub> = ${src(a.reward)}</p>
      <table class="eqtable">
        <thead><tr><th class="lft">Outcome</th><th class="lft">probability &#955;</th>${names.map((n) => `<th class="lft">change in ${esc(n)}</th>`).join("")}</tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>`;
  }).join("");
  tables.innerHTML = blocks;
}

const actionSwatch = actionColour;
