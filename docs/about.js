// "What does this mean?" explainer in the masthead.
// Draws the three-step diagram, and on first opening solves the patch
// selection example with the real engine to show its decision matrix.

import { Model } from "./engine.js";
import { actionColour } from "./plots.js";

const NS = "http://www.w3.org/2000/svg";
const EXAMPLE = "examples/patch_selection.dsvm.json";

function el(name, attrs = {}, text) {
  const n = document.createElementNS(NS, name);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, String(v));
  if (text !== undefined) n.textContent = text;
  return n;
}

// Small state x time grid used by all three panels: 6 columns x 5 rows.
const COLS = 6, ROWS = 5, CW = 34, CH = 22, GX = 26;

function grid(g, { end = false, fill = null } = {}) {
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      const isEnd = c === COLS - 1;
      const cell = el("rect", {
        x: GX + c * CW, y: r * CH, width: CW - 3, height: CH - 3, rx: 2,
        class: "ab-cell" + (isEnd && end ? " end" : ""),
      });
      if (fill && !isEnd) {
        cell.setAttribute("style", `fill:${fill(r, c)};stroke:none`);
      }
      g.appendChild(cell);
    }
  }
  g.appendChild(el("text", { x: GX + COLS * CW / 2, y: ROWS * CH + 14, class: "ab-l" }, "time t →"));
  const ly = el("text", { x: 0, y: ROWS * CH / 2, class: "ab-l",
    transform: `rotate(-90 8 ${ROWS * CH / 2})` }, "state x");
  g.appendChild(ly);
  g.appendChild(el("text", { x: GX + (COLS - 1) * CW + (CW - 3) / 2, y: -4, class: "ab-l" }, "T"));
}

function drawSteps() {
  const g1 = document.getElementById("abGrid1");
  if (!g1 || g1.childNodes.length) return;
  const g2 = document.getElementById("abGrid2");
  const g3 = document.getElementById("abGrid3");

  // 1: blank grid, fitness known only in the last column.
  grid(g1, { end: true });
  for (let r = 0; r < ROWS; r++) {
    g1.appendChild(el("text", { x: GX + (COLS - 1) * CW + 15, y: r * CH + 14, class: "ab-l" },
      ["1", ".8", ".6", ".3", "0"][r]));
  }

  // 2: illustrative policy (colours as action 1 to 3), filled back from T.
  const pol = (r, c) => (c >= COLS - 3 && r <= 2) ? 0 : r <= 1 ? 0 : r <= 2 ? 1 : 2;
  grid(g2, { end: true, fill: (r, c) => actionColour(pol(r, c)) + "cc" });
  g2.appendChild(el("path", {
    d: `M${GX + (COLS - 1) * CW} ${ROWS * CH + 24} H${GX + 4}`,
    class: "ab-arrow", "marker-end": "url(#abHead2)",
  }));

  // 3: grid with two simulated individuals; one dies.
  grid(g3, { end: true });
  const cx = (c) => GX + c * CW + 15, cy = (r) => r * CH + 9;
  const lives = [[2, 3, 2, 1, 1, 0], [3, 4, 3, 4]];
  lives.forEach((path, i) => {
    const d = path.map((r, c) => `${c ? "L" : "M"}${cx(c)} ${cy(r)}`).join(" ");
    g3.appendChild(el("path", { d, class: "ab-path", stroke: i ? "#9e1b3c" : "#0e6e78" }));
    const last = path.length - 1;
    if (i) {
      g3.appendChild(el("text", { x: cx(last) + 10, y: cy(path[last]) + 4, class: "ab-l", style: "fill:#9e1b3c" }, "✕"));
    } else {
      g3.appendChild(el("circle", { cx: cx(last), cy: cy(path[last]), r: 3.5, fill: "#0e6e78" }));
    }
  });
  g3.appendChild(el("path", {
    d: `M${GX + 4} ${ROWS * CH + 24} H${GX + (COLS - 1) * CW}`,
    class: "ab-arrow", "marker-end": "url(#abHead2)",
  }));
}

async function drawDecision() {
  const fig = document.getElementById("abDecision");
  if (!fig || fig.dataset.done) return;
  fig.dataset.done = "1";
  try {
    const res = await fetch(EXAMPLE, { cache: "no-cache" });
    if (!res.ok) throw new Error(`the server returned ${res.status}`);
    const spec = await res.json();
    const model = Model.fromObject(JSON.parse(JSON.stringify(spec)));
    const { D } = model.solve();
    const v = spec.state_variables[0];
    const T = spec.horizon;
    const names = spec.actions.map((a) => a.name);

    const cw = 15, ch = 22, left = 34, top = 8;
    const xs = [];
    for (let x = v.max; x >= v.min; x--) xs.push(x);
    const W = left + (T - 1) * cw + 4, H = top + xs.length * ch + 34;
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img",
      "aria-label": "Optimal patch for each energy level and period in the patch selection example" });
    xs.forEach((x, r) => {
      svg.appendChild(el("text", { x: left - 8, y: top + r * ch + 15, class: "ab-l", style: "text-anchor:end" }, String(x)));
      for (let t = 1; t < T; t++) {
        const a = D[model.stateIndex([x])][t];
        svg.appendChild(el("rect", {
          x: left + (t - 1) * cw, y: top + r * ch, width: cw - 1, height: ch - 1,
          fill: a < 0 ? "#dde5e4" : actionColour(a),
        }));
      }
    });
    for (const t of [1, 5, 10, 15, T - 1]) {
      svg.appendChild(el("text", { x: left + (t - 1) * cw + cw / 2, y: top + xs.length * ch + 13, class: "ab-l" }, String(t)));
    }
    svg.appendChild(el("text", { x: left + (T - 1) * cw / 2, y: top + xs.length * ch + 28, class: "ab-l" }, "period t"));
    svg.appendChild(el("text", { x: 8, y: top + xs.length * ch / 2, class: "ab-l",
      transform: `rotate(-90 8 ${top + xs.length * ch / 2})` }, "reserves x"));

    const legend = document.createElement("ul");
    legend.className = "about-legend";
    names.forEach((n, i) => {
      const li = document.createElement("li");
      li.innerHTML = `<span style="background:${actionColour(i)}"></span>`;
      li.append(n);
      legend.appendChild(li);
    });
    const dead = document.createElement("li");
    dead.innerHTML = `<span style="background:#dde5e4"></span>dead (x = ${v.min})`;
    legend.appendChild(dead);
    fig.replaceChildren(svg, legend);
  } catch (e) {
    fig.replaceChildren(Object.assign(document.createElement("p"), {
      className: "about-wait",
      textContent: `The example could not be solved here (${e && e.message ? e.message : e}). Load it from the toolbar and press Solve to see the same table.`,
    }));
  }
}

const about = document.getElementById("about");
if (about) {
  about.addEventListener("toggle", () => {
    if (!about.open) return;
    drawSteps();
    drawDecision();
  });
}
if (about && location.hash === "#about") about.open = true;
