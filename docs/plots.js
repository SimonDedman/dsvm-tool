// Plotly wrappers. Every plot here takes data already shaped by ui-results.js;
// nothing in this file reads the model or the DOM form.

const Plotly = () => globalThis.Plotly;

/** Okabe-Ito qualitative palette: distinguishable with any common form of
 *  colour vision deficiency. Reused for actions and state variables. */
export const PALETTE = [
  "#0072b2", "#d55e00", "#009e73", "#cc79a7",
  "#e69f00", "#56b4e9", "#8a6a0b", "#000000",
];

export const actionColour = (i) => PALETTE[i % PALETTE.length];

const FONT = { family: '"IBM Plex Sans", "Segoe UI", Helvetica, Arial, sans-serif', size: 13, color: "#10242e" };

const CONFIG = {
  responsive: true,
  displaylogo: false,
  modeBarButtonsToRemove: ["lasso2d", "select2d"],
  toImageButtonOptions: { format: "png", scale: 2 },
};

function baseLayout(extra = {}) {
  return {
    font: FONT,
    paper_bgcolor: "#ffffff",
    plot_bgcolor: "#ffffff",
    margin: { l: 66, r: 24, t: 14, b: 52 },
    hovermode: "closest",
    legend: { orientation: "h", y: -0.2, x: 0 },
    ...extra,
  };
}

function draw(div, traces, layout) {
  const P = Plotly();
  if (!P) {
    div.innerHTML = '<div class="alert error"><strong>Plotly did not load</strong>' +
      "The plotting library is fetched from cdn.plot.ly. Check the network connection and reload; tables and CSV downloads still work without it.</div>";
    return;
  }
  P.react(div, traces, layout, CONFIG);
}

export function purge(div) {
  const P = Plotly();
  if (P && div && div.data) P.purge(div);
}

/** Discrete-valued heatmap: z holds action indices, null where no decision. */
export function decisionHeatmap(div, { z, x, y, xTitle, yTitle, text, actionNames }) {
  const A = Math.max(actionNames.length, 1);
  const colorscale = [];
  for (let a = 0; a < A; a++) {
    colorscale.push([a / A, actionColour(a)], [(a + 1) / A, actionColour(a)]);
  }
  const trace = {
    type: "heatmap", z, x, y, text,
    colorscale, zmin: -0.5, zmax: A - 0.5, showscale: false,
    xgap: 1, ygap: 1,
    hoverongaps: false,
    hovertemplate: "%{text}<extra></extra>",
  };
  draw(div, [trace], baseLayout({
    xaxis: { title: { text: xTitle }, type: "linear", zeroline: false },
    yaxis: { title: { text: yTitle }, type: "linear", zeroline: false },
    showlegend: false,
  }));
}

/** Continuous heatmap for the fitness matrix. */
export function fitnessHeatmap(div, { z, x, y, xTitle, yTitle, text, colourTitle }) {
  const trace = {
    type: "heatmap", z, x, y, text,
    colorscale: "Viridis", xgap: 1, ygap: 1, hoverongaps: false,
    colorbar: { title: { text: colourTitle, side: "top" }, thickness: 12, len: 0.9 },
    hovertemplate: "%{text}<extra></extra>",
  };
  draw(div, [trace], baseLayout({
    xaxis: { title: { text: xTitle }, type: "linear", zeroline: false },
    yaxis: { title: { text: yTitle }, type: "linear", zeroline: false },
    showlegend: false,
    margin: { l: 66, r: 80, t: 14, b: 52 },
  }));
}

const line = (name, ts, ys, colour, dash) => ({
  type: "scatter", mode: "lines", name, x: ts, y: ys,
  line: { color: colour, width: dash ? 1.6 : 2.2, dash: dash || "solid" },
  hovertemplate: `${name}<br>t = %{x}<br>%{y:.4f}<extra></extra>`,
});

export function survivalPlot(div, { ts, sim, exact }) {
  const traces = [
    line("Alive (simulated)", ts, sim.propAlive, PALETTE[0]),
    line("Alive (exact expectation)", ts, exact.propAlive, PALETTE[0], "dash"),
    line("Dead of mortality risk (simulated)", ts, sim.cumMortality, PALETTE[1]),
    line("Dead of mortality risk (exact expectation)", ts, exact.cumMortality, PALETTE[1], "dash"),
    line("Dead of starvation (simulated)", ts, sim.cumStarvation, PALETTE[2]),
    line("Dead of starvation (exact expectation)", ts, exact.cumStarvation, PALETTE[2], "dash"),
  ];
  draw(div, traces, baseLayout({
    xaxis: { title: { text: "Time period t" }, dtick: 1 },
    yaxis: { title: { text: "Proportion of the cohort" }, range: [0, 1.02] },
    margin: { l: 66, r: 24, t: 14, b: 96 },
  }));
}

export function meanStatePlot(div, { ts, series, yTitle }) {
  const traces = series.map((s, k) => line(s.name, ts, s.y, PALETTE[k % PALETTE.length]));
  draw(div, traces, baseLayout({
    xaxis: { title: { text: "Time period t" }, dtick: 1 },
    yaxis: { title: { text: yTitle } },
    margin: { l: 66, r: 24, t: 14, b: 76 },
  }));
}

export function behaviourPlot(div, { ts, series }) {
  const traces = series.map((s, a) => ({
    type: "scatter", mode: "lines", name: s.name, x: ts, y: s.y,
    stackgroup: "one", line: { width: 0.8, color: actionColour(a) },
    fillcolor: actionColour(a),
    hovertemplate: `${s.name}<br>t = %{x}<br>%{y:.3f} of deciders<extra></extra>`,
  }));
  draw(div, traces, baseLayout({
    xaxis: { title: { text: "Time period t" }, dtick: 1 },
    yaxis: { title: { text: "Proportion of individuals deciding at t" }, range: [0, 1] },
    margin: { l: 66, r: 24, t: 14, b: 76 },
  }));
}

export function trajectoryPlot(div, { lineX, lineY, deathX, deathY, yTitle, nShown, nTotal }) {
  const traces = [{
    type: "scatter", mode: "lines", name: `${nShown} of ${nTotal} individuals`,
    x: lineX, y: lineY, connectgaps: false,
    line: { color: "rgba(61, 85, 95, 0.38)", width: 1 },
    hoverinfo: "skip",
  }];
  if (deathX.length) {
    traces.push({
      type: "scatter", mode: "markers", name: "Died",
      x: deathX, y: deathY,
      marker: { color: "#9e1b3c", size: 7, symbol: "x" },
      hovertemplate: `Died at t = %{x}<br>${yTitle} = %{y}<extra></extra>`,
    });
  }
  draw(div, traces, baseLayout({
    xaxis: { title: { text: "Time period t" }, dtick: 1 },
    yaxis: { title: { text: yTitle } },
    margin: { l: 66, r: 24, t: 14, b: 76 },
  }));
}
