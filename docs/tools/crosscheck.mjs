// node tools/crosscheck.mjs model.dsvm.json [n] [seed]
// Prints JSON with F, D, projection, and simulation summary for comparison
// with the Python engine (python/tests/test_crosscheck_js.py).
import { readFileSync } from "node:fs";
import { Model } from "../engine.js";

const [path, nArg, seedArg] = process.argv.slice(2);
if (!path) { console.error("usage: node crosscheck.mjs model.json [n] [seed]"); process.exit(2); }
let model, sol, proj, sim;
try {
  model = Model.fromObject(JSON.parse(readFileSync(path, "utf8")));
  sol = model.solve();
  proj = model.project(sol);
  const n = nArg ? Number(nArg) : null, seed = seedArg ? Number(seedArg) : null;
  sim = model.simulate(sol, { n, seed });
} catch (e) {
  console.log(JSON.stringify({ error: e.name + ": " + e.message }));
  process.exit(0);
}
const out = {
  states: model.states,
  F: sol.F.map((row) => Array.from(row).map((v) => (Number.isNaN(v) ? null : v))),
  D: sol.D.map((row) => Array.from(row)),
  roundingWarnings: sol.roundingWarnings,
  projection: { propAlive: Array.from(proj.propAlive).map((v) => (Number.isNaN(v) ? null : v)), expectedFitness: proj.expectedFitness },
  simulation: { n: sim.n, seed: sim.seed, propAliveT: sim.propAlive[model.T], realisedFitness: sim.realisedFitness },
};
console.log(JSON.stringify(out));
