# DSVM tool: dynamic state variable models for behavioural ecology

**Use it in your browser, nothing to install: <https://simondedman.github.io/dsvm-tool/>**

A dynamic state variable model (DSVM; Mangel & Clark 1988, Clark & Mangel
2000) asks: given an animal's physiological state (energy reserves, gut
contents, egg load, …) and the time left in the season, which behaviour
maximises its expected fitness, when every choice trades foraging gain
against the risk of dying? The answer comes from solving a stochastic dynamic
programming equation backwards from the end of the season, which yields

- the **fitness matrix** F(x, t): expected future fitness of an individual in
  state x at time t behaving optimally from then on, and
- the **decision matrix** d*(x, t): the optimal behaviour for every state and
  time.

Forward Monte Carlo simulation then follows individuals through the season
using d*, to see what the population does and why it dies.

This tool lets you enter the model (state variables, behavioural choices,
their risks, rewards, and outcome probabilities, and the terminal fitness
function), solves the equation, shows both matrices, runs the simulations,
plots everything, and exports CSV files. It was built from Mike Heithaus's
specification (September 2026):

1. Any number of named, discrete-integer state variables.
2. A terminal fitness function (per-period rewards are available but off by default).
3. Individual-based Monte Carlo forward simulation.
4. A graphical interface that prompts for every parameter.
5. Accuracy first: nothing is assumed silently, and every error says where it is.

## Using the web app

1. Open <https://simondedman.github.io/dsvm-tool/>.
2. Pick **Load example → Patch selection (Mangel & Clark 1988)** to see a
   complete model. Every field is labelled with the symbol it corresponds to
   in the book.
3. Edit or add **state variables**: a name (used in formulas), its minimum and
   maximum, whether the minimum is lethal (x_crit), and the starting value for
   simulations.
4. Edit or add **actions**: each has a mortality risk β (predation), and one
   or more **outcomes**, each with a probability λ and the change it makes to
   every state variable (gain minus metabolic cost). Probabilities of an
   action's outcomes must sum to 1.
5. Set the **terminal fitness** φ(x), e.g. `1` for survival or `x / 10` for
   fitness proportional to reserves.
6. Read the **equation being solved**, which is rendered from your inputs.
7. **Validate & solve**. Inspect the decision heatmap and the fitness table.
8. **Run the Monte Carlo simulation**. The survival plot overlays the exact
   expectation (computed by propagating probabilities through the same
   decision matrix) so you can see the simulation converging on the solve.
9. **Save model** writes a `.dsvm.json` file you can reload later, send to a
   colleague, or run from Python or the command line.

Every parameter accepts either a number or a formula in the state variable
names, `t` (current period), and `T` (horizon), e.g. `x < 4 ? 0.03 : 0.015`
for a predation risk that is higher when lean, or `0.6 - 0.1 * g` for a food
encounter probability that falls as the gut fills. The grammar is in
`SPEC.md` §3 and in the app's help panel.

The page runs entirely in your browser: nothing you enter is sent anywhere.
Your current model is kept in the browser's local storage between visits.

## Verification

Accuracy was the first requirement, so the tool checks itself in several
independent ways. All of these run in continuous integration on every commit.

| check | what it establishes |
|---|---|
| Two independent engines (JavaScript in the browser, Python for scripting) written to the same specification must produce **identical** F and D on every example and on adversarial models (ties, time- and state-dependent parameters, rounding, three state variables, non-lethal minima). | No single implementation's bug can go unnoticed. |
| Hand-computed one-step and two-step backward inductions. | The recursion is the one in SPEC §2, not a look-alike. |
| The exact cohort projection (probability mass propagated through d*) must equal F(x₀, 1) to 1e-9. | Solve and forward dynamics are mutually consistent. |
| Monte Carlo survival and mean fitness must fall within the binomial confidence band of the exact projection. | The simulator implements the same model it was solved for. |
| Reimer, Mangel, Derocher & Lewis (2019, *Methods in Ecology and Evolution*) two-patch model: the published optimal policy (risky patch at reserves 1–2, safe patch at 3–5) and the published long-run per-period survival (0.97) are reproduced. | Agreement with an independently published result. |
| Mangel & Clark (1988) ch. 2 patch-selection model: F and d* agree exactly with the independent implementation NotebookLM produced from the book PDFs (`reference/notebooklm/`). | Two independent transcriptions of the book's model agree. The book's own printed table was not available to us in machine-readable form; if you have the book, compare Table 2.1 with `dsvm show examples/patch_selection.dsvm.json` and tell us. |

What the tool deliberately does **not** do (each is an extension point, not a
hidden assumption; see SPEC §8): continuous states with interpolation,
discounting, hidden environmental states, several decision-makers, and
per-action time costs.

## Python package and command line

```
pip install -e python
dsvm validate examples/patch_selection.dsvm.json     # check a model, report problems
dsvm show     examples/patch_selection.dsvm.json     # print d*(x,t) and F(x,1)
dsvm run      examples/patch_selection.dsvm.json -o results/   # solve + simulate + export CSVs
```

```python
from dsvm import load
model = load("examples/patch_selection.dsvm.json")
sol = model.solve()               # sol.F[s, t], sol.D[s, t]; model.states[s] is the state tuple
sim = model.simulate(sol, n=5000, seed=1)
proj = model.project(sol)         # exact expectation for comparison
```

Run the tests with `cd python && pytest` (needs `node` on the path for the
cross-check).

## Repository layout

```
docs/          the web app (GitHub Pages): index.html, app.js, engine.js, expr.js, export.js
docs/examples/ example models served with the app
examples/      the same example models (canonical copies)
python/        Python engine, CLI, and tests (pytest)
reference/     the NotebookLM code this project started from, with a review
SPEC.md        the model format and exact semantics both engines implement
```

## References

- Mangel, M. & Clark, C. W. (1988). *Dynamic Modeling in Behavioral Ecology*. Princeton University Press.
- Clark, C. W. & Mangel, M. (2000). *Dynamic State Variable Models in Ecology: Methods and Applications*. Oxford University Press.
- Houston, A. I. & McNamara, J. M. (1999). *Models of Adaptive Behaviour*. Cambridge University Press.
- Reimer, J. R., Mangel, M., Derocher, A. E. & Lewis, M. A. (2019). Matrix methods for stochastic dynamic programming in ecology and evolutionary biology. *Methods in Ecology and Evolution* 10: 1952–1961. Code: <https://doi.org/10.5281/zenodo.2547815>.
- Marescot, L. et al. (2013). Complex decisions made simple: a primer on stochastic dynamic programming. *Methods in Ecology and Evolution* 4: 872–884.

## Authors

Michael R. Heithaus (concept and specification) and Simon Dedman
(implementation), Florida International University. MIT licence.
