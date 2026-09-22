# DSVM tool: web interface

The static web app for building, solving, and simulating dynamic state
variable models. Served by GitHub Pages from this folder:

**https://simondedman.github.io/dsvm-tool/**

## Running it locally

The page is built from ES modules, and browsers refuse to load those from a
`file://` path, so opening `index.html` by double-clicking it will not work.
Use the hosted URL above, or serve this folder:

```
cd docs
python3 -m http.server 8765
```

Then open the address it prints, http://localhost:8765/.

## What is in here

| file | role |
|---|---|
| `index.html` | page structure, the numbered sections 0 to 9 |
| `style.css` | all styling |
| `app.js` | page controller: the model object, storage, toolbar, solve and simulate |
| `ui-model.js` | sections 1 to 5: the model form and the live equation display |
| `ui-results.js` | sections 7 and 8: matrix heatmaps, tables, simulation output |
| `plots.js` | Plotly wrappers and the colourblind-safe palette |
| `engine.js`, `expr.js`, `export.js` | the engine and CSV export; see `../SPEC.md`. Do not edit these from the interface work |
| `examples/` | the three example models, fetched by the Load example menu |
| `tools/crosscheck.mjs` | cross-check against the Python engine |

Only two things are loaded from a CDN: Plotly 2.35.2 from `cdn.plot.ly` for
the plots, and the Spectral and IBM Plex webfonts from Google Fonts. Both
degrade safely: without the fonts the page falls back to system faces, and
without Plotly the tables and CSV downloads still work and the plot areas say
what is missing.

## How the interface uses the engine

The form edits one plain JavaScript object in the shape of `SPEC.md` section
6, and that object is the only source of truth. Nothing is inferred or
corrected silently:

- Expressions are checked as they are typed by constructing an `Expression`,
  and the parser's own message is shown under the field. Solving is still
  allowed, because the engine's error is the authoritative one.
- Solving calls `Model.fromObject` then `model.solve()`. A `ModelError`
  is shown verbatim in a red alert, which names the state and the period
  where the problem occurred.
- Rounding warnings from the solve appear in a yellow alert, listing the
  expressions concerned.
- Editing the form after a solve marks the displayed matrices as out of date
  and locks the simulation until the model is solved again.
- The simulation overlays the exact cohort projection from `model.project`
  on the simulated survival curve, so the two can be compared directly.

The current form is autosaved to `localStorage` on every change and restored
on the next visit. On a first visit with nothing saved, the patch selection
example is loaded so the page is never empty.
