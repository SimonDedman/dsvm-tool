# NotebookLM reference code (provenance only)

`dsvm_engine.py` and `app.py` are verbatim copies of the code Google
NotebookLM generated for Mike Heithaus on 2026-09-16 (email "Some AI code n
stuff"), from his prompt and the Mangel & Clark PDFs he uploaded. They are
kept so the lineage of the tool is visible and so the new engine can be
regression-tested against them. They are not used by the tool.

## What the review of that code found (2026-09-21)

Ran unchanged on its own benchmark (x_crit = 3, C = 10, T = 10, patches with
β = 0/0.004/0.020, net gains −1, +2/−1 at 0.4/0.6, +4/−1 at 0.6/0.4):

- Backward recursion is correct for the terminal-fitness, integer-state case
  and matches Mangel & Clark's equation. Decisions at t = 1: x = 4–7 → risky
  patch, x = 8–9 → moderate, x = 10 → safe. The new engine reproduces its
  F(x, t) values exactly.
- Monte Carlo survival for 1,000 individuals, seed 42: 94.6 % (the email
  claims 94.0 %; the difference is random-number ordering). The claim that
  this "perfectly matches published outputs" could not be checked: the book
  does not publish a Monte Carlo result for this model, and NotebookLM did
  not say where the figure came from.
- Limitations the new tool removes: fixed rather than state- or
  time-dependent parameters; every state variable's minimum is lethal (so a
  gut-contents or egg-load variable could not sit at 0); no validation that
  outcome probabilities sum to 1 (a typo silently changes the model); the
  first-action-wins tie rule is implicit; no per-period reward term;
  Streamlit requires a local Python install, which was the step Mike got
  stuck on ("There is no .zip file").
