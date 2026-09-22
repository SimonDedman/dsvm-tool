"""Command-line interface.

    python -m dsvm validate model.dsvm.json
    python -m dsvm solve    model.dsvm.json -o results/
    python -m dsvm run      model.dsvm.json -o results/ [-n 1000] [--seed 42] [--individuals]
    python -m dsvm show     model.dsvm.json            # print D and F(t=1) to the terminal
"""

from __future__ import annotations

import argparse
import sys
import time

from . import __version__
from .engine import CAUSE_NAMES, ModelError
from .expr import ExprError
from .io import export_all, load


def _load(path: str):
    try:
        return load(path)
    except FileNotFoundError:
        sys.exit(f"error: file not found: {path}")
    except (ModelError, ExprError, ValueError) as e:
        sys.exit(f"error: invalid model: {e}")


def _print_table(model, sol) -> None:
    T = model.T
    names = model.var_names
    w = max(12, max(len(a.name) for a in model.actions) + 1)
    print("Optimal decision d*(x, t) and fitness F(x, 1):")
    print("  " + " ".join(f"{n:>6}" for n in names) + f" {'F(x,1)':>10}  "
          + " ".join(f"{'t=' + str(t):>{w}}" for t in range(1, T)))
    for s_idx, state in enumerate(model.states):
        cells = []
        for t in range(1, T):
            a = int(sol.D[s_idx, t])
            cells.append(f"{(model.actions[a].name if a >= 0 else '—'):>{w}}")
        f1 = sol.F[s_idx, 1]
        print("  " + " ".join(f"{x:>6}" for x in state) + f" {f1:>10.6f}  " + " ".join(cells))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="dsvm", description="Dynamic state variable model solver and simulator")
    ap.add_argument("--version", action="version", version=f"dsvm {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("validate", help="check a model file and report problems")
    p.add_argument("model")

    p = sub.add_parser("solve", help="solve the dynamic programming equation and export F and D")
    p.add_argument("model")
    p.add_argument("-o", "--out", required=True, help="output directory")

    p = sub.add_parser("run", help="solve, simulate, and export everything")
    p.add_argument("model")
    p.add_argument("-o", "--out", required=True, help="output directory")
    p.add_argument("-n", "--individuals", dest="n", type=int, default=None, help="number of simulated individuals")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--per-individual", action="store_true", help="also export one row per individual per period")

    p = sub.add_parser("show", help="print the decision matrix and F(x,1) to the terminal")
    p.add_argument("model")

    args = ap.parse_args(argv)
    model = _load(args.model)

    if args.cmd == "validate":
        try:
            model.table  # evaluates every expression at every state and time
        except ModelError as e:
            sys.exit(f"invalid: {e}")
        print(f"ok: {model.name or args.model}: {len(model.state_variables)} state variable(s), "
              f"{model.S} states, {len(model.actions)} action(s), T={model.T}")
        for wmsg in model.table["rounding"]:
            print(f"warning: {wmsg}")
        return 0

    t0 = time.time()
    try:
        sol = model.solve()
    except ModelError as e:
        sys.exit(f"error: {e}")
    for wmsg in sol.rounding_warnings:
        print(f"warning: {wmsg}", file=sys.stderr)
    print(f"solved {model.S} states × T={model.T} in {time.time() - t0:.2f}s")

    if args.cmd == "show":
        _print_table(model, sol)
        return 0

    sim = proj = None
    if args.cmd == "run":
        t0 = time.time()
        sim = model.simulate(sol, n=args.n, seed=args.seed)
        proj = model.project(sol)
        print(f"simulated {sim.n} individuals (seed {sim.seed}) in {time.time() - t0:.2f}s")
        init_idx = model.state_index[model.initial_state]
        print(f"  F(initial, 1)            = {sol.F[init_idx, 1]:.6f}")
        print(f"  exact expected fitness   = {proj.expected_fitness:.6f}")
        print(f"  simulated mean fitness   = {sim.realised_fitness:.6f}")
        print(f"  survival to T: simulated {sim.prop_alive[model.T]:.4f}, exact {proj.prop_alive[model.T]:.4f}")
        counts = {CAUSE_NAMES[c]: int((sim.cause == c).sum()) for c in CAUSE_NAMES}
        print(f"  outcomes: {counts}")
    files = export_all(model, sol, args.out, sim=sim, proj=proj,
                       individuals=getattr(args, "per_individual", False))
    print("wrote:")
    for f in files:
        print("  " + f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
