# dsvm (Python)

Python engine, command line, and test suite for the DSVM tool. The web app in
`../docs/` is the primary interface; this package exists so the same model
files can be run in scripts, batch jobs, and continuous-integration tests, and
so the two engines can check each other.

```
pip install -e .            # from this directory
dsvm validate ../examples/patch_selection.dsvm.json
dsvm show     ../examples/patch_selection.dsvm.json
dsvm run      ../examples/patch_selection.dsvm.json -o results/
pytest                      # runs the test suite, including the cross-check against the JS engine if node is installed
```

See `../SPEC.md` for the model format and the exact semantics.
