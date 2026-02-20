# MetaInsight Python conversion plan (uv-based scaffold)

## Current status

- `uv` scaffold is in place on branch **`python-port`**.
- Core package scaffolded under `src/metainsight/` with:
  - CLI (`metainsight`)
  - Initial setup helpers (`metainsight/setup.py`)
  - Validation/unit tests for setup-load behaviors.

## Conversion approach (recommended)

### Phase 1 — Foundations (done / in progress)
- [x] Repo tooling: `pyproject.toml`, lockfile, CI-friendly local workflow.
- [x] Stable package structure (`src/metainsight`, CLI, tests, build pipeline).
- [x] Port input/validation layer:
  - `setup_load`
  - `validate_uploaded_data`
  - Basic default-data loading from `inst/extdata/`.
- [ ] Add shared data models + typing (`Domain` layer).

### Phase 2 — Deterministic analysis kernels
- Implement pure functions first (statistically deterministic and testable):
  - Network extraction / graph connectivity
  - Study/arm wrangling (`clean`, `long <-> wide`, treatment ID handling)
  - Frequentist core calculations (`freq_*` outputs)
- Keep these independent of plotting/UI.
- Add unit tests against handcrafted tables + fixtures from `tests/testthat/data`.

### Phase 3 — Bayesian and model interfaces
- Port Bayesian model inputs/outputs from R routines:
  - model specification
  - summary statistics and diagnostics
  - ranking outputs
- If full Bayesian engines are hard in pure-Python, provide equivalent API with
  documented adapter layer (e.g., via external tools/service) first.

### Phase 4 — Plotting + report rendering
- Port plotting functions to `matplotlib/plotly` equivalents.
- Build a report-export layer (SVG/PNG/CSV).
- Keep plot specs deterministic and regression-test against known baselines.

### Phase 5 — App/API parity
- Add lightweight API entry points and/or web layer.
- Add session/state serialization (compatible with Shiny `common` workflow concept).
- Recreate module-level entry points and parameter names.

## Test strategy (important for confidence)

### A. Unit tests in Python first
- Test each ported helper in isolation with small deterministic fixtures.
- Keep tests colocated under `tests/python/`.

### B. Golden-output comparison against R (same inputs)
Yes — this is a strong strategy.

1. Re-use the same data fixtures currently in `tests/testthat/data`.
2. Add a small R oracle runner that executes the original R functions and prints JSON.
3. In Python, call that runner (in CI where R is available) and compare outputs
   with Python results.

Suggested command shape:

```bash
# compare one case
tools/run_r_oracle.py --fn setup_load --path tests/testthat/data/Binary_long.csv --outcome binary
```

And then:
- compare `is_data_valid`, `nrow`, `treatment labels`, and stable transformed fields
  (`Study`, `T`, `N`, `Mean/SD`/`R`) in pandas.
- assert on both success and failure messages where deterministic.

### C. Test migration mapping from existing `tests/testthat`
- Use existing R tests as a “spec source”, starting with:
  - `test-setup_load.R`
  - `test-data_validation.R`
  - `test-setup_configure.R`
- Convert assertions that are purely data/algorithmic first; postpone shiny UI tests.

## Matching outputs across languages

For each migrated function:
1. Define a canonical Python result schema (dataclasses / TypedDicts).
2. Define the expected R payload fields to compare:
   - input-normalized fields
   - validation status/messages
   - deterministic computed summaries
3. Compare with tolerance for numeric arrays (`np.allclose`) and exact match for
   structured labels and text where possible.

## Practical workflow

- Start each feature behind a feature flag:
  - `METAINSIGHT_ENABLE_PY_<module>=0/1` if needed.
- Convert one function/module at a time:
  1. write tests (including R-fixture-backed tests)
  2. implement
  3. run `uv run pytest`
  4. run parity test block (if R available)
- Keep commits focused per module (e.g., `feat: port setup_load`) and tag blockers.

## Repository updates to support parity testing

- Add `scripts/r_oracle.R` (or `scripts/r_oracle.py`) for deterministic payload output.
- Add `tests/python/parity/` with:
  - case manifests
  - pytest markers to skip when `Rscript` is unavailable
  - tolerance configuration per output type

## Notes

- Some R tests are browser/UI (`shinytest2`) and are not portable directly.
  Focus parity comparisons on pure data + computational functions first.
- The `shinyscholar` branch is the best reference for active behavior,
  so continue tracking it as the source of truth for new behavior.
