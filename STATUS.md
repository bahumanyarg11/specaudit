# specaudit build status

Resume: read this, `git log --oneline | head`, `PYTHONPATH=. .venv/bin/python -m pytest -q`, continue with the first unchecked item.
Workspace on the Seagate drive (`/Volumes/Seagate Expansion Drive/TOPRESEARCH/specaudit`, symlinked from `09_PROJECTS/specaudit`); venv has numpy/scipy/mpmath/jax(cpu)/torch(cpu). No A6000. Commit as the user only, no AI attribution anywhere.

## Done
- [x] mpmath double-double references for 40 functions (`references.py`), ULP metric + anomaly classes (`metrics.py`)
- [x] Function registry with domains/focus points (`functions.py`), adapters for scipy/numpy/jax/torch (`libs.py`)
- [x] Runner → results.json (`run.py`), Markdown atlas (`report.py`), `assert_max_ulp` + pytest fixture, CLI

## Queue
1. [ ] First full float64 audit (all libs, n=3000) → ATLAS.md; triage findings (real bugs vs documented limits)
2. [ ] float32 audit (torch/jax default dtype) → separate section
3. [ ] Cross-check 3 worst findings per library by hand (mpmath at 50 digits) before claiming anything
4. [ ] Heatmap PNGs over the domain for the ❌ cells (matplotlib)
5. [ ] README with headline numbers; CI; publish; file upstream issues for confirmed bugs (user's account)

## Log
- 2026-10-16 chosen over an IBP reducer (months, no collaboration yield) as the next correctness-audit tool after chemout.
- Smoke run (scipy/numpy/jax, 11 functions, n=400) WORKS but shows astronomically large ULP values for lgamma, log_ndtr, expit, ndtri, zeta. Those are almost certainly bugs in the *metric/reference* (ULP for huge/tiny magnitudes, references underflowing, double-double subtraction when |got-ref| is not small), NOT library findings. Do not report anything until each case is checked by hand against mpmath at 50 digits. Plausible real signals so far: jax gamma (2e3 ULP, 80% > 4 ULP), jax erfinv (1.3e4 ULP), erfc tails (~340 ULP both libs), gammainc (~500 ULP both). 2 unit tests fail (sample_domain bounds; scipy erf 2-ULP assertion) — fix first. torch install may still be running (`uv pip install torch`).
