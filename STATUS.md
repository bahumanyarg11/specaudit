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
