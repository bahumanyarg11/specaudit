# specaudit build status

Resume: read this, `git log --oneline | head`, `PYTHONPATH=. .venv/bin/python -m pytest -q`, continue with the first unchecked item.
Workspace on the Seagate drive (`/Volumes/Seagate Expansion Drive/TOPRESEARCH/specaudit`, symlinked from `09_PROJECTS/specaudit`); venv has numpy/scipy/mpmath/jax(cpu)/torch(cpu). No A6000. Commit as the user only, no AI attribution anywhere.

## Done
- [x] mpmath double-double references for 40 functions (`references.py`), ULP metric + anomaly classes (`metrics.py`)
- [x] Function registry with domains/focus points (`functions.py`), adapters for scipy/numpy/jax/torch (`libs.py`)
- [x] Runner → results.json (`run.py`), Markdown atlas (`report.py`), `assert_max_ulp` + pytest fixture, CLI

## Queue
1. [x] First full float64 audit (all libs, n=500) → ATLAS.md; headline triage in README
2. [ ] float32 audit (torch/jax default dtype) → separate section
3. [x] Cross-check every headline finding by hand (mpmath at 50 digits) on the stored worst inputs; issue drafts in docs/outreach/
4. [ ] Heatmap PNGs over the domain for the ❌ cells (matplotlib)
5. [x] README with headline numbers; CI; publish. [ ] Post the drafts in docs/outreach/ (user's account): jax log_ndtr, jax gamma, jax erfinv+betainc, pytorch erfinv (re-check on Linux first), scipy betaln, mpmath zeta; PyTorch Bessel/Airy (#198583) and ndtr (#192295) are already filed, JAX exp1 hang fixed (#41253)

## Log
- 2026-10-16 chosen over an IBP reducer (months, no collaboration yield) as the next correctness-audit tool after chemout.
- Smoke run (scipy/numpy/jax, 11 functions, n=400) WORKS but shows astronomically large ULP values for lgamma, log_ndtr, expit, ndtri, zeta. Those are almost certainly bugs in the *metric/reference* (ULP for huge/tiny magnitudes, references underflowing, double-double subtraction when |got-ref| is not small), NOT library findings. Do not report anything until each case is checked by hand against mpmath at 50 digits. Plausible real signals so far: jax gamma (2e3 ULP, 80% > 4 ULP), jax erfinv (1.3e4 ULP), erfc tails (~340 ULP both libs), gammainc (~500 ULP both). 2 unit tests fail (sample_domain bounds; scipy erf 2-ULP assertion) — fix first. torch install may still be running (`uv pip install torch`).
- Full run done (n=500). Reference fixes along the way: ndtri precision scaling, log_ndtr via log1p, erfcx asymptotic for x>1e4, polygamma domain capped at 1e6 (mpmath hangs beyond), Bessel domains capped at 1e5. JAX exp1/expi hang -> skipped. Published to GitHub.
- 2026-10-16 (verification pass) Every headline claim re-checked at 50 digits on the exact worst inputs. Two were *reference* bugs, now fixed in references.py: mpmath's Hurwitz zeta uses an absolute E-M tolerance (own E-M with relative stop, validated to 1e-37 vs mp.zeta at 30+|log10 v|+30 digits); mp.beta at 30 digits loses a+b when b/a > 1e30 (lgamma form at scaled precision). zeta and betaln re-run (`run -f zeta,betaln --resume`; sampling is now seeded per function so partial re-runs reproduce). Root causes found: torch Bessel/Airy = Cephes p1evl leading 1 dropped (bit-exact re-implementation; already pytorch#198583); torch ndtr = 0.5(1+erf) (pytorch#192295); torch erfinv Newton on erf(x)-y cannot see the residual near 1 (3.5e-4 abs at 1-2^-52, macOS build; pytorch#198133 says Linux eager is fine -> verify before posting); jax log_ndtr = log(ndtr(x)) for 0<x<=8 (7.7% rel err at 7.98; float32 9% at 4.97); jax gamma = exp(lgamma); jax betainc result depends on batch contents (69 vs 3963 ULP), Lentz loop keeps iterating converged lanes; jax exp1/expi hang = jnp.piecewise evaluating the x>1 continued fraction on tiny x under batching (fixed upstream in #41253, also float32); SciPy betaln NaN for 259/500 samples (lbeta_asymp overflow, min arg > ~1e80 and ratio > 1e6). Unverified/follow-ups: jax zeta 2.7e9 ULP at (249.3, 16.8) where the value is 3e-306 (near-subnormal handling, one sample); jax betainc batch dependence root cause not pinned (a 2-element pair did not reproduce it, the 25 worst inputs inside any 500-batch do); SciPy beta 1540 ULP via lgamma route is real but not drafted.
