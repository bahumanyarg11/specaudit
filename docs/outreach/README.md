# Upstream outreach drafts

Issue drafts for findings of the float64 atlas that were verified by hand against mpmath at 50
digits on 2026-10-16 (`specaudit` 0.1.0, Python 3.13.7, macOS 26.3.1 arm64, SciPy 1.18.1,
NumPy 2.5.4, JAX 0.11.2, PyTorch 2.14.1 CPU). Nothing here has been posted; the owner posts.

| draft | target | status |
|---|---|---|
| [jax-log_ndtr.md](jax-log_ndtr.md) | jax-ml/jax | new issue |
| [jax-gamma.md](jax-gamma.md) | jax-ml/jax | new issue |
| [jax-erfinv-betainc.md](jax-erfinv-betainc.md) | jax-ml/jax | new issue (two accuracy items) |
| [jax-exp1-hang.md](jax-exp1-hang.md) | jax-ml/jax | already fixed upstream: reproducer kept for the record, nothing to post |
| [pytorch-erfinv.md](pytorch-erfinv.md) | pytorch/pytorch | new issue (eager, float64, macOS arm64; check on Linux before posting) |
| [pytorch-existing.md](pytorch-existing.md) | pytorch/pytorch | Bessel/Airy and ndtr are already reported (#198583, #192295): optional confirmation comments only |
| [scipy-betaln.md](scipy-betaln.md) | scipy/scipy | new issue |
| [mpmath-hurwitz-zeta.md](mpmath-hurwitz-zeta.md) | mpmath/mpmath | new issue (reference-side bug found while auditing) |
