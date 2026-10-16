# Upstream outreach drafts

Issue drafts for findings of the float64 atlas that were verified by hand against mpmath at 50
digits on 2026-10-16 (`specaudit` 0.1.0, Python 3.13.7, macOS 26.3.1 arm64, SciPy 1.18.1,
NumPy 2.5.4, JAX 0.11.2, PyTorch 2.14.1 CPU). Posted on 2026-10-16 from the owner's account; links below.

| draft | target | status |
|---|---|---|
| [jax-log_ndtr.md](jax-log_ndtr.md) | jax-ml/jax | filed as [jax-ml/jax#41676](https://github.com/jax-ml/jax/issues/41676); fix in [jax-ml/jax#41679](https://github.com/jax-ml/jax/pull/41679) |
| [jax-gamma.md](jax-gamma.md) | jax-ml/jax | filed as [jax-ml/jax#41677](https://github.com/jax-ml/jax/issues/41677) |
| [jax-erfinv-betainc.md](jax-erfinv-betainc.md) | jax-ml/jax | filed as [jax-ml/jax#41678](https://github.com/jax-ml/jax/issues/41678) |
| [jax-exp1-hang.md](jax-exp1-hang.md) | jax-ml/jax | already fixed upstream ([jax-ml/jax#41253](https://github.com/jax-ml/jax/pull/41253)); reproducer kept for the record |
| [pytorch-erfinv.md](pytorch-erfinv.md) | pytorch/pytorch | filed as [pytorch/pytorch#201178](https://github.com/pytorch/pytorch/issues/201178) (macOS arm64 only; Linux x86_64 is exact) |
| [pytorch-existing.md](pytorch-existing.md) | pytorch/pytorch | Bessel/Airy: confirmation [comment on #198583](https://github.com/pytorch/pytorch/issues/198583#issuecomment-6214428091); ndtr: already [#192295](https://github.com/pytorch/pytorch/issues/192295) |
| [scipy-betaln.md](scipy-betaln.md) | scipy/scipy | filed as [scipy/scipy#26423](https://github.com/scipy/scipy/issues/26423); NaN fix in [scipy/xsf#322](https://github.com/scipy/xsf/pull/322) |
| [mpmath-hurwitz-zeta.md](mpmath-hurwitz-zeta.md) | mpmath/mpmath | filed as [mpmath/mpmath#1198](https://github.com/mpmath/mpmath/issues/1198) |
