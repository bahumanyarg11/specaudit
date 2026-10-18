# ulpaudit

*Formerly `specaudit`. Renamed because that name is taken on PyPI by an unrelated package; old links redirect here.*

**How accurate are the special functions you call?**

## Reported upstream

ulpaudit's findings have led to these issues and fix PRs (all hand-verified against mpmath at 50 digits):

| Library | Issue | Fix PR |
|---|---|---|
| PyTorch | [#201178](https://github.com/pytorch/pytorch/issues/201178) `erfinv` float64 on macOS arm64 | [#201324](https://github.com/pytorch/pytorch/pull/201324) Bessel `j0/j1/y0/y1`, `airy_ai` (Cephes denominators) |
| JAX | [#41676](https://github.com/jax-ml/jax/issues/41676) `log_ndtr`, [#41677](https://github.com/jax-ml/jax/issues/41677) `gamma`, [#41678](https://github.com/jax-ml/jax/issues/41678) `erfinv`/`betainc` | [#41679](https://github.com/jax-ml/jax/pull/41679) `log_ndtr` |
| SciPy | [#26423](https://github.com/scipy/scipy/issues/26423) `betaln` NaN for large arguments | [scipy/xsf#322](https://github.com/scipy/xsf/pull/322) |
| mpmath | [#1198](https://github.com/mpmath/mpmath/issues/1198) Hurwitz `zeta` relative accuracy | [#1199](https://github.com/mpmath/mpmath/pull/1199) |

ulpaudit evaluates `scipy.special`, `jax.scipy.special`, `torch.special` and NumPy against mpmath
references carried to 30 significant digits (worst cases re-verified at 50) and reports the error in **units in the last place
(ULP)**, per function, per library, per dtype, over wide parameter ranges including tails, poles and
inverse-function end points. The result is an accuracy atlas you can read, and an assertion you can
put in your own test suite.

```bash
pip install git+https://github.com/bahumanyarg11/ulpaudit   # numpy, scipy, mpmath; jax/torch optional (not on PyPI yet)
ulpaudit list                 # 40 functions, which library provides which
ulpaudit run -n 2000 -o results            # all libraries found, float64
ulpaudit run --float32 -l torch,jax        # the dtype those libraries default to
ulpaudit report results/results.json -o ATLAS.md
```

```python
from ulpaudit import assert_max_ulp
import torch, numpy as np

# fails with the worst inputs listed if any sample is more than 4 ULP off, or returns NaN/inf wrongly
assert_max_ulp(torch.special.erfinv, "erfinv", torch.linspace(-0.999, 0.999, 1001), max_ulp=4)
```

With ulpaudit installed, pytest gets a `ulp` fixture exposing the same helper.

## What the atlas shows

See [ATLAS.md](ATLAS.md) for the current float64 results (SciPy, NumPy, JAX, PyTorch on this machine).
Each cell is *max ULP / 99th-percentile ULP / % of samples worse than 4 ULP*, followed by anomaly
counts: NaN or ±inf where the reference is finite, wrong sign, finite where the reference is not,
and underflow to 0 where the reference is a subnormal.

<!-- HEADLINES -->
### Headline findings (float64, 500 samples per function, this machine; see ATLAS.md for every cell)

Every item below was re-checked by hand against mpmath at 50 digits on the exact worst inputs stored in
`results_f64/results.json` (2026-10-16); issue drafts with reproducers are in [docs/outreach](docs/outreach/).

Library-side, confirmed:

- **PyTorch `bessel_j0/j1/y0/y1` and `airy_ai`** are wrong by up to 8.7e-7 *absolute* just above the
  internal branch point x = 5 (airy: x < −2.09), e.g. `bessel_j0(7.0)` = 0.3000792697 for 0.3000792705,
  and only reach 1e-15 for x ≳ 20; near the zeros this shows as wrong signs (11–21 of 500 samples).
  Cause: the Cephes `p1evl` denominators are evaluated without their implied leading 1 (reproduced
  bit-for-bit). Already reported: pytorch/pytorch#198583.
- **`torch.special.ndtr`** is `0.5·(1 + erf(x/√2))`: relative error 2.3e-6 at x = −7, 2 % at −8, and exactly 0 for
  every x ≤ −8.3744 (true value 2.8e-17 there; SciPy and JAX are correct down to x ≈ −38). Already reported:
  pytorch/pytorch#192295.
- **`torch.special.erfinv`** (float64, eager, this macOS/arm64 build) degrades towards ±1: 2e-12 absolute at
  1 − 2⁻²⁰, 8e-9 at 1 − 2⁻³⁰, 3.5e-4 at 1 − 2⁻⁵² (3.9e11 ULP). The two Newton steps on `erf(x) − y` cannot see the
  residual near 1. SciPy: ≤ 2 ULP everywhere.
- **`jax.scipy.special.log_ndtr`** computes `log(ndtr(x))` for 0 < x ≤ 8 and loses up to 7.7 % relative
  accuracy (x ≈ 7.98); in float32 the same happens for 2 < x ≤ 5 (9 % at x = 4.97). `log1p(-ndtr(-x))`, as
  SciPy and PyTorch do, is within 48 ULP.
- **`jax.scipy.special.gamma`** is `exp(lgamma(x))`, so it carries |ln Γ|·ε relative error: 104 ULP at x = 20
  (19! is exactly representable; SciPy returns it exactly), 2 564 ULP near 171.6; 84 % of samples > 4 ULP.
- **`jax.scipy.special.erfinv`** loses accuracy for 1 − |x| between 1e-12 and 1e-6 (up to 5e5 ULP, 4.4e-10
  absolute); **`betainc`** has p99 ≈ 940 ULP against SciPy's 73, and its result for a given input changes
  with the other elements of the array (69 ULP alone, 3 963 ULP inside the 500-sample batch).
- **`jax.scipy.special.exp1`/`expi`** (float64) hang on any array holding one element ≲ 1e-6 next to another
  (`exp1(jnp.array([1e-20, 2.0]))` never returns; every scalar returns in < 0.5 s). Reported as jax-ml/jax#13543
  and fixed after 0.11.2 by jax-ml/jax#41253; both functions are skipped for JAX in this atlas.
- **`scipy.special.betaln`** returns NaN for 259 of 500 samples over (1e-3, 1e300)², e.g. betaln(1.1e142, 1.7e115)
  (true −1.04e117; JAX returns it), and is off by up to 9e6 ULP (2e-9 relative) for large unequal arguments
  such as betaln(2.2e123, 3.0e117).
- `scipy.special.beta` goes through `exp(lgamma(a)+lgamma(b)−lgamma(a+b))` once a + b > 171.6 and loses
  up to 1 540 ULP (a = 131.3, b = 110.1); p99 ≈ 390 ULP.
- `i0`/`i1` overflow to +inf for 709.78 < x < 713.99 in SciPy, JAX and PyTorch (all Cephes-derived:
  `exp(x)` overflows before `exp(x)/√(2πx)` does); the true value is finite up to x ≈ 713.987.

Shared by all libraries and expected: relative-accuracy loss within ~1e-13 of a function's zeros
(`lgamma` at 1 and 2, `digamma` at 1.4616, `sinc` at integers, `logit` at 0.5, `expi` at 0.3725, SciPy's
Bessel zeros); 100–1 000 ULP in the far tails of `erfc`, `erfcx`, `ndtr`, `log_ndtr`, `gammainc` where SciPy, JAX and
PyTorch share Cephes-derived code.

Reference-side (fixed in this release, the libraries were right): the first atlas showed identical ~9e6 ULP
deviations of all three libraries for Hurwitz `zeta` at s ≳ 40 and "sign errors" for `betaln` at huge arguments.
Both were mpmath/reference problems: `mpmath.zeta(s, q)` stops its Euler–Maclaurin tail on an *absolute*
tolerance and is only ~1e-9 accurate when the value is ~1e-106 (draft: [docs/outreach/mpmath-hurwitz-zeta.md](docs/outreach/mpmath-hurwitz-zeta.md));
and `mp.beta(a, b)` at 30 digits silently computes `a + b == b` when b/a > 1e30. `references.py` now uses its
own relative-tolerance Euler–Maclaurin for zeta and the `lgamma` form at scaled precision for betaln; after
re-running, `zeta` is ≤ 3.3 ULP in all three libraries.
<!-- /HEADLINES -->

## How the error is measured

The reference is computed by mpmath at 30 digits and stored as a double-double `(hi, lo)`, so a
float64 result can be judged to about 1e-3 ULP without carrying arbitrary-precision objects around.
The ULP is that of the reference rounded to the result's dtype; a correctly rounded result is within
0.5 ULP. Float32 results are measured in float32 ULPs. Inputs are sampled log-uniformly over each
function's domain, on both signs where defined, plus dense neighbourhoods of the listed focus points
(poles of Γ, zeros of J0/J1, the end points of erfinv/ndtri/logit, the far tails of erfc/log_ndtr).

Two caveats, stated here because they matter for reading the numbers:

- A large ULP error near a **zero of the function** (lgamma at 1 and 2, sinc at integers) is a loss
  of *relative* accuracy where the absolute error is still ~1e-16. Whether that counts depends on
  your use; glibc's `lgamma` treats those zeros specially, Cephes does not.
- A library that returns 0 where the true value is **subnormal** (expit below −708, erfc beyond 26.5)
  is reported under `uf0`, not as a huge ULP count.

## Functions covered

gamma, lgamma, digamma, polygamma, erf, erfc, erfcx, erfinv, ndtr, ndtri, log_ndtr, expit, logit,
log1p, expm1, i0, i0e, i1, i1e, j0, j1, y0, y1, k0, k1, k0e, k1e, gammainc, gammaincc, betainc,
zeta (Hurwitz), spence, exp1, expi, xlogy, entr, sinc, beta, betaln, airy_ai. Adding one is a reference
function in `references.py` plus a `FunctionSpec` in `functions.py`.

## Install from source

```bash
git clone https://github.com/bahumanyarg11/ulpaudit && cd ulpaudit
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,jax,torch]"
pytest -q
```

## License

BSD 3-Clause.
