# specaudit

**How accurate are the special functions you call?**

specaudit evaluates `scipy.special`, `jax.scipy.special`, `torch.special` and NumPy against mpmath
references carried to 30 significant digits and reports the error in **units in the last place
(ULP)**, per function, per library, per dtype, over wide parameter ranges including tails, poles and
inverse-function end points. The result is an accuracy atlas you can read, and an assertion you can
put in your own test suite.

```bash
pip install specaudit          # numpy, scipy, mpmath; jax/torch optional
specaudit list                 # 40 functions, which library provides which
specaudit run -n 2000 -o results            # all libraries found, float64
specaudit run --float32 -l torch,jax        # the dtype those libraries default to
specaudit report results/results.json -o ATLAS.md
```

```python
from specaudit import assert_max_ulp
import torch, numpy as np

# fails with the worst inputs listed if any sample is more than 4 ULP off, or returns NaN/inf wrongly
assert_max_ulp(torch.special.erfinv, "erfinv", torch.linspace(-0.999, 0.999, 1001), max_ulp=4)
```

With specaudit installed, pytest gets a `ulp` fixture exposing the same helper.

## What the atlas shows

See [ATLAS.md](ATLAS.md) for the current float64 results (SciPy, NumPy, JAX, PyTorch on this machine).
Each cell is *max ULP / 99th-percentile ULP / % of samples worse than 4 ULP*, followed by anomaly
counts: NaN or ±inf where the reference is finite, wrong sign, finite where the reference is not,
and underflow to 0 where the reference is a subnormal.

<!-- HEADLINES -->
### Headline findings (float64, 500 samples per function, this machine; see ATLAS.md for every cell)

Library-side, not explained by zeros or tails:

- **PyTorch Bessel functions `bessel_j0/j1/y0/y1`** have *absolute* errors up to ~4e-7 near their zeros and
  return the wrong sign in 11–21 of 500 samples (e.g. `bessel_y1(5.4296810407942)` = −4.2e-7, true −2.8e-14).
  SciPy's absolute error at the same points is ~4e-17.
- **`torch.special.ndtr`** returns exactly 0 at x ≈ −9.95 where the true value is 1.3e-23; SciPy and JAX are
  correct down to the subnormal range (x ≈ −38).
- **`jax.scipy.special.gamma`**: 84 % of samples are more than 4 ULP off (99th percentile ≈ 1400 ULP);
  SciPy's `gamma` stays within 3 ULP on the same inputs.
- **`erfinv` in JAX and PyTorch** lose accuracy as |x| → 1 (99th percentile 7.5e3 and 2.1e4 ULP; SciPy: 1.6 ULP max).
- **`jax.scipy.special.exp1` and `expi` (float64) did not return** within 15 minutes on 500 inputs spanning
  1e-300…700 and were skipped; everything else in this atlas ran in milliseconds.
- `jax.scipy.special.betainc`: 57 % of samples > 4 ULP (p99 ≈ 940 ULP) against SciPy's 14 % / 73 ULP;
  `jax.scipy.special.log_ndtr` p99 ≈ 9e5 ULP against ~700 for SciPy and PyTorch.
- `i0`/`i1` overflow to +inf for 700 < x < 709.7 in all three libraries although the true value is still finite.

Shared by all libraries and expected: relative-accuracy loss within ~1e-13 of a function's zeros
(`lgamma` at 1 and 2, `digamma` at 1.4616, `sinc` at integers, `logit` at 0.5, `expi` at 0.3725, SciPy's
Bessel zeros); 100–350 ULP in the far tails of `erfc`, `erfcx`, `ndtr`, `gammainc` where SciPy, JAX and PyTorch
share Cephes-derived code.

Not yet verified by hand (the reference may be at fault, treat as open questions): identical ~9e6 ULP
deviations of all three libraries for Hurwitz `zeta` at s ≳ 40; SciPy `betaln` NaN at huge arguments;
SciPy `beta` p99 ≈ 390 ULP; PyTorch `airy_ai`.
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
git clone https://github.com/bahumanyarg11/specaudit && cd specaudit
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,jax,torch]"
pytest -q
```

## License

BSD 3-Clause.
