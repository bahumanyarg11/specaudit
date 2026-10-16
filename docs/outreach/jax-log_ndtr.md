# jax.scipy.special.log_ndtr loses up to 8 % relative accuracy for 5 < x ≤ 8 (float64) and 2 < x ≤ 5 (float32)

### Description

For positive `x` below `upper_segment` (8 in float64, 5 in float32), `log_ndtr` is computed as
`log(ndtr(x))`. There `ndtr(x) = 1 - δ` with δ down to 1e-16 (float64) or 1e-7 (float32), so the
subtraction inside `ndtr` throws away most of δ before the log is taken and the result is only
correct to a few digits. SciPy and PyTorch compute `log1p(-ndtr(-x))` in that range and are
correct to ~1e-15.

```python
import jax, jax.numpy as jnp
from jax.scipy.special import log_ndtr, ndtr
import mpmath
jax.config.update("jax_enable_x64", True)
mpmath.mp.dps = 50

for x in [5.0, 6.0, 7.0, 7.9, 7.98]:
    ref = float(mpmath.log1p(-mpmath.ncdf(-mpmath.mpf(x))))
    got = float(log_ndtr(jnp.float64(x)))
    alt = float(jnp.log1p(-ndtr(jnp.float64(-x))))
    print(f"x={x}: log_ndtr={got:.17g}  reference={ref:.17g}  rel err {abs(got-ref)/abs(ref):.1e}  "
          f"log1p(-ndtr(-x)) rel err {abs(alt-ref)/abs(ref):.1e}")
```

```
x=5.0: log_ndtr=-2.866516130081049e-07  reference=-2.8665161296376358e-07  rel err 1.5e-10  log1p(-ndtr(-x)) rel err 1.7e-15
x=6.0: log_ndtr=-9.8658770091115712e-10  reference=-9.865876455243758e-10  rel err 5.6e-08  log1p(-ndtr(-x)) rel err 4.0e-15
x=7.0: log_ndtr=-1.2798651027886996e-12  reference=-1.2798125438866541e-12  rel err 4.1e-05  log1p(-ndtr(-x)) rel err 1.6e-16
x=7.9: log_ndtr=-1.4432899320127045e-15  reference=-1.3945171466592653e-15  rel err 3.5e-02  log1p(-ndtr(-x)) rel err 2.7e-15
x=7.98: log_ndtr=-7.7715611723760987e-16  reference=-7.316664535724895e-16  rel err 6.2e-02  log1p(-ndtr(-x)) rel err 6.5e-15
```

On a 30 000-point grid over (5, 8] the maximum relative error is 7.7e-2 (at x ≈ 7.98); SciPy and
PyTorch agree with each other to 2.5e-15 over the same grid. In float32 (the default dtype) the same
thing happens for 2 < x ≤ 5: at `x = float32(4.974)` the relative error is 9.1e-2, and at x = 5.0 it is
4.0e-2, where SciPy's float32 result is within 1e-9.

Measured with `specaudit` (`pip install specaudit`): on 500 float64 samples `jax.scipy.special.log_ndtr`
has a 99th-percentile error of 9.2e5 ULP against ~700 ULP for SciPy and PyTorch (those 700 ULP are in
the far tail x ≈ 30 and shared by all three).

### Expected behaviour

Relative error at the level of a few ULP, as for x ≤ 0 and x > 8.

### Suggested fix

In `jax/_src/scipy/special.py::log_ndtr`, for `lower_segment < x` use
`lax.log1p(-_ndtr(-x))` instead of `lax.log(_ndtr(x))` (SciPy's `log_ndtr` does exactly this for
0 < x ≤ 6, PyTorch for all x > 0). On the grid above this gives ≤ 48 ULP everywhere in float64 and
≤ 1.7e-6 relative in float32, with no change to the series branch for x < lower_segment. The
`upper_segment` branch (`-ndtr(-x)` for x > 8) can stay.

### System info

jax 0.11.2, jaxlib 0.11.2 (CPU), Python 3.13.7, macOS 26.3.1 arm64. Reference: mpmath 1.3.0 at 50 digits.
