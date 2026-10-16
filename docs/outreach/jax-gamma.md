# jax.scipy.special.gamma is exp(lgamma) and loses up to ~2 500 ULP (float64); exact integer values are wrong

### Description

`gamma` is implemented as `gammasgn(x) * exp(lgamma(x))`. The absolute rounding error of `lgamma(x)`,
about ½ ULP of a number of size |ln Γ(x)|, becomes a *relative* error of `exp`, so the result is off
by roughly |ln Γ(x)| / ln 2 ULP: ~100 ULP at x = 20, ~1 400 ULP near x = 170. SciPy's `gamma` (a
direct Stirling/rational evaluation) stays within 3 ULP over the same inputs.

```python
import jax, jax.numpy as jnp, mpmath, scipy.special
jax.config.update("jax_enable_x64", True)
mpmath.mp.dps = 50
def ulp(got, ref): return abs(got - ref) / (abs(ref) * 2.0**-52)

for x in [5.0, 10.0, 20.0, 50.25, 170.0, 171.6000020007015, -50.21846812810949]:
    ref = float(mpmath.gamma(mpmath.mpf(x)))
    got = float(jax.scipy.special.gamma(jnp.float64(x)))
    print(f"x={x:<20} jax={got!r:26} ref={ref!r:26} jax {ulp(got, ref):7.0f} ULP   scipy {ulp(float(scipy.special.gamma(x)), ref):4.0f} ULP")
```

```
x=5.0                  jax=24.000000000000004        ref=24.0                       jax       4 ULP   scipy    0 ULP
x=10.0                 jax=362880.00000000006        ref=362880.0                   jax      18 ULP   scipy    0 ULP
x=20.0                 jax=1.2164510040883034e+17    ref=1.21645100408832e+17       jax     104 ULP   scipy    0 ULP
x=50.25                jax=1.614476471241216e+63     ref=1.6144764712412441e+63     jax     154 ULP   scipy    1 ULP
x=170.0                jax=...                       ref=...                        jax    1364 ULP   scipy    2 ULP
x=171.6000020007015    jax=1.585913225626989e+308    ref=1.5859132256275007e+308    jax    2564 ULP   scipy    1 ULP
x=-50.21846812810949   jax=-6.916145190916293e-65    ref=-6.916145190915793e-65     jax     593 ULP   scipy    2 ULP
```

`Γ(5) = 24`, `Γ(10) = 362880` and `Γ(20) = 19!` are exactly representable in float64 and SciPy returns
them exactly; JAX returns the neighbours. On 500 log-uniform float64 samples over (1e-300, 180) plus
dense neighbourhoods of the poles and of 171.6 (`specaudit run -f gamma`), 84 % of JAX's results are
more than 4 ULP off and the 99th percentile is 1 366 ULP; SciPy's 99th percentile is 2.5 ULP.

### Expected behaviour

A few ULP, as SciPy. Possible implementations: the Lanczos approximation JAX already uses for complex
inputs (`_complex_gamma`), applied to real inputs; or Stirling's series with the reflection formula
for x < 0.5, as Cephes/SciPy do. `exp(lgamma)` cannot be made accurate for |x| ≳ 10 because the
error is inherent to representing ln Γ in float64.

### System info

jax 0.11.2 (CPU), Python 3.13.7, macOS 26.3.1 arm64. Reference: mpmath 1.3.0 at 50 digits; SciPy 1.18.1.
