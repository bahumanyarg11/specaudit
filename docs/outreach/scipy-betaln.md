# scipy.special.betaln returns NaN for large arguments with ratio > 1e6, and is only ~1e-9 accurate for large unequal arguments

### Describe your issue

`betaln(a, b)` is NaN whenever min(a, b) ≳ 1e80 and max/min ≳ 1e6, although log B(a, b) is a perfectly
ordinary negative number there (JAX returns it; mpmath via `loggamma` at a precision where a + b is exact
confirms it). For large unequal arguments below that threshold the result carries a relative error of
~1e-9, far above the ~1e-16 the function achieves elsewhere.

```python
import numpy as np, scipy.special as sp, mpmath
def ref(a, b):  # lgamma form at a precision where a + b is exact
    with mpmath.workdps(50 + int(mpmath.log10(max(a, b) / min(a, b))) + 10):
        return float(mpmath.loggamma(a) + mpmath.loggamma(b) - mpmath.loggamma(mpmath.mpf(a) + b))
for a, b in [(1e100, 1e107), (1.0967934945056546e+142, 1.6614036713282665e+115), (2.383656518478368e+159, 7.822100164232287e+255),
             (2.2350235285263674e+123, 3.029877315654731e+117), (1.724473080349204e+106, 7.006411699535731e+100), (1e10, 1e20)]:
    got, want = sp.betaln(a, b), ref(a, b)
    print(f"betaln({a:.6g}, {b:.6g}) = {got!r:26} reference {want!r:26} rel err {abs(got - want) / abs(want):.1e}")
```

```
betaln(1e+100, 1e+107) = nan                        reference -1.7118095700958324e+101  rel err nan
betaln(1.09679e+142, 1.6614e+115) = nan             reference -1.0426059922188929e+117  rel err nan
betaln(2.38366e+159, 7.8221e+255) = nan             reference -5.321191058459192e+161   rel err nan
betaln(2.23502e+123, 3.02988e+117) = -4.3967279428272305e+118  reference -4.396727951061843e+118   rel err 1.9e-09
betaln(1.72447e+106, 7.00641e+100) = -9.398126308683334e+101   reference -9.39812631468069e+101    rel err 6.4e-10
betaln(1e+10, 1e+20) = -240258509310.49857          reference -240258509310.49857       rel err 0.0e+00
```

Grid (rows a, columns b/a = 1, 10, 1e3, 1e5, 1e6, 1e7, 1e10, 1e20, 1e50): every entry is finite for
a ≤ 1e50; for a = 1e100 … 1e200 the entries with b/a ≥ 1e7 are all NaN and the ones with b/a ≤ 1e6 are
finite. On 500 (a, b) pairs log-uniform in (1e-3, 1e300)² `betaln` is NaN for 259 of them and more than
4 ULP off for 44 of the rest (p99 2e5 ULP). `jax.scipy.special.betaln` on the same inputs: no NaN, p99 58 ULP.

Likely cause: `lbeta_asymp` in Cephes `beta.c` (used when the ratio exceeds `ASYMP_FACTOR` = 1e6) evaluates
`b*(1-b)/(2*a)` and `b*b*(1-b)*(1-b)/(12*a*a*a)` as written, so `b*(1-b)` overflows to inf for b ≳ 1.3e154,
and `a*a*a` overflows to inf for a ≳ 5.6e102, giving inf − inf or inf/inf = NaN; the sub-leading
Stirling terms themselves are negligible at these sizes. Rewriting them as `(b/a)*(1-b)/2` etc. (or just
dropping them when a > 1e30) removes the NaN; the 1e-9 accuracy loss below the threshold is the
`lgam(a) + lgam(b) - lgam(a+b)` path cancelling, which `lbeta_asymp`-style expansions in b/a would fix.

### SciPy/NumPy/Python version and system information

SciPy 1.18.1, NumPy 2.5.4, Python 3.13.7, macOS 26.3.1 arm64. Reference: mpmath 1.3.0.
