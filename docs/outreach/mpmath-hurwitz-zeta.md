# zeta(s, a) loses relative accuracy when the value is tiny: the Euler-Maclaurin tail stops on an absolute tolerance

### Description

For real s > 1 and a > 1, `mpmath.zeta(s, a)` (Hurwitz zeta, `_hurwitz_em`) is accurate only in an
*absolute* sense: when the result is much smaller than 2^-prec the Euler–Maclaurin correction terms
are truncated too early and the relative error is large. mpmath otherwise promises relative accuracy
at the working precision.

```python
import mpmath as mp
s, q = mp.mpf('43.01303985093054'), mp.mpf('305.4243263250737')
with mp.workdps(60):   # direct sum (terms fall off like (q/(q+k))^43) plus the integral tail: converged to all digits shown
    direct = mp.fsum((q + k)**(-s) for k in range(20000)) + (q + 20000)**(1 - s) / (s - 1)
print(mp.nstr(direct, 25))
for dps in [15, 30, 60, 90, 120, 150]:
    with mp.workdps(dps):
        z = mp.zeta(s, q)
    print(dps, mp.nstr(z, 25), f"relative error {float(abs(z - direct) / direct):.1e}")
```

```
1.018131113260128483346694e-106
15 1.018131138740301780410155e-106 relative error 2.5e-08
30 1.018131116078344174333486e-106 relative error 2.8e-09
60 1.018131113296291920822832e-106 relative error 3.6e-11
90 1.018131113260885323206319e-106 relative error 7.4e-13
120 1.018131113260128483346694e-106 relative error 5.5e-25
150 1.018131113260128483346694e-106 relative error 7.6e-54
```

The value is ≈ 2^-352; the result becomes correct once prec exceeds ~352 bits (dps ≈ 106), i.e. once
the absolute tolerance 2^-prec is below the value. Other examples at dps = 30: zeta(20.994, 160.78)
(value 4e-46) is off by 2.9e-9 relative; zeta(56.07, 1214.5) (value 2.5e-172) by 1.1e-9. SciPy, JAX and
PyTorch (Cephes-derived Euler–Maclaurin in double precision, evaluated at the float64 roundings of s and
q) all agree with the corresponding direct sum to 16 digits, which is how this was noticed: an accuracy audit of those libraries against mpmath flagged
"9e6 ULP" errors that turned out to be the reference.

### Expected

Relative accuracy at the working precision, as for s near 1 or a near 1. The stopping test in
`_hurwitz_em` should compare the Euler–Maclaurin term against the running total (relative), not
against 2^-prec.

mpmath 1.3.0, Python 3.13.7.
