# PyTorch findings that are already on the tracker (no new issue; optional confirmation comments)

## torch.special.bessel_j0 / j1 / y0 / y1 and airy_ai: pytorch/pytorch#198583 (open, 2026-09-25)

Our atlas finding "PyTorch Bessel functions have absolute errors up to ~4e-7 and wrong signs near
their zeros" is this bug. It is not confined to zeros: `bessel_j0(7.0)` = 0.3000792697173623 against
0.30007927051955560 (2.7e-9 relative), and the error is up to 3.8e-7 absolute just above the internal
branch point x = 5 (j0, j1, y0, y1 all affected; y1 worst at 8.7e-7), decaying as (25/x²)^7 so it is
below 1e-15 only for x ≳ 20. `airy_ai` has the same bug for x < −2.09 (3.0e-6 absolute at x = −2.09,
5.3e-9 at −3, 3.9e-14 at −5).

Root cause, confirmed by re-implementing `bessel_j0_forward` in Python: the Cephes `p1evl`
denominators (`QQ` in j0/y0/j1/y1, `RQ`/`YQ` for x ≤ 5, `AFD`/`AGD` in airy_ai) are evaluated with a
Horner loop seeded at 0, dropping the implied leading coefficient 1. Seeding the accumulator with 1
reproduces SciPy/Cephes to the last bit:

```
x=5.0000001  torch=-0.17759711690909064  horner(lead=0)=-0.17759711690909064  horner(lead=1)=-0.17759673855642388  scipy=-0.17759673855642388
x=7.0        torch=0.3000792697173623    horner(lead=0)=0.3000792697173623    horner(lead=1)=0.3000792705195556    scipy=0.3000792705195556
x=10.0       torch=-0.24593576441958856  horner(lead=0)=-0.24593576441958856  horner(lead=1)=-0.24593576445134832  scipy=-0.2459357644513483
```

The issue's second comment already lists y1 and airy_ai and the ten accumulators. Still unfixed on
`main` as of 2026-10-16 (`aten/src/ATen/native/Math.h` identical to v2.14.1 for these functions). A
confirmation comment could add the absolute-error profile by band from our scan (max |torch − scipy|,
float64): j0 [5, 5.5] 3.8e-7, [6, 7] 5.7e-8, [8, 10] 2.0e-9, [10, 15] 3.2e-11, [15, 20] 6.4e-13; y1 [5, 5.5]
8.7e-7; airy_ai [−2.2, −2.09] 3.0e-6, [−3, −2.5] 2.3e-7, [−5, −4] 2.6e-11.

## torch.special.ndtr: pytorch/pytorch#192295 (open, 2026-08-06) "flushes lower-tail float64 values to zero"

Confirmed: `ndtr` is `0.5 * (1 + erf(x / √2))` (`calc_ndtr` in `aten/src/ATen/native/UnaryOps.cpp`), so
the lower tail is lost to cancellation. Relative error 3.9e-11 at x = −5, 2.3e-6 at −7, 1.8 % at −8,
and exactly 0 for every x ≤ −8.374388923067421 (true value there 2.8e-17; still 2.9e-316 at x = −38).
SciPy and JAX return the correct value down to the subnormal range. `0.5 * torch.special.erfc(-x / √2)`
(what SciPy does) is within 12 ULP at x = −5 and 850 ULP at x = −37, the same as SciPy's own tail
error. Our atlas: 22.6 % of 500 samples > 4 ULP, p99 7.9e15 ULP, vs 818 ULP for SciPy/JAX.
