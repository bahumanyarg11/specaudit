# torch.special.erfinv (eager, CPU, float64) is off by 3.5e-4 at x = 1 − 2⁻⁵² and by > 1e-9 for all 1 − x < 1e-7

**Before posting: reproduce on Linux/x86_64.** Measured on macOS 26.3.1 arm64 (Apple libm). pytorch/pytorch#198133
reports that *eager* `erfinv(1 - 1e-16)` is correct on Linux and that only the Inductor path (which
emits `calc_erfinv` from `ATen/native/Math.h`) is wrong there. On this machine the eager float64 result
is bit-identical to a Python re-implementation of `calc_erfinv` (rational guess + two Newton steps on
`erf(x) − y`), so the eager CPU kernel takes the same code path here. If Linux eager is correct the
report should be reframed as "macOS/arm64 eager erfinv uses calc_erfinv and is inaccurate near ±1".

### Description

`torch.special.erfinv` in float64 is within 2 ULP for |x| ≤ 0.999 but degrades steadily towards ±1
and ends 3.5e-4 absolute (6e-5 relative, 3.9e11 ULP) off at the largest double below 1. SciPy's
`erfinv` (Boost) is ≤ 2 ULP everywhere; JAX is ≤ 5e5 ULP.

```python
import torch, mpmath, scipy.special
mpmath.mp.dps = 60
def ulp(got, ref): return abs(got - ref) / (abs(ref) * 2.0**-52)
for k in [14, 20, 24, 30, 40, 52]:
    x = 1 - 2.0**-k
    ref = float(mpmath.erfinv(mpmath.mpf(x)))
    got = torch.special.erfinv(torch.tensor(x, dtype=torch.float64)).item()
    print(f"x = 1 - 2^-{k:<3d} torch={got!r:22} ref={ref!r:22} abs err {got-ref:+.1e}  {ulp(got, ref):10.0f} ULP   scipy {ulp(float(scipy.special.erfinv(x)), ref):.0f} ULP")
```

```
x = 1 - 2^-14  torch=2.8346302855715778     ref=2.834630285571394      abs err +1.8e-13         410 ULP   scipy 1 ULP
x = 1 - 2^-20  torch=3.4655050258053155     ref=3.4655050258033307     abs err +2.0e-12        4470 ULP   scipy 1 ULP
x = 1 - 2^-24  torch=3.8325068569873975     ref=3.832506856900711      abs err +8.7e-11      195000 ULP   scipy 0 ULP
x = 1 - 2^-30  torch=4.328028283825263      ref=4.3280282758009465     abs err +8.0e-09     9030000 ULP   scipy 0 ULP
x = 1 - 2^-40  torch=5.051259863019412      ref=5.05125408524939       abs err +5.8e-06  6510000000 ULP   scipy 0 ULP
x = 1 - 2^-52  torch=5.805365688510647      ref=5.8050186831934533     abs err +3.5e-04 391000000000 ULP   scipy 0 ULP
```

Maximum error by decade of 1 − x on a 20 000-point log grid: [1e-16, 1e-14) 3.6e-4 absolute, [1e-12, 1e-10)
7.5e-6, [1e-10, 1e-8) 8.4e-8, [1e-8, 1e-7) 9.0e-10, [1e-7, 1e-6) 9.8e-11, [1e-5, 1e-4) 1.3e-12. On 500
samples with dense coverage of ±0.999999 (`specaudit run -f erfinv`), 13 % are > 4 ULP off and the 99th
percentile is 2.1e4 ULP. Float32 is fine (≤ 6.8e-6 relative, like JAX).

Cause: `calc_erfinv` polishes a ~1e-9 rational guess with two Newton steps on the residual
`erf(x) − y`. Near ±1 both terms round to within 1e-16 of 1 while the derivative `2/√π·exp(−x²)` is
~1e-15, so the residual carries no information and the guess is returned essentially unpolished.
Boost/SciPy evaluate in terms of `1 − |y|` (exact in floating point) with rational approximations
per segment of `sqrt(-log(1 − |y|))` and need no Newton step.

### Versions

PyTorch 2.14.1 (CPU wheel), Python 3.13.7, macOS 26.3.1 arm64. Reference mpmath 1.3.0 at 60 digits.
