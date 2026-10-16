# torch.special.erfinv float64 on macOS arm64 (CPU) is off by 3.5e-4 at x = 1 − 2⁻⁵² and by > 1e-9 for all 1 − x < 1e-7; the Linux x86_64 wheel is exact

### Description

On the macOS arm64 CPU wheel, `torch.special.erfinv` in float64 is within 2 ULP for |x| ≤ 0.999 but
degrades steadily towards ±1. The same script on Linux x86_64 (torch 2.14.1+cu130, glibc 2.39, CPU
tensor) returns the correctly rounded value at every point below (0 ULP), so this is specific to the
macOS build (Apple libm / the arm64 kernel), not to the algorithm as such.

It ends 3.5e-4 absolute (6e-5 relative, 2.7e11 ULP) off at the largest double below 1. SciPy's
`erfinv` (Boost) is ≤ 2 ULP everywhere on the same machine.

```python
import torch, mpmath, numpy as np, scipy.special
mpmath.mp.dps = 60
def ulp(got, ref): return abs(got - ref) / np.spacing(abs(ref))  # ULP of the reference
for k in [14, 20, 24, 30, 40, 52]:
    x = 1 - 2.0**-k
    ref = float(mpmath.erfinv(mpmath.mpf(x)))
    got = torch.special.erfinv(torch.tensor(x, dtype=torch.float64)).item()
    print(f"x = 1 - 2^-{k:<3d} torch={got!r:22} ref={ref!r:22} abs err {got-ref:+.1e}  {ulp(got, ref):10.0f} ULP   scipy {ulp(float(scipy.special.erfinv(x)), ref):.0f} ULP")
```

```
x = 1 - 2^-14  torch=2.834630285571578      ref=2.8346302855713943     abs err +1.8e-13         414 ULP   scipy 1 ULP
x = 1 - 2^-20  torch=3.4655050258053155     ref=3.465505025803331      abs err +2.0e-12        4469 ULP   scipy 1 ULP
x = 1 - 2^-24  torch=3.8325068569873615     ref=3.832506856900711      abs err +8.7e-11      195120 ULP   scipy 0 ULP
x = 1 - 2^-30  torch=4.328028283825263      ref=4.328028275800946      abs err +8.0e-09     9034577 ULP   scipy 0 ULP
x = 1 - 2^-40  torch=5.051259863019412      ref=5.05125408524939       abs err +5.8e-06  6505190730 ULP   scipy 0 ULP
x = 1 - 2^-52  torch=5.805365688510647      ref=5.805018683193453      abs err +3.5e-04  390693254302 ULP   scipy 0 ULP
```

Maximum error by decade of 1 − x on a 20 000-point log grid: [1e-16, 1e-14) 3.6e-4 absolute, [1e-12, 1e-10)
7.5e-6, [1e-10, 1e-8) 8.4e-8, [1e-8, 1e-7) 9.0e-10, [1e-7, 1e-6) 9.8e-11, [1e-5, 1e-4) 1.3e-12. On 500
samples with dense coverage of ±0.999999 (`specaudit run -f erfinv`), 13 % are > 4 ULP off and the 99th
percentile is 2.1e4 ULP. Float32 is fine (≤ 6.8e-6 relative, like JAX).

Likely cause: the macOS result is bit-identical to a Python re-implementation of `calc_erfinv`
(`ATen/native/Math.h`: rational guess plus two Newton steps on `erf(x) − y`) using Python's `math.erf`.
Near ±1 the residual `erf(x) − y` is a difference of two numbers within 1e-16 of 1 while the derivative
`2/√π·exp(−x²)` is ~1e-15, so the Newton steps cannot improve the ~1e-9 guess. On Linux the eager
float64 kernel evidently takes a different path (the glibc `erf` or a vectorized kernel) that does not
have this limitation. pytorch/pytorch#198133 reports the same symptom for the Inductor path on Linux.

### Versions

Affected: PyTorch 2.14.1 CPU wheel, Python 3.13.7, macOS 26.3.1 arm64 (Apple Silicon).
Not affected: PyTorch 2.14.1+cu130 on Ubuntu, kernel 6.17, glibc 2.39, x86_64, CPU tensors.
Reference: mpmath 1.3.0 at 60 digits.
