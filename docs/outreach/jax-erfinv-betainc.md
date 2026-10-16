# Two float64 accuracy gaps in jax.scipy.special: erfinv near ±1 (up to 5e5 ULP) and betainc (p99 ≈ 940 ULP, batch-dependent)

Both found with `specaudit` (`pip install specaudit`), 500 float64 samples per function, reference mpmath
at 50 digits, SciPy 1.18.1 for comparison. jax 0.11.2 CPU, Python 3.13.7, macOS 26.3.1 arm64.

## 1. erfinv for |x| → 1

`jax.scipy.special.erfinv` (= `lax.erf_inv`) is within 2 ULP for |x| < 0.999 but loses accuracy as
|x| → 1, worst around 1 − x ≈ 1e-9 … 1e-10 where the error reaches 5e5 ULP (4.4e-10 absolute). It
recovers for 1 − x < 1e-14. SciPy's `erfinv` (Boost) stays ≤ 2 ULP throughout.

```python
import jax, jax.numpy as jnp, mpmath, numpy as np, scipy.special
jax.config.update("jax_enable_x64", True)
mpmath.mp.dps = 60
def ulp(got, ref): return abs(got - ref) / np.spacing(abs(ref))  # ULP of the reference
for k in [20, 24, 26, 28, 30, 32, 36, 40]:
    x = 1 - 2.0**-k
    ref = float(mpmath.erfinv(mpmath.mpf(x)))
    got = float(jax.scipy.special.erfinv(jnp.float64(x)))
    print(f"x = 1 - 2^-{k:<3d} jax={got!r:22} ref={ref!r:22} {ulp(got, ref):8.0f} ULP   scipy {ulp(float(scipy.special.erfinv(x)), ref):.0f} ULP")
```

```
x = 1 - 2^-20  jax=3.4655050258033313     ref=3.465505025803331             1 ULP   scipy 1 ULP
x = 1 - 2^-24  jax=3.8325068569007104     ref=3.832506856900711             1 ULP   scipy 0 ULP
x = 1 - 2^-26  jax=4.004131885114474      ref=4.004131885114475             1 ULP   scipy 0 ULP
x = 1 - 2^-28  jax=4.1690670866841755     ref=4.1690670869016255       244827 ULP   scipy 0 ULP
x = 1 - 2^-30  jax=4.328028275748485      ref=4.328028275800946         59066 ULP   scipy 0 ULP
x = 1 - 2^-32  jax=4.481612907106162      ref=4.481612907118849         14284 ULP   scipy 0 ULP
x = 1 - 2^-36  jax=4.774597423656005      ref=4.7745974236567505          839 ULP   scipy 0 ULP
x = 1 - 2^-40  jax=5.051254085249347      ref=5.05125408524939             48 ULP   scipy 0 ULP
```

On a 20 000-point grid of 1 − x log-spaced in [1e-16, 1e-2] the maximum error by decade is: [1e-10, 1e-8)
5.0e5 ULP, [1e-8, 1e-7) 4.6e5, [1e-7, 1e-6) 7.9e4, [1e-6, 1e-5) 8.7e3, [1e-5, 1e-4) 906, [1e-4, 1e-3)
102, [1e-3, 1e-2) 12. This looks like the error profile of the Giles (2010) double-precision rational
approximation in its middle segment (w = −log(1−x²) between ~16 and ~27) without a Newton/Halley
polish step; one Newton step on `erf(x) − y` computed via `erfc` would bring it to ~1 ULP. Float32 is
fine (≤ 6e-6 relative, same as PyTorch).

## 2. betainc

`jax.scipy.special.betainc` (= `lax.betainc`, Lentz continued fraction, 600 iterations max) has a
99th-percentile error of 937 ULP and 57 % of samples more than 4 ULP off, against 73 ULP / 14 % for
SciPy's `betainc` (Boost `ibeta`) on the same 500 samples (a, b log-uniform in [0.01, 200], x in [0, 1]).

```python
import jax, jax.numpy as jnp, mpmath, numpy as np, scipy.special
jax.config.update("jax_enable_x64", True)
mpmath.mp.dps = 50
def ulp(got, ref): return abs(got - ref) / np.spacing(abs(ref))  # ULP of the reference
cases = [(170.14007970410853, 7.063291545468863, 0.381646572597626),
         (164.93605942738154, 0.027065444102122704, 0.6156738229997158),
         (5.071001278782578, 0.11186044140964943, 0.8874567825471773),
         (50.0, 60.0, 0.45)]
for a, b, x in cases:
    ref = float(mpmath.betainc(a, b, 0, x, regularized=True))
    got = float(jax.scipy.special.betainc(jnp.float64(a), jnp.float64(b), jnp.float64(x)))
    print(f"betainc({a:.6g}, {b:.6g}, {x:.6g}) = {got!r}  ref {ref!r}  jax {ulp(got, ref):6.0f} ULP  scipy {ulp(float(scipy.special.betainc(a, b, x)), ref):4.0f} ULP")
```

```
betainc(170.14, 7.06329, 0.381647) = 1.732454973012789e-62  ref 1.732454973013431e-62  jax   2974 ULP  scipy  122 ULP
betainc(164.936, 0.0270654, 0.615674) = 8.66909529055828e-39  ref 8.66909529056035e-39  jax   1586 ULP  scipy   66 ULP
betainc(5.071, 0.11186, 0.887457) = 0.059185706302427143  ref 0.05918570630242508  jax    297 ULP  scipy    1 ULP
betainc(50, 60, 0.45) = 0.46423529143061076  ref 0.4642352914306036  jax    129 ULP  scipy   16 ULP
```

The result also depends on what else is in the array: `betainc(0.6729797933675461, 0.01446774277865901,
0.9425232008397618)` is 69 ULP off evaluated alone (0.050057453983172406) but 3 963 ULP off
(0.050057453983200384) when evaluated as one element of the 500-sample array above; its 25 worst
inputs reproduce the same large errors inside any 500-element batch we tried and the small ones alone.
In `lentz_thompson_barnett_algorithm` the loop runs until *every* element has converged and keeps
multiplying already-converged elements by their `delta`, so the extra iterations forced by slow elements
change converged results; a per-element `select` on convergence would make results batch-independent.

### Expected behaviour

Within a few ULP of the reference (SciPy manages ≤ 2 ULP for erfinv everywhere, and ~1 ULP for most
betainc inputs), and `betainc` results that do not depend on the other elements of the array.
