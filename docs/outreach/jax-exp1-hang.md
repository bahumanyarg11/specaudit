# jax.scipy.special.exp1 / expi hang on float64 arrays (for the record: fixed upstream in jax-ml/jax#41253)

Status: **already reported and fixed**. jax-ml/jax#13543 (2022, closed) and the merged fix
jax-ml/jax#41253 (2026-10-05, after the 0.11.2 release we audited). Nothing to post; this file records
the reproducer used for the atlas, where `exp1`/`expi` are skipped for JAX.

### Reproducer (jax 0.11.2, float64, CPU; each line run in a subprocess with a 60 s timeout)

```python
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
jax.scipy.special.exp1(jnp.array([1e-300]))        # 0.14 s -> 690.198...
jax.scipy.special.exp1(jnp.array([2.0]))           # 0.10 s -> 0.0489005...
jax.scipy.special.exp1(jnp.array([1e-300, 2.0]))   # no result after 60 s (killed)
jax.scipy.special.exp1(jnp.array([1e-20, 2.0]))    # no result after 60 s
jax.scipy.special.exp1(jnp.array([1e-10, 2.0]))    # no result after 60 s
jax.scipy.special.exp1(jnp.array([1e-6, 2.0]))     # 26 s
jax.scipy.special.exp1(jnp.array([1e-4, 2.0]))     # 0.5 s
jax.scipy.special.exp1(jnp.array([1e-300, 0.5]))   # no result after 60 s (both elements <= 1!)
```

Float32 (the default dtype) hangs too, contrary to the PR's "slow" characterisation: `exp1(jnp.array([1e-10, 2.0],
dtype=jnp.float32))` and `exp1(jnp.array([1e-30, 1e-30], dtype=jnp.float32))` gave no result in 60 s; `[1e-6, 2.0]`
in float32 returns in 0.3 s. `expi(jnp.array([-1e-20, 1.0]))`, `expi([-1e-20, -2.0])` (float64) and
`expi([-1e-10, -2.0])` (float32) hang; `expi([1e-20, 2.0])` (both positive) returns at once.

Every scalar in {1e-300, 1e-200, …, 700} returns in < 0.5 s. The hang needs an array with at least
two elements, one of them ≲ 1e-6: `expn` dispatches with `jnp.piecewise`, which under batching
evaluates every branch on every element, and the x > 1 continued-fraction branch (`_expn2`, loop
condition `(x > 0) & (t > eps)`) does not converge for tiny x. `expi(x < 0)` is `-exp1(-x)` and hangs the
same way. #41253 bounds each branch's loop to its own domain.

Scalar accuracy of `exp1` in 0.11.2 is fine (≤ 8 ULP on the points we checked, same as SciPy).
