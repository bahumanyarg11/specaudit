"""Function registry: reference name, argument domains, and the attribute path in each library.

Domains are sampled by :func:`sample_domain`: log-spaced magnitudes over ``(lo, hi)`` on both signs
where allowed, plus a dense linear band around the origin and the special points listed in
``focus`` (poles, branch points, tails).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Domain:
    lo: float  # smallest magnitude
    hi: float  # largest magnitude
    negative: bool = True  # also sample -x
    linear: tuple[float, float] | None = None  # dense linear band
    focus: tuple[float, ...] = ()  # points to sample densely around (±1e-12..1e-2 offsets)
    integer: bool = False  # integer-valued parameter (e.g. polygamma order)
    integers: tuple[int, ...] = ()


@dataclass(frozen=True)
class FunctionSpec:
    name: str
    domains: tuple[Domain, ...]
    scipy: str | None = None  # attribute in scipy.special
    jax: str | None = None  # attribute in jax.scipy.special (or "jax.numpy:" prefix)
    torch: str | None = None  # attribute in torch.special (or "torch:" prefix)
    numpy: str | None = None  # attribute in numpy
    description: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def arity(self) -> int:
        return len(self.domains)


R = Domain
_POS = R(1e-300, 1e300, negative=False)
_REAL = R(1e-300, 1e300)
_REAL_MOD = R(1e-300, 1e4)

FUNCTIONS: dict[str, FunctionSpec] = {}


def _add(spec: FunctionSpec) -> None:
    FUNCTIONS[spec.name] = spec


_add(
    FunctionSpec(
        "gamma",
        (R(1e-300, 180.0, focus=(-1, -2, -3, -10, -50, 0.5, 1, 2, 171.6)),),
        scipy="gamma",
        jax="gamma",
        description="Gamma function",
    )
)
_add(
    FunctionSpec(
        "lgamma",
        (R(1e-300, 1e300, focus=(-1, -2, -3, -10, -50, 1, 2, 0.5)),),
        scipy="gammaln",
        jax="gammaln",
        torch="gammaln",
        description="log|Gamma(x)|",
    )
)
_add(
    FunctionSpec(
        "digamma",
        (R(1e-300, 1e300, focus=(-1, -2, -10, 1.4616321449683623, 0.5)),),
        scipy="digamma",
        jax="digamma",
        torch="digamma",
        description="psi(x)",
    )
)
_add(
    FunctionSpec(
        "polygamma",
        (R(1, 4, negative=False, integer=True, integers=(1, 2, 3, 4)), R(1e-300, 1e6, focus=(-1, -2, 0.5, 1))),
        scipy="polygamma",
        jax="polygamma",
        torch="polygamma",
        description="psi^(n)(x), n = 1..4",
    )
)
_add(FunctionSpec("erf", (R(1e-300, 1e3, focus=(0.5, 1, 2, 5.9)),), scipy="erf", jax="erf", torch="erf"))
_add(
    FunctionSpec(
        "erfc", (R(1e-300, 1e3, focus=(5, 10, 20, 26.5, -5)),), scipy="erfc", jax="erfc", torch="erfc", tags=("tail",)
    )
)
_add(FunctionSpec("erfcx", (R(1e-300, 1e300, focus=(-26, -20, 20, 1e4)),), scipy="erfcx", torch="erfcx"))
_add(
    FunctionSpec(
        "erfinv",
        (R(1e-300, 1.0, linear=(-1.0, 1.0), focus=(0.999999, -0.999999, 0.5)),),
        scipy="erfinv",
        jax="erfinv",
        torch="erfinv",
        tags=("inverse",),
    )
)
_add(FunctionSpec("ndtr", (R(1e-300, 1e3, focus=(-38, -30, -10, 8, 10)),), scipy="ndtr", jax="ndtr", torch="ndtr"))
_add(
    FunctionSpec(
        "ndtri",
        (R(1e-300, 1.0, negative=False, linear=(0.0, 1.0), focus=(1e-300, 1e-100, 1e-20, 0.5, 0.999999, 1 - 1e-15)),),
        scipy="ndtri",
        jax="ndtri",
        torch="ndtri",
        tags=("inverse", "tail"),
    )
)
_add(
    FunctionSpec(
        "log_ndtr",
        (R(1e-300, 1e4, focus=(-38, -30, -20, -10, -5, 5, 10, 30)),),
        scipy="log_ndtr",
        jax="log_ndtr",
        torch="log_ndtr",
        tags=("tail",),
    )
)
_add(
    FunctionSpec(
        "expit", (R(1e-300, 1e3, focus=(-745, -709, -36, 36, 709)),), scipy="expit", jax="expit", torch="expit"
    )
)
_add(
    FunctionSpec(
        "logit",
        (R(1e-300, 1.0, negative=False, linear=(0.0, 1.0), focus=(1e-300, 1e-20, 0.5, 1 - 1e-16)),),
        scipy="logit",
        jax="logit",
        torch="logit",
    )
)
_add(
    FunctionSpec(
        "log1p", (R(1e-300, 1e300, focus=(-1 + 1e-15, -0.5, 1e-16, 1)),), scipy="log1p", numpy="log1p", torch="log1p"
    )
)
_add(FunctionSpec("expm1", (R(1e-300, 710.0, focus=(-745, 1e-16, 709)),), scipy="expm1", numpy="expm1", torch="expm1"))
_add(FunctionSpec("i0", (R(1e-300, 710.0, focus=(700, 709)),), scipy="i0", jax="i0", torch="i0"))
_add(FunctionSpec("i0e", (_REAL,), scipy="i0e", jax="i0e", torch="i0e"))
_add(FunctionSpec("i1", (R(1e-300, 710.0, focus=(700, 709)),), scipy="i1", jax="i1", torch="i1"))
_add(FunctionSpec("i1e", (_REAL,), scipy="i1e", jax="i1e", torch="i1e"))
_add(
    FunctionSpec(
        "j0",
        (R(1e-300, 1e5, focus=(2.404825557695773, 5.520078110286311, 100, 1e4)),),
        scipy="j0",
        torch="bessel_j0",
        description="Bessel J0; zeros are hard",
    )
)
_add(
    FunctionSpec(
        "j1", (R(1e-300, 1e5, focus=(3.8317059702075125, 7.015586669815619, 100)),), scipy="j1", torch="bessel_j1"
    )
)
_add(
    FunctionSpec(
        "y0",
        (R(1e-300, 1e5, negative=False, focus=(0.8935769662791675, 3.957678419314858, 100)),),
        scipy="y0",
        torch="bessel_y0",
    )
)
_add(
    FunctionSpec(
        "y1",
        (R(1e-300, 1e5, negative=False, focus=(2.197141326031017, 5.429681040794135, 100)),),
        scipy="y1",
        torch="bessel_y1",
    )
)
_add(FunctionSpec("k0", (R(1e-300, 700.0, negative=False),), scipy="k0", torch="modified_bessel_k0"))
_add(FunctionSpec("k1", (R(1e-300, 700.0, negative=False),), scipy="k1", torch="modified_bessel_k1"))
_add(FunctionSpec("k0e", (R(1e-300, 1e300, negative=False),), scipy="k0e", torch="scaled_modified_bessel_k0"))
_add(FunctionSpec("k1e", (R(1e-300, 1e300, negative=False),), scipy="k1e", torch="scaled_modified_bessel_k1"))
_add(
    FunctionSpec(
        "gammainc",
        (R(1e-3, 1e3, negative=False), R(1e-300, 1e3, negative=False)),
        scipy="gammainc",
        jax="gammainc",
        torch="gammainc",
        description="regularised lower incomplete gamma P(a, x)",
    )
)
_add(
    FunctionSpec(
        "gammaincc",
        (R(1e-3, 1e3, negative=False), R(1e-300, 1e3, negative=False)),
        scipy="gammaincc",
        jax="gammaincc",
        torch="gammaincc",
        description="regularised upper incomplete gamma Q(a, x)",
        tags=("tail",),
    )
)
_add(
    FunctionSpec(
        "betainc",
        (R(1e-2, 2e2, negative=False), R(1e-2, 2e2, negative=False), R(1e-300, 1.0, negative=False, linear=(0.0, 1.0))),
        scipy="betainc",
        jax="betainc",
        description="regularised incomplete beta I_x(a, b)",
    )
)
_add(
    FunctionSpec(
        "zeta",
        (R(1.0 + 1e-6, 1e3, negative=False, focus=(1.000001, 1.5, 2, 3)), R(1e-6, 1e4, negative=False)),
        scipy="zeta",
        jax="zeta",
        torch="zeta",
        description="Hurwitz zeta(s, q)",
    )
)
_add(FunctionSpec("spence", (R(1e-300, 1e300, negative=False, focus=(1, 2, 0.5)),), scipy="spence", jax="spence"))
_add(
    FunctionSpec(
        "exp1",
        (R(1e-300, 700.0, negative=False),),
        scipy="exp1",
        jax=None,  # jax.scipy.special.exp1 (float64) did not return within 15 min on this sample set: skipped, see README
        description="E1(x); JAX implementation skipped (hangs on extreme arguments)",
    )
)
_add(
    FunctionSpec(
        "expi",
        (R(1e-300, 700.0, focus=(0.37250741078136663,)),),
        scipy="expi",
        jax=None,
        description="Ei(x); JAX implementation skipped (hangs on extreme arguments)",
    )
)
_add(
    FunctionSpec(
        "xlogy", (R(1e-300, 1e300), R(1e-300, 1e300, negative=False)), scipy="xlogy", jax="xlogy", torch="xlogy"
    )
)
_add(FunctionSpec("entr", (R(1e-300, 1e300, negative=False),), scipy="entr", jax="entr", torch="entr"))
_add(FunctionSpec("sinc", (R(1e-300, 1e6, focus=(1, 2, 3, 1e-8)),), scipy="sinc", numpy="sinc", torch="sinc"))
_add(FunctionSpec("beta", (R(1e-3, 170.0, negative=False), R(1e-3, 170.0, negative=False)), scipy="beta"))
_add(
    FunctionSpec(
        "betaln", (R(1e-3, 1e300, negative=False), R(1e-3, 1e300, negative=False)), scipy="betaln", jax="betaln"
    )
)
_add(
    FunctionSpec("airy_ai", (R(1e-300, 100.0, focus=(-2.338107410459767, -4.08794944413097, 5, 50)),), torch="airy_ai")
)


def sample_domain(domain: Domain, n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample ``n`` points from a domain: log-uniform magnitudes (both signs if allowed), a linear
    band, and dense neighbourhoods of the focus points."""
    if domain.integer:
        return rng.choice(np.asarray(domain.integers, dtype=float), size=n)
    parts = []
    n_log = n // 2 if (domain.linear or domain.focus) else n
    mags = 10 ** rng.uniform(np.log10(domain.lo), np.log10(domain.hi), n_log)
    if domain.negative:
        mags = mags * rng.choice([-1.0, 1.0], size=n_log)
    parts.append(mags)
    rest = n - n_log
    if domain.linear and domain.focus:
        n_lin, n_focus = rest // 2, rest - rest // 2
    elif domain.linear:
        n_lin, n_focus = rest, 0
    else:
        n_lin, n_focus = 0, rest
    if n_lin:
        parts.append(rng.uniform(domain.linear[0], domain.linear[1], n_lin))
    if n_focus and domain.focus:
        per = max(1, n_focus // len(domain.focus))
        for f in domain.focus:
            offs = 10 ** rng.uniform(-14, -1, per) * rng.choice([-1.0, 1.0], size=per) * max(abs(f), 1.0)
            parts.append(f + offs)
    x = np.concatenate(parts)
    if x.size < n:  # rounding in the focus split: top up with log-uniform points
        extra = 10 ** rng.uniform(np.log10(domain.lo), np.log10(domain.hi), n - x.size)
        if domain.negative:
            extra = extra * rng.choice([-1.0, 1.0], size=extra.size)
        x = np.concatenate([x, extra])
    x = x[:n]
    lo_abs = min(domain.lo, 1e-300)
    if not domain.negative:
        x = np.abs(x)
        x = np.where(x < lo_abs, lo_abs, x)
    return x
