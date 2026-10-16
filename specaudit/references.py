"""Arbitrary-precision reference values via mpmath.

Every reference returns a *double-double*: ``hi`` is the value rounded to float64, ``lo`` the
remainder ``value - hi`` also as float64. That keeps ~32 significant digits in two floats, enough
to measure float64 results to a small fraction of an ULP without carrying mpmath objects around.
Non-finite results are encoded as ``hi = ±inf``/``nan`` with ``lo = 0``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from functools import lru_cache

import mpmath as mp
import numpy as np

DPS = 30


def _dd(value) -> tuple[float, float]:
    """Split an mpf into (hi, lo) float64s."""
    if isinstance(value, mp.mpc):
        if abs(value.imag) > abs(value.real) * mp.mpf(10) ** -25:
            return math.nan, 0.0
        value = value.real
    if mp.isnan(value):
        return math.nan, 0.0
    if mp.isinf(value):
        return (math.inf if value > 0 else -math.inf), 0.0
    hi = float(value)
    if math.isinf(hi):  # overflow in float64
        return hi, 0.0
    lo = float(value - mp.mpf(hi))
    return hi, lo


TIMEOUTS: list[tuple[str, tuple[float, ...]]] = []  # kept for the results schema; domains are capped instead


def _wrap(fn: Callable) -> Callable[..., tuple[float, float]]:
    def wrapped(*args: float) -> tuple[float, float]:
        with mp.workdps(DPS):
            try:
                margs = [mp.mpf(a) for a in args]
                return _dd(fn(*margs))
            except (ValueError, ZeroDivisionError, OverflowError, TypeError):
                return math.nan, 0.0

    return wrapped


# ---------------------------------------------------------------- scalar references (mpmath)
def _lgamma(x):
    # log|Gamma(x)| as scipy/torch/jax define it for real x
    if x <= 0 and x == mp.floor(x):
        return mp.inf
    return mp.log(abs(mp.gamma(x)))


def _gamma(x):
    if x <= 0 and x == mp.floor(x):
        return mp.nan if x != 0 else mp.inf
    return mp.gamma(x)


def _digamma(x):
    if x <= 0 and x == mp.floor(x):
        return mp.nan
    return mp.digamma(x)


def _polygamma(n, x):
    if x <= 0 and x == mp.floor(x):
        return mp.nan
    return mp.polygamma(int(n), x)


def _erfcx(x):
    if x > 1e4:  # asymptotic series; mp.erfc itself returns 0 for astronomically large x
        inv = 1 / (2 * x * x)
        return (1 - inv + 3 * inv**2 - 15 * inv**3 + 105 * inv**4) / (x * mp.sqrt(mp.pi))
    return mp.exp(x * x) * mp.erfc(x)


def _erfinv(x):
    if abs(x) > 1:
        return mp.nan
    if x == 1:
        return mp.inf
    if x == -1:
        return -mp.inf
    return mp.erfinv(x)


def _ndtr(x):
    return mp.ncdf(x)


def _ndtri(p):
    if p < 0 or p > 1:
        return mp.nan
    if p == 0:
        return -mp.inf
    if p == 1:
        return mp.inf
    # erfinv(2q-1) needs about -log10(q) extra digits for tiny q: raise the precision accordingly.
    mirrored = p > mp.mpf("0.5")
    q = 1 - p if mirrored else p
    extra = int(-mp.log10(q)) + 10 if q < mp.mpf("1e-10") else 0
    with mp.workdps(DPS + extra):
        root = mp.sqrt(2) * mp.erfinv(2 * mp.mpf(q) - 1)
    return -root if mirrored else root


def _log_ndtr(x):
    # for x > 0, ncdf(x) = 1 - tiny rounds to 1: use log1p(-ncdf(-x)) instead
    if x > 0:
        return mp.log1p(-mp.ncdf(-x))
    return mp.log(mp.ncdf(x))


def _expit(x):
    return 1 / (1 + mp.exp(-x))


def _logit(p):
    if p < 0 or p > 1:
        return mp.nan
    if p == 0:
        return -mp.inf
    if p == 1:
        return mp.inf
    return mp.log(p / (1 - p))


def _i0e(x):
    return mp.besseli(0, x) * mp.exp(-abs(x))


def _i1e(x):
    return mp.besseli(1, x) * mp.exp(-abs(x))


def _k0e(x):
    return mp.besselk(0, x) * mp.exp(x)


def _k1e(x):
    return mp.besselk(1, x) * mp.exp(x)


def _gammainc_p(a, x):
    if a <= 0 or x < 0:
        return mp.nan
    return mp.gammainc(a, 0, x, regularized=True)


def _gammainc_q(a, x):
    if a <= 0 or x < 0:
        return mp.nan
    return mp.gammainc(a, x, mp.inf, regularized=True)


def _betainc(a, b, x):
    if a <= 0 or b <= 0 or x < 0 or x > 1:
        return mp.nan
    return mp.betainc(a, b, 0, x, regularized=True)


def _hurwitz_zeta(s, q):
    if s <= 1 or q <= 0:
        return mp.nan
    return mp.zeta(s, q)


def _spence(z):
    # SciPy convention: spence(z) = integral_1^z log(t)/(1-t) dt = polylog(2, 1 - z) for real z >= 0
    if z < 0:
        return mp.nan
    return mp.polylog(2, 1 - z)


def _exp1(x):
    if x < 0:
        return mp.nan
    if x == 0:
        return mp.inf
    return mp.expint(1, x)


def _expi(x):
    if x == 0:
        return -mp.inf
    return mp.ei(x)


def _xlogy(x, y):
    if x == 0 and not mp.isnan(y):
        return mp.mpf(0)
    if y < 0:
        return mp.nan
    return x * mp.log(y)


def _entr(x):
    if x < 0:
        return -mp.inf
    if x == 0:
        return mp.mpf(0)
    return -x * mp.log(x)


def _sinc(x):
    if x == 0:
        return mp.mpf(1)
    return mp.sin(mp.pi * x) / (mp.pi * x)


def _beta(a, b):
    return mp.beta(a, b)


def _betaln(a, b):
    if a <= 0 or b <= 0:
        return mp.nan
    return mp.log(abs(mp.beta(a, b)))


def _log1p(x):
    if x <= -1:
        return -mp.inf if x == -1 else mp.nan
    return mp.log1p(x)


def _expm1(x):
    return mp.expm1(x)


REFERENCES: dict[str, Callable[..., tuple[float, float]]] = {
    "gamma": _wrap(_gamma),
    "lgamma": _wrap(_lgamma),
    "digamma": _wrap(_digamma),
    "polygamma": _wrap(_polygamma),
    "erf": _wrap(mp.erf),
    "erfc": _wrap(mp.erfc),
    "erfcx": _wrap(_erfcx),
    "erfinv": _wrap(_erfinv),
    "ndtr": _wrap(_ndtr),
    "ndtri": _wrap(_ndtri),
    "log_ndtr": _wrap(_log_ndtr),
    "expit": _wrap(_expit),
    "logit": _wrap(_logit),
    "log1p": _wrap(_log1p),
    "expm1": _wrap(_expm1),
    "i0": _wrap(lambda x: mp.besseli(0, x)),
    "i1": _wrap(lambda x: mp.besseli(1, x)),
    "i0e": _wrap(_i0e),
    "i1e": _wrap(_i1e),
    "j0": _wrap(lambda x: mp.besselj(0, x)),
    "j1": _wrap(lambda x: mp.besselj(1, x)),
    "y0": _wrap(lambda x: mp.bessely(0, x) if x > 0 else mp.nan),
    "y1": _wrap(lambda x: mp.bessely(1, x) if x > 0 else mp.nan),
    "k0": _wrap(lambda x: mp.besselk(0, x) if x > 0 else mp.nan),
    "k1": _wrap(lambda x: mp.besselk(1, x) if x > 0 else mp.nan),
    "k0e": _wrap(lambda x: _k0e(x) if x > 0 else mp.nan),
    "k1e": _wrap(lambda x: _k1e(x) if x > 0 else mp.nan),
    "gammainc": _wrap(_gammainc_p),
    "gammaincc": _wrap(_gammainc_q),
    "betainc": _wrap(_betainc),
    "zeta": _wrap(_hurwitz_zeta),
    "spence": _wrap(_spence),
    "exp1": _wrap(_exp1),
    "expi": _wrap(_expi),
    "xlogy": _wrap(_xlogy),
    "entr": _wrap(_entr),
    "sinc": _wrap(_sinc),
    "beta": _wrap(_beta),
    "betaln": _wrap(_betaln),
    "airy_ai": _wrap(lambda x: mp.airyai(x)),
}


@lru_cache(maxsize=1_000_000)
def _cached(name: str, args: tuple[float, ...]) -> tuple[float, float]:
    return REFERENCES[name](*args)


def reference(name: str, *arrays: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised reference: returns (hi, lo) float64 arrays for broadcast inputs."""
    arrays = np.broadcast_arrays(*[np.asarray(a, dtype=np.float64) for a in arrays])
    flat = [a.ravel() for a in arrays]
    n = flat[0].size
    hi = np.empty(n)
    lo = np.empty(n)
    for i in range(n):
        args = tuple(float(f[i]) for f in flat)
        if any(math.isnan(a) for a in args):
            hi[i], lo[i] = math.nan, 0.0
            continue
        hi[i], lo[i] = _cached(name, args)
    return hi.reshape(arrays[0].shape), lo.reshape(arrays[0].shape)
