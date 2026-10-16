"""Error measures.

The unit of measure is the ULP (unit in the last place) *of the reference value in the result's
dtype*: ``|got - ref| / spacing(ref)``. One ULP means "the nearest representable neighbour"; a
correctly rounded result is within 0.5 ULP. Values are computed from the double-double reference
``(hi, lo)`` so that float64 results can be judged to ~1e-3 ULP.

Special cases are classified rather than measured:

* ``both_nonfinite``   reference and result agree on inf/-inf/nan (counts as correct)
* ``spurious_nan``     result is NaN where the reference is finite
* ``spurious_inf``     result is ±inf where the reference is finite (overflow in the implementation)
* ``missing_nonfinite`` reference is non-finite but the result is finite
* ``sign_error``       finite, nonzero, opposite sign
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CATEGORIES = ("ok", "both_nonfinite", "spurious_nan", "spurious_inf", "missing_nonfinite", "sign_error")


def ulp_error(got: np.ndarray, ref_hi: np.ndarray, ref_lo: np.ndarray, dtype=np.float64) -> np.ndarray:
    """ULP distance between ``got`` and the double-double reference, in the ULP of ``dtype``.

    Returns NaN where either side is non-finite (see :func:`classify`). Results in a narrower dtype
    (float32) are measured in float32 ULPs of the reference rounded to float32.
    """
    got = np.asarray(got, dtype=np.float64)
    ref_hi = np.asarray(ref_hi, dtype=np.float64)
    ref_lo = np.asarray(ref_lo, dtype=np.float64)
    out = np.full(got.shape, np.nan)
    finite = np.isfinite(got) & np.isfinite(ref_hi)
    if not finite.any():
        return out
    g, h, lo = got[finite], ref_hi[finite], ref_lo[finite]
    if np.dtype(dtype) == np.float64:
        spacing = np.spacing(np.abs(h))
        # (g - h) is exact when g and h are within a factor of 2 (Sterbenz); subtracting lo adds the
        # sub-ULP part of the reference.
        err = (g - h) - lo
    else:
        href = h.astype(dtype).astype(np.float64)
        spacing = np.spacing(np.abs(h.astype(dtype))).astype(np.float64)
        err = (g - href) - ((h - href) + lo)
    # Avoid a zero spacing at exact 0 (reference exactly zero): use the smallest subnormal.
    spacing = np.where(spacing == 0, np.finfo(dtype).smallest_subnormal, spacing)
    out[finite] = np.abs(err) / spacing
    return out


def classify(got: np.ndarray, ref_hi: np.ndarray) -> np.ndarray:
    """Return an array of category strings (see module docstring)."""
    got = np.asarray(got, dtype=np.float64)
    ref_hi = np.asarray(ref_hi, dtype=np.float64)
    cat = np.full(got.shape, "ok", dtype=object)
    gf, rf = np.isfinite(got), np.isfinite(ref_hi)
    both_nf = ~gf & ~rf
    same = both_nf & ((np.isnan(got) & np.isnan(ref_hi)) | (got == ref_hi))
    cat[same] = "both_nonfinite"
    cat[both_nf & ~same] = "missing_nonfinite"  # e.g. +inf vs nan: treat as disagreement
    cat[~gf & rf & np.isnan(got)] = "spurious_nan"
    cat[~gf & rf & np.isinf(got)] = "spurious_inf"
    cat[gf & ~rf] = "missing_nonfinite"
    sign = gf & rf & (got != 0) & (ref_hi != 0) & (np.sign(got) != np.sign(ref_hi))
    cat[sign] = "sign_error"
    return cat


@dataclass
class Summary:
    n: int
    n_ok: int
    max_ulp: float
    p50_ulp: float
    p99_ulp: float
    frac_le_1ulp: float
    frac_gt_4ulp: float
    n_spurious_nan: int
    n_spurious_inf: int
    n_missing_nonfinite: int
    n_sign_error: int
    worst_index: int

    def to_dict(self) -> dict:
        return {k: (float(v) if isinstance(v, float) else int(v)) for k, v in self.__dict__.items()}


def summarise(ulp: np.ndarray, cat: np.ndarray) -> Summary:
    ok = cat == "ok"
    vals = ulp[ok]
    vals = vals[np.isfinite(vals)]
    if vals.size:
        worst_local = int(np.argmax(np.where(ok, np.nan_to_num(ulp, nan=-1.0), -1.0)))
        mx, p50, p99 = float(vals.max()), float(np.percentile(vals, 50)), float(np.percentile(vals, 99))
        le1, gt4 = float(np.mean(vals <= 1.0)), float(np.mean(vals > 4.0))
    else:
        worst_local, mx, p50, p99, le1, gt4 = -1, float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
    return Summary(
        n=int(cat.size),
        n_ok=int(ok.sum()),
        max_ulp=mx,
        p50_ulp=p50,
        p99_ulp=p99,
        frac_le_1ulp=le1,
        frac_gt_4ulp=gt4,
        n_spurious_nan=int((cat == "spurious_nan").sum()),
        n_spurious_inf=int((cat == "spurious_inf").sum()),
        n_missing_nonfinite=int((cat == "missing_nonfinite").sum()),
        n_sign_error=int((cat == "sign_error").sum()),
        worst_index=worst_local,
    )
