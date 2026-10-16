"""Helpers for asserting accuracy in your own test suite.

    from specaudit import assert_max_ulp
    assert_max_ulp(torch.special.erfinv, "erfinv", torch.linspace(-0.999, 0.999, 1001), max_ulp=4)

The callable may take/return NumPy arrays, JAX arrays or PyTorch tensors; the reference is looked
up by name in :mod:`specaudit.references`.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .metrics import classify, summarise, ulp_error
from .references import REFERENCES, reference


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu()
    return np.asarray(x, dtype=np.float64)


def assert_max_ulp(
    fn: Callable, ref_name: str, *inputs, max_ulp: float = 4.0, dtype=np.float64, allow_anomalies: int = 0
) -> dict:
    """Assert that ``fn(*inputs)`` is within ``max_ulp`` of the reference everywhere.

    Returns the summary dict on success; raises AssertionError listing the worst inputs otherwise.
    """
    if ref_name not in REFERENCES:
        raise KeyError(f"no reference named {ref_name!r}; known: {sorted(REFERENCES)}")
    arrays = [_to_numpy(x) for x in inputs]
    got = _to_numpy(fn(*inputs))
    hi, lo = reference(ref_name, *arrays)
    got = got.reshape(hi.shape)
    ulp = ulp_error(got, hi, lo, dtype)
    cat = classify(got, hi)
    s = summarise(ulp, cat)
    anomalies = s.n_spurious_nan + s.n_spurious_inf + s.n_sign_error + s.n_missing_nonfinite
    if (s.max_ulp == s.max_ulp and s.max_ulp > max_ulp) or anomalies > allow_anomalies:
        ok = cat == "ok"
        order = np.argsort(np.where(ok, np.nan_to_num(ulp, nan=-1.0), -1.0))[::-1][:5]
        worst = "; ".join(
            f"x={[float(a.ravel()[i]) for a in arrays]} ref={hi.ravel()[i]:.17g} got={got.ravel()[i]:.17g} ulp={ulp.ravel()[i]:.2f}"
            for i in order
            if ok.ravel()[i]
        )
        bad = np.nonzero(~ok & (cat != "both_nonfinite"))[0][:5]
        anom = "; ".join(
            f"x={[float(a.ravel()[i]) for a in arrays]} {cat.ravel()[i]} got={got.ravel()[i]}" for i in bad
        )
        raise AssertionError(
            f"{ref_name}: max {s.max_ulp:.2f} ULP (limit {max_ulp}), {anomalies} anomalies (limit {allow_anomalies}). "
            f"Worst: {worst or '-'}. Anomalies: {anom or '-'}"
        )
    return s.to_dict()
