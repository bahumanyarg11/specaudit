"""Sweep runner: sample domains, compute references once, evaluate every library, summarise."""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ._version import __version__
from .functions import FUNCTIONS, FunctionSpec, sample_domain
from .libs import LIBRARIES, Library, adapter, load_library
from .metrics import classify, summarise, ulp_error
from .references import reference

WORST_KEEP = 25


@dataclass
class CaseResult:
    function: str
    library: str
    library_version: str
    dtype: str
    summary: dict
    worst: list[dict] = field(default_factory=list)  # the WORST_KEEP largest-ULP inputs
    anomalies: list[dict] = field(default_factory=list)  # spurious nan/inf, sign errors (up to WORST_KEEP)
    seconds: float = 0.0


def evaluate(spec: FunctionSpec, lib: Library, dtype, args: list[np.ndarray], ref_hi, ref_lo) -> CaseResult | None:
    fn = adapter(lib, spec, dtype)
    if fn is None:
        return None
    t0 = time.perf_counter()
    try:
        got = fn(*args)
    except Exception as exc:  # noqa: BLE001 - a library raising on valid input is itself a finding
        return CaseResult(
            spec.name,
            lib.name,
            lib.version,
            np.dtype(dtype).name,
            {"error": f"{type(exc).__name__}: {str(exc)[:200]}"},
            seconds=time.perf_counter() - t0,
        )
    dt = time.perf_counter() - t0
    got = np.asarray(got, dtype=np.float64).reshape(ref_hi.shape)
    ulp = ulp_error(got, ref_hi, ref_lo, dtype)
    cat = classify(got, ref_hi)
    summ = summarise(ulp, cat)
    ok = cat == "ok"
    order = np.argsort(np.where(ok, np.nan_to_num(ulp, nan=-1.0), -1.0))[::-1][:WORST_KEEP]
    worst = [
        {"args": [float(a[i]) for a in args], "ref": float(ref_hi[i]), "got": float(got[i]), "ulp": float(ulp[i])}
        for i in order
        if ok[i] and np.isfinite(ulp[i]) and ulp[i] > 0.5
    ]
    bad = np.nonzero(~ok & (cat != "both_nonfinite"))[0][:WORST_KEEP]
    anomalies = [
        {"args": [float(a[i]) for a in args], "ref": float(ref_hi[i]), "got": float(got[i]), "kind": str(cat[i])}
        for i in bad
    ]
    return CaseResult(spec.name, lib.name, lib.version, np.dtype(dtype).name, summ.to_dict(), worst, anomalies, dt)


def run_audit(
    functions: list[str] | None = None,
    libraries: list[str] | None = None,
    dtypes: tuple = (np.float64,),
    n: int = 2000,
    seed: int = 0,
    out_dir: str | Path | None = None,
    verbose: bool = True,
) -> dict:
    functions = functions or list(FUNCTIONS)
    libraries = libraries or list(LIBRARIES)
    libs = {name: load_library(name) for name in libraries}
    rng = np.random.default_rng(seed)
    results: list[CaseResult] = []
    for fname in functions:
        spec = FUNCTIONS[fname]
        args = [sample_domain(d, n, rng) for d in spec.domains]
        t0 = time.perf_counter()
        ref_hi, ref_lo = reference(spec.name, *args)
        tref = time.perf_counter() - t0
        for dtype in dtypes:
            for lib in libs.values():
                res = evaluate(spec, lib, dtype, args, ref_hi, ref_lo)
                if res is None:
                    continue
                results.append(res)
                if verbose:
                    s = res.summary
                    line = (
                        f"{fname:10s} {lib.name:6s} {np.dtype(dtype).name:8s} "
                        + (
                            f"max {s['max_ulp']:9.2f} ulp  p99 {s['p99_ulp']:7.2f}  >4ulp {100 * s['frac_gt_4ulp']:5.1f}%  "
                            f"nan {s['n_spurious_nan']:3d} inf {s['n_spurious_inf']:3d} sign {s['n_sign_error']:2d}"
                            if "max_ulp" in s
                            else f"ERROR {s['error']}"
                        )
                        + f"  ({res.seconds * 1e3:.0f} ms; ref {tref:.1f} s)"
                    )
                    print(line, flush=True)
    meta = {
        "specaudit": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "n_per_function": n,
        "seed": seed,
        "libraries": {k: {"version": v.version, "available": v.available, "reason": v.reason} for k, v in libs.items()},
        "dtypes": [np.dtype(d).name for d in dtypes],
    }
    payload = {"meta": meta, "results": [asdict(r) for r in results]}
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "results.json").write_text(json.dumps(payload, indent=1))
    return payload
