"""Markdown atlas from results.json."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .functions import FUNCTIONS


def _fmt(x: float) -> str:
    if x != x:
        return "—"
    if x >= 1e6:
        return f"{x:.1e}"
    if x >= 100:
        return f"{x:.0f}"
    if x >= 10:
        return f"{x:.1f}"
    return f"{x:.2f}"


def _grade(s: dict) -> str:
    """One glyph per cell: ✅ ≤2 ULP everywhere and no anomalies; ⚠️ ≤1e3 ULP max / rare >4 ULP; ❌ otherwise."""
    if "error" in s:
        return "💥"
    bad = s["n_spurious_nan"] + s["n_spurious_inf"] + s["n_sign_error"] + s["n_missing_nonfinite"]
    if bad == 0 and s["max_ulp"] <= 2.0:
        return "✅"
    if bad <= 2 and s["max_ulp"] <= 1e3 and s["frac_gt_4ulp"] <= 0.01:
        return "⚠️"
    return "❌"


def build_report(results_path: str | Path, out_path: str | Path) -> str:
    payload = json.loads(Path(results_path).read_text())
    meta, results = payload["meta"], payload["results"]
    libs = [k for k, v in meta["libraries"].items() if v["available"]]
    by = defaultdict(dict)
    for r in results:
        by[(r["function"], r["dtype"])][r["library"]] = r
    dtypes = meta["dtypes"]
    lines = [
        "# Special-function accuracy atlas",
        "",
        f"specaudit {meta['specaudit']} · {meta['n_per_function']} samples per function · seed {meta['seed']} · "
        f"Python {meta['python']} · {meta['platform']}",
        "",
        "Libraries: " + ", ".join(f"{k} {meta['libraries'][k]['version']}" for k in libs),
        "",
        "Error is in units in the last place (ULP) of the reference value rounded to the result dtype; the reference is "
        "mpmath at 30 digits. A correctly rounded result is ≤ 0.5 ULP. Cells: max ULP / 99th percentile ULP / % of "
        "samples worse than 4 ULP, then anomalies: `nan` = NaN where the reference is finite, `inf` = overflow where the "
        "reference is finite, `sgn` = wrong sign, `fin` = finite where the reference is ±inf/NaN.",
        "",
        "✅ max ≤ 2 ULP and no anomalies · ⚠️ max ≤ 1000 ULP, < 1 % of samples > 4 ULP, ≤ 2 anomalies · ❌ otherwise · 💥 raised",
        "",
    ]
    for dtype in dtypes:
        lines += [f"## {dtype}", "", "| function | " + " | ".join(libs) + " |", "|---|" + "---|" * len(libs)]
        for fname in FUNCTIONS:
            row = by.get((fname, dtype))
            if not row:
                continue
            cells = []
            for lib in libs:
                r = row.get(lib)
                if r is None:
                    cells.append("·")
                    continue
                s = r["summary"]
                if "error" in s:
                    cells.append("💥 " + s["error"][:40])
                    continue
                anom = []
                if s["n_spurious_nan"]:
                    anom.append(f"nan {s['n_spurious_nan']}")
                if s["n_spurious_inf"]:
                    anom.append(f"inf {s['n_spurious_inf']}")
                if s["n_sign_error"]:
                    anom.append(f"sgn {s['n_sign_error']}")
                if s["n_missing_nonfinite"]:
                    anom.append(f"fin {s['n_missing_nonfinite']}")
                cells.append(
                    f"{_grade(s)} {_fmt(s['max_ulp'])} / {_fmt(s['p99_ulp'])} / {100 * s['frac_gt_4ulp']:.1f}%"
                    + (" · " + ", ".join(anom) if anom else "")
                )
            lines.append(f"| `{fname}` | " + " | ".join(cells) + " |")
        lines.append("")
    # Worst offenders
    lines += ["## Largest errors", "", "The single worst sample per (function, library, dtype) with max ULP > 4:", ""]
    lines += ["| function | library | dtype | input | reference | returned | ULP |", "|---|---|---|---|---|---|---|"]
    rows = []
    for r in results:
        s = r["summary"]
        if "max_ulp" in s and s["max_ulp"] > 4 and r["worst"]:
            w = r["worst"][0]
            rows.append((s["max_ulp"], r, w))
    for _, r, w in sorted(rows, key=lambda t: -t[0])[:60]:
        args = ", ".join(f"{a:.17g}" for a in w["args"])
        lines.append(
            f"| `{r['function']}` | {r['library']} | {r['dtype']} | `{args}` | `{w['ref']:.17g}` | `{w['got']:.17g}` | {_fmt(w['ulp'])} |"
        )
    lines += [
        "",
        "## Anomalies",
        "",
        "NaN / overflow / sign disagreements with the reference (first few per case):",
        "",
    ]
    lines += ["| function | library | dtype | kind | input | reference | returned |", "|---|---|---|---|---|---|---|"]
    for r in results:
        for a in r["anomalies"][:3]:
            args = ", ".join(f"{v:.17g}" for v in a["args"])
            lines.append(
                f"| `{r['function']}` | {r['library']} | {r['dtype']} | {a['kind']} | `{args}` | `{a['ref']:.17g}` | `{a['got']:.17g}` |"
            )
    text = "\n".join(lines) + "\n"
    Path(out_path).write_text(text)
    return text
