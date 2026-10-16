"""specaudit CLI.

specaudit run [-f gamma,erf,...] [-l scipy,jax,torch] [--float32] [-n 2000] [-o results/]
specaudit report results/results.json -o ATLAS.md
specaudit list
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from ._version import __version__
from .functions import FUNCTIONS
from .libs import LIBRARIES


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="specaudit", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--version", action="version", version=f"specaudit {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="evaluate libraries against the references")
    p.add_argument("-f", "--functions", help="comma-separated subset (default: all)")
    p.add_argument("-l", "--libraries", help="comma-separated subset of " + ",".join(LIBRARIES))
    p.add_argument("--float32", action="store_true", help="also audit float32 results")
    p.add_argument("--only-float32", action="store_true")
    p.add_argument("-n", type=int, default=2000, help="samples per function")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("-o", "--out", default="results")
    p.add_argument("-q", "--quiet", action="store_true")

    r = sub.add_parser("report", help="write the Markdown atlas")
    r.add_argument("results")
    r.add_argument("-o", "--out", default="ATLAS.md")

    sub.add_parser("list", help="list audited functions and library coverage")

    args = ap.parse_args(argv)
    if args.cmd == "list":
        print(f"{'function':11s} {'args':4s} {'scipy':8s} {'numpy':8s} {'jax':10s} {'torch':26s} description")
        for s in FUNCTIONS.values():
            print(
                f"{s.name:11s} {s.arity:<4d} {s.scipy or '-':8s} {s.numpy or '-':8s} {s.jax or '-':10s} {s.torch or '-':26s} {s.description}"
            )
        return 0
    if args.cmd == "report":
        from .report import build_report

        build_report(args.results, args.out)
        print(f"wrote {args.out}")
        return 0
    from .run import run_audit

    functions = args.functions.split(",") if args.functions else None
    unknown = [f for f in functions or [] if f not in FUNCTIONS]
    if unknown:
        print(f"unknown functions: {unknown}", file=sys.stderr)
        return 2
    libraries = args.libraries.split(",") if args.libraries else None
    dtypes = (np.float32,) if args.only_float32 else ((np.float64, np.float32) if args.float32 else (np.float64,))
    run_audit(functions, libraries, dtypes, n=args.n, seed=args.seed, out_dir=args.out, verbose=not args.quiet)
    print(f"wrote {args.out}/results.json", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
