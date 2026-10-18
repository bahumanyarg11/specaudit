"""Library adapters: resolve a FunctionSpec to a callable ``f(*float64 arrays) -> float64 array``
for SciPy, NumPy, JAX and PyTorch, in the requested dtype, with the library version recorded."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .functions import FunctionSpec

LIBRARIES = ("scipy", "numpy", "jax", "torch")


@dataclass
class Library:
    name: str
    version: str
    module: object
    available: bool
    reason: str = ""


def load_library(name: str) -> Library:
    try:
        if name == "scipy":
            import scipy
            import scipy.special as mod

            return Library(name, scipy.__version__, mod, True)
        if name == "numpy":
            import numpy as mod

            return Library(name, np.__version__, mod, True)
        if name == "jax":
            import jax

            jax.config.update("jax_enable_x64", True)
            import jax.scipy.special as mod

            return Library(name, jax.__version__, mod, True)
        if name == "torch":
            import torch

            return Library(name, torch.__version__, torch.special, True)
    except Exception as exc:  # noqa: BLE001
        return Library(name, "", None, False, f"{type(exc).__name__}: {exc}")
    raise ValueError(name)


def _resolve(lib: Library, attr: str):
    if ":" in attr:  # explicit module path, e.g. "torch:lgamma" or "jax.numpy:sinc"
        modname, fn = attr.split(":")
        return getattr(importlib.import_module(modname), fn)
    return getattr(lib.module, attr)


def adapter(lib: Library, spec: FunctionSpec, dtype=np.float64) -> Callable[..., np.ndarray] | None:
    """Return a callable evaluating ``spec`` in ``lib`` at ``dtype``, or None if not provided."""
    attr = getattr(spec, lib.name)
    if attr is None or not lib.available:
        return None
    try:
        fn = _resolve(lib, attr)
    except AttributeError:
        return None
    npdtype = np.dtype(dtype)

    if lib.name in ("scipy", "numpy"):

        def call(*args: np.ndarray) -> np.ndarray:
            cast = [np.asarray(a).astype(npdtype) for a in args]
            if spec.name == "polygamma" and lib.name == "scipy":
                cast[0] = np.asarray(args[0]).astype(np.int64)
            with np.errstate(all="ignore"):
                return np.asarray(fn(*cast), dtype=np.float64)

        return call

    if lib.name == "jax":
        import jax.numpy as jnp

        def call(*args: np.ndarray) -> np.ndarray:
            cast = [jnp.asarray(np.asarray(a).astype(npdtype)) for a in args]
            if spec.name == "polygamma":
                cast[0] = jnp.asarray(np.asarray(args[0]).astype(np.int32))
            return np.asarray(fn(*cast), dtype=np.float64)

        return call

    if lib.name == "torch":
        import torch

        tdtype = {np.dtype(np.float64): torch.float64, np.dtype(np.float32): torch.float32}[npdtype]

        def call(*args: np.ndarray) -> np.ndarray:
            cast = [torch.from_numpy(np.ascontiguousarray(np.asarray(a).astype(npdtype))).to(tdtype) for a in args]
            if spec.name == "polygamma":
                # torch.special.polygamma(n: int, x): n must be a scalar -> loop over orders
                n = np.asarray(args[0]).astype(int)
                out = np.empty(n.shape, dtype=np.float64)
                for order in np.unique(n):
                    m = n == order
                    out[m] = fn(int(order), cast[1][torch.from_numpy(m)]).numpy()
                return out
            return fn(*cast).to(torch.float64).numpy()

        return call

    return None
