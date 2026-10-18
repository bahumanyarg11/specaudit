import math

import numpy as np
import pytest
import scipy.special as sc

from ulpaudit.functions import FUNCTIONS, sample_domain
from ulpaudit.metrics import classify, summarise, ulp_error
from ulpaudit.references import REFERENCES, reference
from ulpaudit.testing import assert_max_ulp


def test_reference_values():
    hi, lo = reference("gamma", np.array([5.0, 0.5]))
    assert hi[0] == 24.0 and lo[0] == 0.0
    assert hi[1] == pytest.approx(math.sqrt(math.pi))
    assert abs(lo[1]) < 1e-16
    hi, _ = reference("lgamma", np.array([-1.0]))
    assert hi[0] == math.inf
    hi, _ = reference("ndtri", np.array([0.5, 0.0, 1.0, 2.0, 1e-32]))
    assert abs(hi[0]) < 1e-29 and hi[1] == -math.inf and hi[2] == math.inf and math.isnan(hi[3])
    assert hi[4] == pytest.approx(-11.856132, abs=1e-5)  # ndtri(1e-32)
    hi, _ = reference("log_ndtr", np.array([20.0]))
    assert hi[0] == pytest.approx(-2.7536241186062337e-89, rel=1e-6)
    hi, _ = reference("gammainc", np.array([1.0]), np.array([1.0]))
    assert hi[0] == pytest.approx(1 - math.exp(-1))


def test_every_function_has_a_reference():
    for name in FUNCTIONS:
        assert name in REFERENCES, name


def test_ulp_error_basics():
    ref = np.array([1.0, 1e-300, 3.0])
    got = np.array([1.0 + np.spacing(1.0), 1e-300, 3.0 - 2 * np.spacing(3.0)])
    u = ulp_error(got, ref, np.zeros(3))
    assert u == pytest.approx([1.0, 0.0, 2.0])


def test_ulp_error_uses_the_low_part():
    # reference = 1 + 0.25 ulp; a result of exactly 1.0 is 0.25 ULP off, not 0.
    hi = np.array([1.0])
    lo = np.array([0.25 * np.spacing(1.0)])
    assert ulp_error(np.array([1.0]), hi, lo)[0] == pytest.approx(0.25)


def test_float32_ulps():
    ref = np.array([1.0])
    got = np.array([np.float32(1.0) + np.spacing(np.float32(1.0))], dtype=np.float64)
    assert ulp_error(got, ref, np.zeros(1), np.float32)[0] == pytest.approx(1.0)


def test_classify():
    got = np.array([1.0, np.nan, np.inf, 2.0, -1.0, np.nan])
    ref = np.array([1.0, 1.0, 1.0, np.inf, 1.0, np.nan])
    assert list(classify(got, ref)) == [
        "ok",
        "spurious_nan",
        "spurious_inf",
        "missing_nonfinite",
        "sign_error",
        "both_nonfinite",
    ]
    s = summarise(ulp_error(got, ref, np.zeros(6)), classify(got, ref))
    assert s.n_ok == 1 and s.n_spurious_nan == 1 and s.n_spurious_inf == 1 and s.n_sign_error == 1


def test_sample_domain_shapes_and_bounds():
    rng = np.random.default_rng(0)
    for spec in FUNCTIONS.values():
        for d in spec.domains:
            x = sample_domain(d, 300, rng)
            assert x.shape == (300,)
            assert np.all(np.isfinite(x))
            if not d.negative:
                assert np.all(x >= 0)


def test_assert_max_ulp_passes_for_scipy_erf():
    x = np.linspace(-5, 5, 2001)
    s = assert_max_ulp(sc.erf, "erf", x, max_ulp=3.0)
    assert s["n"] == 2001


def test_assert_max_ulp_fails_for_a_bad_implementation():
    x = np.linspace(0.1, 3, 50)
    with pytest.raises(AssertionError, match="gamma"):
        assert_max_ulp(lambda v: sc.gamma(v) * (1 + 1e-10), "gamma", x, max_ulp=4.0)


def test_small_audit_runs(tmp_path):
    from ulpaudit.run import run_audit

    payload = run_audit(["erf", "gammainc"], ["scipy", "numpy"], (np.float64,), n=60, out_dir=tmp_path, verbose=False)
    assert (tmp_path / "results.json").exists()
    names = {(r["function"], r["library"]) for r in payload["results"]}
    assert ("erf", "scipy") in names and ("gammainc", "scipy") in names
    from ulpaudit.report import build_report

    text = build_report(tmp_path / "results.json", tmp_path / "ATLAS.md")
    assert "`erf`" in text


def test_zeta_reference_keeps_relative_accuracy_for_tiny_values():
    # mpmath.zeta(43.013, 305.42) at 30 digits is only ~1e-9 accurate (absolute E-M tolerance);
    # the direct sum and SciPy/JAX/PyTorch agree on 1.0181311132601285e-106 (float64 inputs).
    hi, lo = REFERENCES["zeta"](43.01303985093054, 305.4243263250737)
    assert abs(hi - 1.0181311132601285e-106) / 1.0181311132601285e-106 < 3e-16


def test_betaln_reference_handles_huge_argument_ratios():
    # at 30 digits a + b == b for b/a > 1e30 and mp.beta returned Gamma(a): +4.0e42 instead of -1.248e43
    hi, lo = REFERENCES["betaln"](4.329415142252114e40, 2.5198543297732718e165)
    assert abs(hi - (-1.2480920458554066e43)) / 1.2480920458554066e43 < 1e-15
    hi, lo = REFERENCES["betaln"](3.660038090358462e260, 0.006290321899976063)
    assert abs(hi - 1.2911427050257500575) < 1e-14
