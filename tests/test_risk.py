"""Tail risk, risk decomposition and benchmark statistics."""
from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from prr import risk


def series(values, start="2024-01-01"):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


def test_historical_var_and_cvar_on_a_known_sample():
    # 100 returns: -10%, -9%, ..., -1%, then ninety small positives.
    r = series([-0.01 * k for k in range(10, 0, -1)] + [0.001] * 90)
    assert risk.historical_var(r, 0.95) == pytest.approx(0.0505, abs=1e-6)  # 5th percentile interpolates -5%/-6%
    assert risk.historical_var(r, 0.99) == pytest.approx(0.0901, abs=1e-6)  # 1st percentile interpolates -10%/-9%
    cvar = risk.historical_cvar(r, 0.95)
    assert cvar == pytest.approx(np.mean([0.10, 0.09, 0.08, 0.07, 0.06]), abs=0.005)
    assert cvar > risk.historical_var(r, 0.95)


def test_parametric_var_matches_normal_formula():
    r = series(np.random.default_rng(1).normal(0.0004, 0.012, 500))
    z = NormalDist().inv_cdf(0.05)
    assert risk.parametric_var(r, 0.95) == pytest.approx(-(r.mean() + z * r.std(ddof=1)))


def test_cornish_fisher_reduces_to_parametric_for_symmetric_light_tails():
    # Symmetric two-point distribution has zero skew; its excess kurtosis is -2,
    # so CF differs. Use a series with skew and kurtosis both ~0 instead: a
    # large normal sample.
    r = series(np.random.default_rng(2).normal(0, 0.01, 200_000))
    assert risk.cornish_fisher_var(r, 0.99) == pytest.approx(risk.parametric_var(r, 0.99), rel=0.02)


def test_negative_skew_raises_cornish_fisher_var_above_parametric():
    rng = np.random.default_rng(3)
    r = series(np.where(rng.uniform(size=5000) < 0.05, -0.05, 0.003))   # occasional large losses
    assert r.skew() < 0
    assert risk.cornish_fisher_var(r, 0.99) > risk.parametric_var(r, 0.99)


def test_level_validation():
    r = series([0.0, 0.1])
    with pytest.raises(ValueError):
        risk.historical_var(r, 0.05)


def test_risk_contributions_sum_to_portfolio_volatility_and_split_equal_uncorrelated_assets():
    cov = pd.DataFrame([[0.04, 0.0], [0.0, 0.04]], index=["a", "b"], columns=["a", "b"])
    w = pd.Series({"a": 0.5, "b": 0.5})
    rc = risk.risk_contributions(w, cov)
    sigma_p = np.sqrt(0.25 * 0.04 + 0.25 * 0.04)
    assert rc.component.sum() == pytest.approx(sigma_p)
    assert rc.share.tolist() == pytest.approx([0.5, 0.5])
    assert risk.portfolio_volatility(w, cov) == pytest.approx(sigma_p)
    # Two uncorrelated equal-vol assets: diversification ratio is sqrt(2).
    assert risk.diversification_ratio(w, cov) == pytest.approx(np.sqrt(2))


def test_risk_contributions_follow_correlation():
    # Perfectly correlated assets: no diversification, shares equal weights x vol proportion.
    cov = pd.DataFrame([[0.04, 0.02], [0.02, 0.01]], index=["a", "b"], columns=["a", "b"])  # vols 0.2, 0.1, rho 1
    w = pd.Series({"a": 0.5, "b": 0.5})
    rc = risk.risk_contributions(w, cov)
    assert risk.diversification_ratio(w, cov) == pytest.approx(1.0)
    assert rc.share.tolist() == pytest.approx([2 / 3, 1 / 3])


def test_beta_alpha_tracking_and_capture_on_scaled_benchmark():
    rng = np.random.default_rng(4)
    b = series(rng.normal(0.0005, 0.01, 400))
    p = 2 * b                                         # exactly twice the benchmark
    assert risk.beta(p, b) == pytest.approx(2.0)
    assert risk.jensen_alpha(p, b, 0.0) == pytest.approx(0.0, abs=1e-12)
    assert risk.tracking_error(p, b) == pytest.approx(b.std(ddof=1) * np.sqrt(252))
    up, down = risk.capture_ratios(p, b)
    assert up == pytest.approx(2.0) and down == pytest.approx(2.0)
    assert np.isnan(risk.beta(series([0.01] * 10), series([0.0] * 10)))


def test_information_ratio_zero_when_identical():
    r = series([0.01, -0.02, 0.03])
    assert np.isnan(risk.information_ratio(r, r))     # active std is 0


def test_tail_summary_shape():
    r = series(np.random.default_rng(5).normal(0, 0.01, 300))
    t = risk.tail_summary(r)
    assert list(t.index) == [0.95, 0.99]
    assert (t.loc[0.99] >= t.loc[0.95]).all()
