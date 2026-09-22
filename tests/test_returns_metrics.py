"""Return construction and performance metrics against hand-computed values."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prr import metrics
from prr.returns import drawdowns, growth_index, normalise_weights, portfolio_returns, simple_returns


def series(values, start="2024-01-01"):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


def test_simple_returns_drop_first_row():
    prices = series([100, 110, 99])
    r = simple_returns(prices)
    assert list(r.round(10)) == [0.1, -0.1]
    assert r.index[0] == prices.index[1]


def test_weights_are_normalised_and_validated():
    assert normalise_weights(pd.Series({"a": 2, "b": 2})).tolist() == [0.5, 0.5]
    with pytest.raises(ValueError):
        normalise_weights(pd.Series({"a": -1, "b": 2}))
    with pytest.raises(ValueError):
        normalise_weights(pd.Series(dtype=float))


def test_daily_rebalanced_portfolio_is_weighted_sum():
    r = pd.DataFrame({"a": [0.10, -0.05], "b": [0.00, 0.02]}, index=pd.bdate_range("2024-01-01", periods=2))
    p = portfolio_returns(r, pd.Series({"a": 0.5, "b": 0.5}))
    assert p.round(10).tolist() == [0.05, -0.015]


def test_buy_and_hold_lets_weights_drift():
    # Day 1: a doubles, b flat. Day 2: a flat, b halves. 50/50 start.
    r = pd.DataFrame({"a": [1.0, 0.0], "b": [0.0, -0.5]}, index=pd.bdate_range("2024-01-01", periods=2))
    p = portfolio_returns(r, pd.Series({"a": 0.5, "b": 0.5}), rebalance="never")
    # After day 1 value = 1.0 + 0.5 = 1.5. Day 2: a stays 1.0, b -> 0.25, total 1.25: return -1/6.
    assert p.round(10).tolist() == [0.5, round(1.25 / 1.5 - 1, 10)]
    daily = portfolio_returns(r, pd.Series({"a": 0.5, "b": 0.5}), rebalance="daily")
    assert daily.round(10).tolist() == [0.5, -0.25]


def test_portfolio_returns_refuses_missing_weight():
    r = pd.DataFrame({"a": [0.1], "b": [0.1]}, index=pd.bdate_range("2024-01-01", periods=1))
    with pytest.raises(ValueError, match="no weight"):
        portfolio_returns(r, pd.Series({"a": 1.0}))


def test_growth_and_drawdown_of_designed_path():
    # 100 -> 120 -> 90 -> 108 -> 130: peak 120, trough 90 (-25%), recovered at 130.
    prices = series([100, 120, 90, 108, 130])
    r = simple_returns(prices)
    g = growth_index(r, 100.0)
    assert g.round(8).tolist() == [120, 90, 108, 130]
    dd = drawdowns(r)
    assert dd.round(8).tolist() == [0.0, -0.25, -0.1, 0.0]
    m = metrics.max_drawdown(r)
    assert m.depth == pytest.approx(-0.25)
    assert m.peak == prices.index[1] and m.trough == prices.index[2] and m.recovery == prices.index[4]
    assert m.days_to_trough == 1 and m.days_to_recover == 2


def test_unrecovered_drawdown_has_no_recovery_date():
    r = simple_returns(series([100, 120, 90, 95]))
    m = metrics.max_drawdown(r)
    assert m.recovery is None and m.days_to_recover is None


def test_annualized_return_reproduces_total_return():
    r = series([0.001] * 504)                       # exactly two years of a constant daily return
    total = (1.001**504) - 1
    assert metrics.total_return(r) == pytest.approx(total)
    assert metrics.annualized_return(r) == pytest.approx((1 + total) ** 0.5 - 1)


def test_constant_returns_have_zero_volatility_and_nan_sharpe():
    r = series([0.001] * 100)
    assert metrics.annualized_volatility(r) == pytest.approx(0.0, abs=1e-12)
    assert np.isnan(metrics.sharpe_ratio(r)) and np.isnan(metrics.sortino_ratio(r, 0.0))


def test_volatility_and_sharpe_formulas():
    r = series([0.01, -0.01, 0.02, -0.02, 0.0])
    sd = r.std(ddof=1)
    assert metrics.annualized_volatility(r) == pytest.approx(sd * np.sqrt(252))
    rf_daily = 1.02 ** (1 / 252) - 1
    ex = r - rf_daily
    assert metrics.sharpe_ratio(r, 0.02) == pytest.approx(ex.mean() / ex.std(ddof=1) * np.sqrt(252))


def test_downside_deviation_ignores_gains():
    r = series([0.05, -0.02, 0.03, -0.04])
    expected = np.sqrt(np.mean([0, 0.02**2, 0, 0.04**2])) * np.sqrt(252)
    assert metrics.downside_deviation(r, 0.0) == pytest.approx(expected)


def test_monthly_table_compounds_within_month():
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-02-01", "2024-02-02"])
    r = pd.Series([0.10, 0.10, -0.10, 0.10], index=idx)
    table = metrics.monthly_returns(r)
    assert table.loc[2024, 1] == pytest.approx(0.21)
    assert table.loc[2024, 2] == pytest.approx(-0.01)


def test_summary_has_every_key():
    r = series(np.random.default_rng(0).normal(0.0005, 0.01, 300))
    s = metrics.summary(r, 0.03)
    assert set(s) >= {"annualized_return", "annualized_volatility", "sharpe", "sortino", "max_drawdown", "calmar", "skew"}
    assert s["periods"] == 300
