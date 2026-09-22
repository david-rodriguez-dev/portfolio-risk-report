"""Tail risk, risk decomposition and benchmark-relative statistics.

Value at risk is reported as a positive loss fraction at a confidence level:
``historical_var(r, 0.95) == 0.021`` means "on the worst 5 % of days the
loss was at least 2.1 %". Three estimators are provided because they
disagree, and the disagreement is informative: historical uses the sample
as is, parametric assumes normality, Cornish-Fisher corrects the normal
quantile for the sample's skew and kurtosis.
"""
from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd

_NORMAL = NormalDist()


def _check_level(level: float) -> None:
    if not 0.5 < level < 1.0:
        raise ValueError("confidence level must be between 0.5 and 1, e.g. 0.95")


def historical_var(r: pd.Series, level: float = 0.95) -> float:
    _check_level(level)
    return float(-np.quantile(r.to_numpy(), 1 - level))


def historical_cvar(r: pd.Series, level: float = 0.95) -> float:
    """Expected shortfall: mean loss on the days at or beyond the VaR quantile."""
    _check_level(level)
    x = r.to_numpy()
    cutoff = np.quantile(x, 1 - level)
    return float(-x[x <= cutoff].mean())


def parametric_var(r: pd.Series, level: float = 0.95) -> float:
    """Normal-distribution VaR from the sample mean and standard deviation."""
    _check_level(level)
    z = _NORMAL.inv_cdf(1 - level)
    return float(-(r.mean() + z * r.std(ddof=1)))


def cornish_fisher_var(r: pd.Series, level: float = 0.95) -> float:
    """Parametric VaR with the quantile adjusted for skewness and excess kurtosis."""
    _check_level(level)
    z = _NORMAL.inv_cdf(1 - level)
    s, k = float(r.skew()), float(r.kurt())
    z_cf = z + (z**2 - 1) * s / 6 + (z**3 - 3 * z) * k / 24 - (2 * z**3 - 5 * z) * s**2 / 36
    return float(-(r.mean() + z_cf * r.std(ddof=1)))


def annualized_covariance(returns: pd.DataFrame, periods_per_year: int = 252) -> pd.DataFrame:
    return returns.cov(ddof=1) * periods_per_year


def risk_contributions(weights: pd.Series, cov: pd.DataFrame) -> pd.DataFrame:
    """Decompose portfolio volatility by asset.

    marginal    d sigma_p / d w_i  = (Sigma w)_i / sigma_p
    component   w_i * marginal_i   these sum exactly to sigma_p (Euler)
    share       component_i / sigma_p, sums to one
    """
    w = weights.reindex(cov.index).fillna(0.0).to_numpy()
    sigma = cov.to_numpy()
    variance = float(w @ sigma @ w)
    if variance <= 0:
        raise ValueError("portfolio variance is not positive")
    sigma_p = np.sqrt(variance)
    marginal = sigma @ w / sigma_p
    component = w * marginal
    return pd.DataFrame(
        {"weight": w, "volatility": np.sqrt(np.diag(sigma)), "marginal": marginal,
         "component": component, "share": component / sigma_p},
        index=cov.index,
    )


def portfolio_volatility(weights: pd.Series, cov: pd.DataFrame) -> float:
    w = weights.reindex(cov.index).fillna(0.0).to_numpy()
    return float(np.sqrt(w @ cov.to_numpy() @ w))


def diversification_ratio(weights: pd.Series, cov: pd.DataFrame) -> float:
    """Weighted average of asset volatilities over portfolio volatility; 1 means no diversification."""
    w = weights.reindex(cov.index).fillna(0.0).to_numpy()
    stand_alone = float(w @ np.sqrt(np.diag(cov.to_numpy())))
    return stand_alone / portfolio_volatility(weights, cov)


def beta(r: pd.Series, benchmark: pd.Series) -> float:
    aligned = pd.concat([r, benchmark], axis=1, join="inner").dropna()
    x, y = aligned.iloc[:, 1], aligned.iloc[:, 0]
    var = x.var(ddof=1)
    if var == 0:
        return float("nan")
    return float(x.cov(y) / var)


def jensen_alpha(r: pd.Series, benchmark: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """Annualised intercept of the CAPM regression of excess returns."""
    rf = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    aligned = pd.concat([r, benchmark], axis=1, join="inner").dropna()
    ex_p, ex_b = aligned.iloc[:, 0] - rf, aligned.iloc[:, 1] - rf
    b = beta(ex_p, ex_b)
    return float((ex_p.mean() - b * ex_b.mean()) * periods_per_year)


def tracking_error(r: pd.Series, benchmark: pd.Series, periods_per_year: int = 252) -> float:
    active = (r - benchmark).dropna()
    return float(active.std(ddof=1) * np.sqrt(periods_per_year))


def information_ratio(r: pd.Series, benchmark: pd.Series, periods_per_year: int = 252) -> float:
    active = (r - benchmark).dropna()
    sd = active.std(ddof=1)
    if not sd > 1e-12:
        return float("nan")
    return float(active.mean() / sd * np.sqrt(periods_per_year))


def capture_ratios(r: pd.Series, benchmark: pd.Series) -> tuple[float, float]:
    """(up capture, down capture): average portfolio return over average
    benchmark return on the periods where the benchmark rose / fell.

    Averages, not compounded totals: over a decade of daily data the
    compounded return of 1,300 up-days is astronomically large and the ratio
    of two such numbers says nothing (an early version printed 5 % up
    capture for a portfolio with a beta of 0.68).
    """
    aligned = pd.concat([r, benchmark], axis=1, join="inner").dropna()
    p, b = aligned.iloc[:, 0], aligned.iloc[:, 1]

    def ratio(mask: pd.Series) -> float:
        if mask.sum() == 0 or b[mask].mean() == 0:
            return float("nan")
        return float(p[mask].mean() / b[mask].mean())

    return ratio(b > 0), ratio(b < 0)


def worst_periods(r: pd.Series, n: int = 5) -> pd.Series:
    return r.nsmallest(n)


def tail_summary(r: pd.Series, levels: tuple[float, ...] = (0.95, 0.99)) -> pd.DataFrame:
    rows = []
    for level in levels:
        rows.append({
            "level": level,
            "historical_var": historical_var(r, level),
            "historical_cvar": historical_cvar(r, level),
            "parametric_var": parametric_var(r, level),
            "cornish_fisher_var": cornish_fisher_var(r, level),
        })
    return pd.DataFrame(rows).set_index("level")
