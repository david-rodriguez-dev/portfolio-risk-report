"""Performance and drawdown metrics on a single return series.

Every function takes a ``pd.Series`` of periodic simple returns and, where
annualisation is involved, an explicit ``periods_per_year``. The risk-free
rate is an annual figure and is converted to a per-period rate with
compounding, not division, so a 4 % rate is 4 % over a year of daily periods.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from prr.returns import drawdowns, growth_index

TRADING_DAYS = 252
ZERO_VOL = 1e-12   # a dispersion below this is floating-point noise, not volatility


def per_period_rate(annual_rate: float, periods_per_year: int) -> float:
    return (1 + annual_rate) ** (1 / periods_per_year) - 1


def total_return(r: pd.Series) -> float:
    return float((1 + r).prod() - 1)


def annualized_return(r: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Geometric: the constant annual rate that reproduces the total return."""
    n = len(r)
    if n == 0:
        return float("nan")
    return float((1 + r).prod() ** (periods_per_year / n) - 1)


def annualized_volatility(r: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    return float(r.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(r: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = TRADING_DAYS) -> float:
    """Mean excess return over its own volatility, annualised by sqrt(periods)."""
    excess = r - per_period_rate(risk_free_rate, periods_per_year)
    sd = excess.std(ddof=1)
    if not sd > ZERO_VOL:            # also catches NaN
        return float("nan")
    return float(excess.mean() / sd * np.sqrt(periods_per_year))


def downside_deviation(r: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = TRADING_DAYS) -> float:
    """Root mean square of shortfall below the risk-free rate, annualised."""
    shortfall = np.minimum(r - per_period_rate(risk_free_rate, periods_per_year), 0.0)
    return float(np.sqrt(np.mean(shortfall**2)) * np.sqrt(periods_per_year))


def sortino_ratio(r: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = TRADING_DAYS) -> float:
    dd = downside_deviation(r, risk_free_rate, periods_per_year)
    if not dd > ZERO_VOL:
        return float("nan")
    return float((annualized_return(r, periods_per_year) - risk_free_rate) / dd)


@dataclass(frozen=True)
class MaxDrawdown:
    depth: float                      # negative fraction, e.g. -0.34
    peak: pd.Timestamp
    trough: pd.Timestamp
    recovery: pd.Timestamp | None     # None if not yet recovered
    days_to_trough: int
    days_to_recover: int | None


def max_drawdown(r: pd.Series) -> MaxDrawdown:
    dd = drawdowns(r)
    if dd.empty:
        raise ValueError("empty return series")
    trough = dd.idxmin()
    depth = float(dd.loc[trough])
    g = growth_index(r)
    # The peak is the last new high before the trough.
    peak = g.loc[:trough].idxmax()
    after = g.loc[trough:]
    recovered = after[after >= g.loc[peak]]
    recovery = recovered.index[0] if not recovered.empty else None
    return MaxDrawdown(
        depth=depth,
        peak=peak,
        trough=trough,
        recovery=recovery,
        days_to_trough=int((trough - peak).days),
        days_to_recover=int((recovery - trough).days) if recovery is not None else None,
    )


def calmar_ratio(r: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    mdd = max_drawdown(r).depth
    if mdd == 0:
        return float("nan")
    return float(annualized_return(r, periods_per_year) / abs(mdd))


def skewness(r: pd.Series) -> float:
    return float(r.skew())


def excess_kurtosis(r: pd.Series) -> float:
    return float(r.kurt())


def rolling_volatility(r: pd.Series, window: int = 63, periods_per_year: int = TRADING_DAYS) -> pd.Series:
    return r.rolling(window).std(ddof=1) * np.sqrt(periods_per_year)


def monthly_returns(r: pd.Series) -> pd.DataFrame:
    """Calendar table: rows are years, columns 1..12, values compounded monthly returns."""
    monthly = (1 + r).resample("ME").prod() - 1
    table = monthly.to_frame("ret")
    table["year"] = table.index.year
    table["month"] = table.index.month
    return table.pivot(index="year", columns="month", values="ret")


def summary(r: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = TRADING_DAYS) -> dict[str, float]:
    mdd = max_drawdown(r)
    return {
        "periods": float(len(r)),
        "total_return": total_return(r),
        "annualized_return": annualized_return(r, periods_per_year),
        "annualized_volatility": annualized_volatility(r, periods_per_year),
        "sharpe": sharpe_ratio(r, risk_free_rate, periods_per_year),
        "sortino": sortino_ratio(r, risk_free_rate, periods_per_year),
        "max_drawdown": mdd.depth,
        "calmar": calmar_ratio(r, periods_per_year),
        "skew": skewness(r),
        "excess_kurtosis": excess_kurtosis(r),
        "best_period": float(r.max()),
        "worst_period": float(r.min()),
        "positive_share": float((r > 0).mean()),
    }
