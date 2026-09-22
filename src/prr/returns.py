"""From prices to return series.

Conventions used throughout the package:

* Prices and returns are pandas objects indexed by a DatetimeIndex, one row
  per trading day, one column per asset.
* Returns are simple (arithmetic) unless a function says otherwise; the
  portfolio return of a set of weights is the weighted sum of simple
  returns, which is exact for a daily-rebalanced book.
* Nothing here annualises. That happens in ``metrics`` with an explicit
  periods-per-year, so the same code serves daily or monthly data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def simple_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """P_t / P_{t-1} - 1, first row dropped."""
    return prices.pct_change().iloc[1:]


def log_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    return np.log(prices).diff().iloc[1:]


def normalise_weights(weights: pd.Series) -> pd.Series:
    """Weights as a Series summing to one; refuses negatives and empties."""
    w = pd.Series(weights, dtype=float)
    if w.empty or (w < 0).any():
        raise ValueError("weights must be non-empty and non-negative")
    total = w.sum()
    if total <= 0:
        raise ValueError("weights must sum to a positive number")
    return w / total


def portfolio_returns(
    returns: pd.DataFrame,
    weights: pd.Series,
    rebalance: str = "daily",
) -> pd.Series:
    """Portfolio return series for a set of target weights.

    ``rebalance="daily"``  the book is reset to target weights every day, so
                           r_p = sum_i w_i r_i exactly.
    ``rebalance="never"``  buy and hold: each sleeve compounds on its own and
                           the weights drift with performance.
    """
    w = normalise_weights(weights).reindex(returns.columns)
    if w.isna().any():
        missing = list(w[w.isna()].index)
        raise ValueError(f"no weight for {missing}")
    if rebalance == "daily":
        out = returns.mul(w, axis=1).sum(axis=1)
    elif rebalance == "never":
        sleeve_value = (1 + returns).cumprod().mul(w, axis=1)
        total = sleeve_value.sum(axis=1)
        out = total / total.shift(1).fillna(1.0) - 1
    else:
        raise ValueError(f"unknown rebalance mode {rebalance!r}")
    out.name = "portfolio"
    return out


def growth_index(returns: pd.Series | pd.DataFrame, start: float = 1.0) -> pd.Series | pd.DataFrame:
    """Value of ``start`` invested at the beginning, compounding the returns."""
    return start * (1 + returns).cumprod()


def drawdowns(returns: pd.Series) -> pd.Series:
    """Fraction below the running peak, zero at every new high, always <= 0."""
    g = growth_index(returns)
    return g / g.cummax() - 1
