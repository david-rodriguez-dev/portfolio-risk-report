"""Price panels from three sources: FRED, a CSV file, or a synthetic generator.

Whatever the source, the result is one wide ``DataFrame`` of daily levels,
one column per asset, on the dates where *every* asset printed a value.
Inner-joining rather than forward-filling means a market holiday in one
series drops the day for all of them; the alternative, stale prices with a
zero return, quietly deflates volatility and correlation.
"""
from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from prr import config

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"


class DataError(RuntimeError):
    """Source unavailable or unusable."""


# ---- FRED --------------------------------------------------------------------

@dataclass
class FredClient:
    """Fetch one series as CSV, cached on disk. FRED needs no key for this endpoint."""

    cache_dir: Path = config.DATA_DIR
    max_age_days: float = 1.0
    user_agent: str = "portfolio-risk-report/0.1"
    timeout: float = 60.0
    session: requests.Session = field(default_factory=requests.Session)

    def fetch(self, series_id: str, *, force: bool = False, log=print) -> pd.Series:
        path = self.cache_dir / f"fred_{series_id}.csv"
        if not force and path.exists() and (time.time() - path.stat().st_mtime) < self.max_age_days * 86400:
            text = path.read_text(encoding="utf-8")
            log(f"[fetch] {series_id:<14} cached")
        else:
            resp = self.session.get(FRED_CSV_URL, params={"id": series_id},
                                    headers={"User-Agent": self.user_agent}, timeout=self.timeout)
            if resp.status_code == 404:
                raise DataError(f"FRED has no series {series_id!r} (discontinued series return 404)")
            if resp.status_code != 200:
                raise DataError(f"FRED returned HTTP {resp.status_code} for {series_id}")
            text = resp.text
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            log(f"[fetch] {series_id:<14} downloaded")
        return parse_fred_csv(text, series_id)


def parse_fred_csv(text: str, series_id: str) -> pd.Series:
    """FRED marks missing observations with '.', which becomes NaN and is dropped."""
    frame = pd.read_csv(io.StringIO(text), na_values=["."])
    if frame.shape[1] != 2:
        raise DataError(f"unexpected FRED layout for {series_id}: columns {list(frame.columns)}")
    date_col, value_col = frame.columns
    s = pd.Series(frame[value_col].to_numpy(dtype=float), index=pd.to_datetime(frame[date_col]), name=series_id)
    return s.dropna()


# ---- transforms --------------------------------------------------------------

def par_bond_price(coupon: float, yield_: float, maturity: float, freq: int = 2) -> float:
    """Price per 1 of face of a bond paying ``coupon`` (annual rate) ``freq`` times a year."""
    n = maturity * freq
    y = yield_ / freq
    if y == 0:
        return 1 + coupon * maturity
    discount = (1 + y) ** -n
    return (coupon / freq) * (1 - discount) / y + discount


def bond_total_return_index(yields_pct: pd.Series, maturity: float = 10.0, periods_per_year: int = 252) -> pd.Series:
    """Constant-maturity total-return index from a yield series (in percent).

    Each day the holder owns a par bond issued at yesterday's yield, earns
    one day of that coupon, and marks it at today's yield. The one-day roll
    down the curve is ignored; for a 10-year bond it is negligible.
    Verified in tests: flat yields earn exactly the carry, rising yields lose.
    """
    y = yields_pct.dropna() / 100.0
    prev = y.shift(1)
    price = np.array([par_bond_price(c, yt, maturity) for c, yt in zip(prev.to_numpy(), y.to_numpy())])
    daily = pd.Series(price - 1 + prev.to_numpy() / periods_per_year, index=y.index)
    daily.iloc[0] = 0.0
    return 100.0 * (1 + daily).cumprod().rename(yields_pct.name)


# ---- CSV ---------------------------------------------------------------------

def load_prices_csv(path: Path) -> pd.DataFrame:
    """Wide CSV: first column a date, one column per asset, levels not returns."""
    frame = pd.read_csv(path)
    if frame.shape[1] < 2:
        raise DataError(f"{path}: need a date column and at least one price column")
    date_col = frame.columns[0]
    frame[date_col] = pd.to_datetime(frame[date_col])
    return frame.set_index(date_col).sort_index().astype(float)


# ---- synthetic ---------------------------------------------------------------

SYNTHETIC_SPEC = {
    # label: (annual drift, annual vol)
    "Equity A": (0.08, 0.18),
    "Equity B": (0.10, 0.25),
    "Bond": (0.03, 0.06),
    "Commodity": (0.02, 0.30),
    "FX": (0.00, 0.08),
}
SYNTHETIC_CORR = np.array([
    [1.00, 0.80, -0.20, 0.30, 0.10],
    [0.80, 1.00, -0.25, 0.30, 0.10],
    [-0.20, -0.25, 1.00, -0.10, 0.05],
    [0.30, 0.30, -0.10, 1.00, 0.15],
    [0.10, 0.10, 0.05, 0.15, 1.00],
])


def synthetic_prices(n_days: int = 1500, seed: int = 7, start: str = "2019-01-01") -> pd.DataFrame:
    """Correlated geometric Brownian motion, deterministic for a seed, with one
    engineered crash so drawdown and tail statistics have something to find."""
    rng = np.random.default_rng(seed)
    labels = list(SYNTHETIC_SPEC)
    mu = np.array([SYNTHETIC_SPEC[k][0] for k in labels])
    sigma = np.array([SYNTHETIC_SPEC[k][1] for k in labels])
    chol = np.linalg.cholesky(SYNTHETIC_CORR)
    z = rng.standard_normal((n_days, len(labels))) @ chol.T
    dt = 1 / 252
    daily = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    # a 20-day drawdown of ~-30% in the equities, ~-8% in the bond sleeve (flight to quality inverted)
    crash = slice(n_days // 3, n_days // 3 + 20)
    daily[crash, 0] -= 0.018
    daily[crash, 1] -= 0.022
    daily[crash, 3] -= 0.010
    dates = pd.bdate_range(start, periods=n_days)
    return pd.DataFrame(100.0 * np.exp(np.cumsum(daily, axis=0)), index=dates, columns=labels)


def synthetic_portfolio() -> config.Portfolio:
    return config.Portfolio(
        source="synthetic",
        assets=(
            config.Asset("Equity A", 0.35, "Equity A"),
            config.Asset("Equity B", 0.20, "Equity B"),
            config.Asset("Bond", 0.30, "Bond"),
            config.Asset("Commodity", 0.10, "Commodity"),
            config.Asset("FX", 0.05, "FX"),
        ),
        risk_free_rate=0.03,
        benchmark="Equity A",
        name="Synthetic sample portfolio",
    )


# ---- assembly ----------------------------------------------------------------

def align(series: dict[str, pd.Series], start: str | None = None, end: str | None = None) -> pd.DataFrame:
    panel = pd.concat(series.values(), axis=1, join="inner", keys=list(series))
    panel.columns = list(series)
    if start:
        panel = panel.loc[pd.Timestamp(start):]
    if end:
        panel = panel.loc[:pd.Timestamp(end)]
    panel = panel[panel.index.dayofweek < 5]
    if len(panel) < 30:
        raise DataError(f"only {len(panel)} common trading days after alignment; check dates and sources")
    return panel


def build_price_panel(portfolio: config.Portfolio, client: FredClient | None = None, log=print) -> pd.DataFrame:
    if portfolio.source == "synthetic":
        raw = {a.id: synthetic_prices()[a.id] for a in portfolio.assets}
    elif portfolio.source == "csv":
        frame = load_prices_csv(Path(portfolio.csv_path))
        missing = [a.id for a in portfolio.assets if a.id not in frame.columns]
        if missing:
            raise DataError(f"{portfolio.csv_path} lacks columns {missing}")
        raw = {a.id: frame[a.id] for a in portfolio.assets}
    else:
        client = client or FredClient()
        raw = {a.id: client.fetch(a.id, log=log) for a in portfolio.assets}

    levels: dict[str, pd.Series] = {}
    for asset in portfolio.assets:
        s = raw[asset.id]
        if asset.transform == "bond_total_return":
            s = bond_total_return_index(s, maturity=float(asset.maturity or 10.0))
        levels[asset.id] = s
    return align(levels, portfolio.start, portfolio.end)
