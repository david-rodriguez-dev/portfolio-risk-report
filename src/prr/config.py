"""Portfolio definition file and project paths."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "portfolio.json"

VALID_SOURCES = ("fred", "csv", "synthetic")
VALID_TRANSFORMS = (None, "bond_total_return")


class ConfigError(RuntimeError):
    """Malformed portfolio definition."""


@dataclass(frozen=True)
class Asset:
    id: str
    weight: float
    label: str
    transform: str | None = None
    maturity: float | None = None


@dataclass(frozen=True)
class Portfolio:
    source: str
    assets: tuple[Asset, ...]
    risk_free_rate: float = 0.0
    rebalance: str = "daily"
    benchmark: str | None = None
    start: str | None = None
    end: str | None = None
    csv_path: str | None = None
    name: str = "Portfolio"

    def weights(self) -> pd.Series:
        return pd.Series({a.id: a.weight for a in self.assets}, dtype=float)

    def labels(self) -> dict[str, str]:
        return {a.id: a.label for a in self.assets}


def load_portfolio(path: Path = DEFAULT_CONFIG) -> Portfolio:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"portfolio file not found: {path}") from exc
    source = raw.get("source", "fred")
    if source not in VALID_SOURCES:
        raise ConfigError(f"source must be one of {VALID_SOURCES}, got {source!r}")
    assets_raw = raw.get("assets")
    if not isinstance(assets_raw, dict) or not assets_raw:
        raise ConfigError("'assets' must be a non-empty object of id -> {weight, label}")
    assets = []
    for asset_id, spec in assets_raw.items():
        if not isinstance(spec, dict) or "weight" not in spec:
            raise ConfigError(f"asset {asset_id!r} needs a 'weight'")
        transform = spec.get("transform")
        if transform not in VALID_TRANSFORMS:
            raise ConfigError(f"asset {asset_id!r}: unknown transform {transform!r}")
        weight = float(spec["weight"])
        if weight < 0:
            raise ConfigError(f"asset {asset_id!r}: weight must be non-negative")
        assets.append(Asset(asset_id, weight, spec.get("label", asset_id), transform, spec.get("maturity")))
    if sum(a.weight for a in assets) <= 0:
        raise ConfigError("weights must sum to a positive number")
    benchmark = raw.get("benchmark")
    if benchmark is not None and benchmark not in assets_raw:
        raise ConfigError(f"benchmark {benchmark!r} is not one of the assets")
    rebalance = raw.get("rebalance", "daily")
    if rebalance not in ("daily", "never"):
        raise ConfigError("rebalance must be 'daily' or 'never'")
    if source == "csv" and not raw.get("csv_path"):
        raise ConfigError("source 'csv' needs 'csv_path'")
    return Portfolio(
        source=source,
        assets=tuple(assets),
        risk_free_rate=float(raw.get("risk_free_rate", 0.0)),
        rebalance=rebalance,
        benchmark=benchmark,
        start=raw.get("start"),
        end=raw.get("end"),
        csv_path=raw.get("csv_path"),
        name=raw.get("name", "Portfolio"),
    )
