"""Data sources without the network, the bond proxy, and the rendered report."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from prr import cli, config, data, report


# ---- FRED parsing ------------------------------------------------------------

def test_parse_fred_csv_drops_missing_markers():
    text = "observation_date,SP500\n2024-01-02,4700.5\n2024-01-03,.\n2024-01-04,4710.0\n"
    s = data.parse_fred_csv(text, "SP500")
    assert s.tolist() == [4700.5, 4710.0]
    assert s.index[1] == pd.Timestamp("2024-01-04") and s.name == "SP500"


class FakeResponse:
    def __init__(self, status, text=""):
        self.status_code, self.text = status, text


class FakeSession:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append((url, params, headers))
        return self.responses.pop(0)


def test_fred_client_caches_and_reports_404(tmp_path):
    session = FakeSession([FakeResponse(200, "observation_date,X\n2024-01-02,1\n2024-01-03,2\n"), FakeResponse(404)])
    client = data.FredClient(cache_dir=tmp_path, session=session)
    first = client.fetch("X", log=lambda *_: None)
    second = client.fetch("X", log=lambda *_: None)          # served from cache, no second call
    assert first.tolist() == second.tolist() == [1.0, 2.0]
    assert len(session.calls) == 1
    assert session.calls[0][2]["User-Agent"].startswith("portfolio-risk-report")
    with pytest.raises(data.DataError, match="404"):
        client.fetch("GONE", log=lambda *_: None)


# ---- bond proxy --------------------------------------------------------------

def test_par_bond_prices_at_par_and_moves_inversely():
    assert data.par_bond_price(0.04, 0.04, 10) == pytest.approx(1.0)
    assert data.par_bond_price(0.04, 0.05, 10) < 1.0 < data.par_bond_price(0.04, 0.03, 10)


def test_bond_index_earns_exactly_the_carry_when_yields_are_flat():
    y = pd.Series([4.0] * 253, index=pd.bdate_range("2024-01-01", periods=253), name="DGS10")
    idx = data.bond_total_return_index(y, maturity=10)
    daily = idx.pct_change().dropna()
    assert daily.round(12).nunique() == 1
    assert daily.iloc[0] == pytest.approx(0.04 / 252)


def test_bond_index_falls_when_yields_rise():
    y = pd.Series([4.0, 4.0, 4.5], index=pd.bdate_range("2024-01-01", periods=3), name="DGS10")
    idx = data.bond_total_return_index(y, maturity=10)
    # 50bp rise on a ~8-year-duration bond: about -4% plus a day of carry.
    assert idx.iloc[2] / idx.iloc[1] - 1 == pytest.approx(-0.039, abs=0.004)


# ---- alignment and synthetic -------------------------------------------------

def test_align_inner_joins_and_drops_weekends():
    a = pd.Series([1, 2, 3, 4], index=pd.to_datetime(["2024-01-05", "2024-01-06", "2024-01-08", "2024-01-09"]))
    b = pd.Series([1, 2, 3], index=pd.to_datetime(["2024-01-05", "2024-01-08", "2024-01-10"]))
    with pytest.raises(data.DataError, match="common trading days"):
        data.align({"a": a, "b": b})
    long_a = pd.Series(np.arange(60.0), index=pd.bdate_range("2024-01-01", periods=60))
    long_b = long_a * 2
    panel = data.align({"a": long_a, "b": long_b}, start="2024-02-01")
    assert panel.index[0] >= pd.Timestamp("2024-02-01") and list(panel.columns) == ["a", "b"]


def test_synthetic_prices_are_deterministic_and_positive():
    p1, p2 = data.synthetic_prices(seed=7), data.synthetic_prices(seed=7)
    pd.testing.assert_frame_equal(p1, p2)
    assert (p1 > 0).all().all() and p1.shape == (1500, 5)
    assert not p1.equals(data.synthetic_prices(seed=8))


def test_csv_source_needs_every_asset(tmp_path):
    csv = tmp_path / "p.csv"
    pd.DataFrame({"date": pd.bdate_range("2024-01-01", periods=40), "a": np.linspace(100, 110, 40)}).to_csv(csv, index=False)
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"source": "csv", "csv_path": str(csv), "assets": {"a": {"weight": 1}, "b": {"weight": 1}}}))
    with pytest.raises(data.DataError, match="lacks columns"):
        data.build_price_panel(config.load_portfolio(cfg))


# ---- config ------------------------------------------------------------------

def test_config_validation(tmp_path):
    def write(obj):
        p = tmp_path / "c.json"
        p.write_text(json.dumps(obj))
        return p
    good = write({"source": "fred", "benchmark": "X", "assets": {"X": {"weight": 0.6}, "Y": {"weight": 0.4, "transform": "bond_total_return"}}})
    pf = config.load_portfolio(good)
    assert pf.weights().tolist() == [0.6, 0.4] and pf.assets[1].transform == "bond_total_return"
    for bad, msg in [
        ({"source": "ftp", "assets": {"X": {"weight": 1}}}, "source"),
        ({"assets": {}}, "non-empty"),
        ({"assets": {"X": {"weight": -1}}}, "non-negative"),
        ({"assets": {"X": {"weight": 1}}, "benchmark": "Z"}, "benchmark"),
        ({"assets": {"X": {"weight": 1, "transform": "nope"}}}, "transform"),
        ({"source": "csv", "assets": {"X": {"weight": 1}}}, "csv_path"),
    ]:
        with pytest.raises(config.ConfigError, match=msg):
            config.load_portfolio(write(bad))


# ---- end to end on synthetic data --------------------------------------------

@pytest.fixture(scope="module")
def sample():
    pf = data.synthetic_portfolio()
    return report.compute(pf, data.build_price_panel(pf))


def test_report_numbers_are_internally_consistent(sample):
    c = sample.contributions
    assert c.component.sum() == pytest.approx(sample.summary["annualized_volatility"], rel=0.02)
    assert c.share.sum() == pytest.approx(1.0)
    assert sample.mdd.depth < -0.10                                  # the engineered crash is visible
    assert sample.tails.loc[0.99, "historical_var"] > sample.tails.loc[0.95, "historical_var"]
    assert sample.relative is not None and 0 < sample.relative["beta"] < 1   # 35% in the benchmark plus correlated sleeve


def test_html_report_is_self_contained(sample):
    html = report.render_html(sample)
    assert html.startswith("<!doctype html>") and "<svg" in html and "</html>" in html
    assert "http" not in html.split("</style>")[1].split("<h2>Performance")[0]   # no external assets in the chart block
    md = report.render_markdown(sample)
    assert "### Risk contribution" in md and "Max drawdown" in md


def test_cli_sample_writes_files(tmp_path):
    assert cli.main(["sample", "--out", str(tmp_path)]) == 0
    assert (tmp_path / "sample_report.html").stat().st_size > 20_000
    assert (tmp_path / "sample_summary.md").exists()
