"""Compute everything once, then render it as HTML or Markdown."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape

import pandas as pd

from prr import metrics, risk, svg
from prr.config import Portfolio
from prr.returns import drawdowns, growth_index, portfolio_returns, simple_returns

TRADING_DAYS = 252


@dataclass
class ReportData:
    portfolio: Portfolio
    prices: pd.DataFrame
    asset_returns: pd.DataFrame
    port: pd.Series
    bench: pd.Series | None
    summary: dict[str, float]
    bench_summary: dict[str, float] | None
    tails: pd.DataFrame
    contributions: pd.DataFrame
    correlation: pd.DataFrame
    mdd: metrics.MaxDrawdown
    monthly: pd.DataFrame
    rolling_vol: pd.Series
    relative: dict[str, float] | None
    worst_days: pd.Series


def compute(portfolio: Portfolio, prices: pd.DataFrame) -> ReportData:
    rf, ppy = portfolio.risk_free_rate, TRADING_DAYS
    asset_returns = simple_returns(prices)
    weights = portfolio.weights()
    port = portfolio_returns(asset_returns, weights, portfolio.rebalance)
    bench = asset_returns[portfolio.benchmark] if portfolio.benchmark else None

    cov = risk.annualized_covariance(asset_returns, ppy)
    contributions = risk.risk_contributions(weights / weights.sum(), cov)
    contributions["diversification_ratio"] = risk.diversification_ratio(weights, cov)

    relative = None
    if bench is not None:
        up, down = risk.capture_ratios(port, bench)
        relative = {
            "beta": risk.beta(port, bench),
            "alpha": risk.jensen_alpha(port, bench, rf, ppy),
            "correlation": float(port.corr(bench)),
            "tracking_error": risk.tracking_error(port, bench, ppy),
            "information_ratio": risk.information_ratio(port, bench, ppy),
            "up_capture": up,
            "down_capture": down,
        }

    return ReportData(
        portfolio=portfolio,
        prices=prices,
        asset_returns=asset_returns,
        port=port,
        bench=bench,
        summary=metrics.summary(port, rf, ppy),
        bench_summary=metrics.summary(bench, rf, ppy) if bench is not None else None,
        tails=risk.tail_summary(port),
        contributions=contributions,
        correlation=asset_returns.corr(),
        mdd=metrics.max_drawdown(port),
        monthly=metrics.monthly_returns(port),
        rolling_vol=metrics.rolling_volatility(port, 63, ppy).dropna(),
        relative=relative,
        worst_days=risk.worst_periods(port, 5),
    )


# ---- formatting --------------------------------------------------------------

def pct(v: float, digits: int = 1) -> str:
    return "" if v is None or pd.isna(v) else f"{v:+.{digits}%}" if v < 0 else f"{v:.{digits}%}"


def num(v: float, digits: int = 2) -> str:
    return "" if v is None or pd.isna(v) else f"{v:.{digits}f}"


SUMMARY_ROWS = [
    ("Total return", "total_return", pct),
    ("Annualised return", "annualized_return", pct),
    ("Annualised volatility", "annualized_volatility", pct),
    ("Sharpe ratio", "sharpe", num),
    ("Sortino ratio", "sortino", num),
    ("Max drawdown", "max_drawdown", pct),
    ("Calmar ratio", "calmar", num),
    ("Skewness", "skew", num),
    ("Excess kurtosis", "excess_kurtosis", num),
    ("Best day", "best_period", lambda v: pct(v, 2)),
    ("Worst day", "worst_period", lambda v: pct(v, 2)),
    ("Positive days", "positive_share", pct),
]

RELATIVE_ROWS = [
    ("Beta", "beta", num), ("Jensen alpha (annualised)", "alpha", pct), ("Correlation", "correlation", num),
    ("Tracking error", "tracking_error", pct), ("Information ratio", "information_ratio", num),
    ("Up capture", "up_capture", lambda v: pct(v, 0)), ("Down capture", "down_capture", lambda v: pct(v, 0)),
]


def _labels(data: ReportData) -> dict[str, str]:
    return data.portfolio.labels()


def _relabel(df: pd.DataFrame, labels: dict[str, str]) -> pd.DataFrame:
    return df.rename(index=labels, columns=labels)


# ---- markdown ----------------------------------------------------------------

def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def render_markdown(data: ReportData) -> str:
    p, labels = data.portfolio, _labels(data)
    bench_label = labels.get(p.benchmark, p.benchmark) if p.benchmark else None
    lines = [f"## {p.name}", "",
             f"{data.port.index[0].date()} to {data.port.index[-1].date()}, {len(data.port):,} trading days, "
             f"{p.rebalance} rebalancing, risk-free {p.risk_free_rate:.1%}.", ""]

    headers = ["metric", "portfolio"] + ([bench_label] if bench_label else [])
    rows = []
    for name, key, fmt in SUMMARY_ROWS:
        row = [name, fmt(data.summary[key])]
        if data.bench_summary:
            row.append(fmt(data.bench_summary[key]))
        rows.append(row)
    lines += ["### Performance", "", _md_table(headers, rows), ""]

    lines += ["### Tail risk (one-day loss)", "",
              _md_table(["level", "historical VaR", "historical CVaR", "parametric VaR", "Cornish-Fisher VaR"],
                        [[f"{lvl:.0%}", pct(r.historical_var, 2), pct(r.historical_cvar, 2), pct(r.parametric_var, 2), pct(r.cornish_fisher_var, 2)]
                         for lvl, r in data.tails.iterrows()]), ""]

    c = data.contributions
    lines += ["### Risk contribution", "",
              _md_table(["asset", "weight", "stand-alone vol", "marginal", "component", "share of risk"],
                        [[labels.get(i, i), pct(r.weight), pct(r.volatility), num(r.marginal, 3), pct(r.component, 2), pct(r.share)]
                         for i, r in c.iterrows()]),
              "", f"Portfolio volatility {pct(c.component.sum())} = sum of components; diversification ratio "
                  f"{num(c.diversification_ratio.iloc[0])}.", ""]

    m = data.mdd
    lines += ["### Max drawdown", "",
              f"{pct(m.depth)} from {m.peak.date()} to {m.trough.date()} ({m.days_to_trough} calendar days), "
              + (f"recovered {m.recovery.date()} after {m.days_to_recover} days." if m.recovery is not None else "not yet recovered."), ""]

    if data.relative:
        lines += [f"### Relative to {bench_label}", "",
                  _md_table(["metric", "value"], [[n, f(data.relative[k])] for n, k, f in RELATIVE_ROWS]), ""]
    return "\n".join(lines)


# ---- html --------------------------------------------------------------------

_CSS = """
:root{--ink:#222;--muted:#666;--line:#e5e5e5;--accent:#1f5fbf;--bg:#fff;--neg:#b3263a;--pos:#1c7c54}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 ui-sans-serif,system-ui,sans-serif}
main{max-width:960px;margin:0 auto;padding:32px 20px 64px}h1{font-size:26px;margin:0 0 4px}h2{font-size:18px;margin:36px 0 12px;border-bottom:1px solid var(--line);padding-bottom:6px}
.sub{color:var(--muted);margin:0 0 24px}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:6px 10px;border-bottom:1px solid var(--line);text-align:right}th:first-child,td:first-child{text-align:left}
th{color:var(--muted);font-weight:600;font-size:13px}.neg{color:var(--neg)}.pos{color:var(--pos)}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:24px}@media(max-width:720px){.grid{grid-template-columns:1fr}}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:16px 0}
.tile{border:1px solid var(--line);border-radius:8px;padding:12px 14px}.tile .k{font-size:12px;color:var(--muted)}.tile .v{font-size:22px;font-weight:600}
figure{margin:16px 0}.note{color:var(--muted);font-size:13px}.cal td{padding:4px 6px;font-size:13px}
"""


def _cls(v: float) -> str:
    return "neg" if v is not None and not pd.isna(v) and v < 0 else "pos" if v is not None and not pd.isna(v) and v > 0 else ""


def _html_table(headers: list[str], rows: list[list[tuple[str, str]]], cls: str = "") -> str:
    head = "".join(f"<th>{escape(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td class='{c}'>{escape(t)}</td>" for t, c in r) + "</tr>" for r in rows)
    return f"<table class='{cls}'><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _calendar(monthly: pd.DataFrame) -> str:
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    rows = []
    for year, row in monthly.iterrows():
        cells = [(str(year), "")]
        for m in range(1, 13):
            v = row.get(m)
            cells.append(("" if v is None or pd.isna(v) else f"{v:.1%}", _cls(v)))
        yr = (1 + row.dropna()).prod() - 1
        cells.append((f"{yr:.1%}", _cls(yr)))
        rows.append(cells)
    return _html_table(["year", *months, "year"], rows, cls="cal")


def render_html(data: ReportData) -> str:
    p, labels = data.portfolio, _labels(data)
    s, b = data.summary, data.bench_summary
    bench_label = labels.get(p.benchmark, p.benchmark) if p.benchmark else None
    growth = {p.name: growth_index(data.port, 100.0)}
    if data.bench is not None:
        growth[bench_label] = growth_index(data.bench, 100.0)

    tiles = "".join(
        f"<div class='tile'><div class='k'>{escape(k)}</div><div class='v {_cls(v) if k != 'Annualised volatility' else ''}'>{escape(f(v))}</div></div>"
        for k, v, f in [("Annualised return", s["annualized_return"], pct), ("Annualised volatility", s["annualized_volatility"], pct),
                        ("Sharpe", s["sharpe"], num), ("Max drawdown", s["max_drawdown"], pct),
                        ("95% 1-day VaR", data.tails.loc[0.95, "historical_var"], lambda v: pct(v, 2))]
    )

    perf_rows = [[(n, ""), (f(s[k]), _cls(s[k]) if k in ("total_return", "annualized_return", "max_drawdown") else "")]
                 + ([(f(b[k]), "")] if b else []) for n, k, f in SUMMARY_ROWS]
    tail_rows = [[(f"{lvl:.0%}", ""), *((pct(v, 2), "") for v in r)] for lvl, r in data.tails.iterrows()]
    c = data.contributions
    contrib_rows = [[(labels.get(i, i), ""), (pct(r.weight), ""), (pct(r.volatility), ""), (num(r.marginal, 3), ""), (pct(r.component, 2), ""), (pct(r.share), "")]
                    for i, r in c.iterrows()]
    m = data.mdd
    recovery = f"recovered on {m.recovery.date()} after {m.days_to_recover} days" if m.recovery is not None else "not yet recovered"
    worst = [[(str(d.date()), ""), (pct(v, 2), "neg")] for d, v in data.worst_days.items()]

    relative_html = ""
    if data.relative:
        relative_html = f"<h2>Relative to {escape(bench_label)}</h2>" + _html_table(
            ["metric", "value"], [[(n, ""), (f(data.relative[k]), "")] for n, k, f in RELATIVE_ROWS])

    weights = pd.Series({labels.get(a.id, a.id): a.weight for a in p.assets})
    asset_growth = {labels.get(k, k): v for k, v in growth_index(data.asset_returns, 100.0).items()}

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(p.name)} risk report</title><style>{_CSS}</style></head>
<body><main>
<h1>{escape(p.name)}</h1>
<p class="sub">{data.port.index[0].date()} to {data.port.index[-1].date()} · {len(data.port):,} trading days · {escape(p.rebalance)} rebalancing · risk-free {p.risk_free_rate:.1%} · generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC</p>
<div class="tiles">{tiles}</div>
<figure>{svg.line_chart(growth, "Growth of 100", y_fmt=lambda v: f"{v:,.0f}", log_scale=True)}</figure>
<figure>{svg.area_chart(drawdowns(data.port), "Drawdown from peak")}</figure>
<h2>Performance</h2>
{_html_table(["metric", p.name] + ([bench_label] if bench_label else []), perf_rows)}
<h2>Tail risk — one-day loss</h2>
{_html_table(["confidence", "historical VaR", "historical CVaR", "parametric VaR", "Cornish-Fisher VaR"], tail_rows)}
<p class="note">Three estimators on purpose. Historical uses the sample as is; parametric assumes normal returns; Cornish-Fisher adjusts the normal quantile for the sample's skew ({num(s["skew"])}) and excess kurtosis ({num(s["excess_kurtosis"])}). When they disagree, the tails are not normal.</p>
<h2>Where the risk comes from</h2>
{_html_table(["asset", "weight", "stand-alone vol", "marginal", "component", "share of risk"], contrib_rows)}
<p class="note">Component contributions sum to portfolio volatility ({pct(c.component.sum())}) by Euler's theorem. Diversification ratio {num(c.diversification_ratio.iloc[0])}: the weighted average of stand-alone volatilities over portfolio volatility.</p>
<div class="grid">
<figure>{svg.bar_chart(pd.Series(c.share.to_numpy(), index=[labels.get(i, i) for i in c.index]), "Share of portfolio risk", width=440)}</figure>
<figure>{svg.bar_chart(weights, "Target weights", width=440, fmt=lambda v: f"{v:.0%}")}</figure>
</div>
<figure>{svg.heatmap(_relabel(data.correlation, labels), "Correlation of daily returns")}</figure>
<h2>Drawdown and worst days</h2>
<p>Max drawdown <span class="neg">{pct(m.depth)}</span> from {m.peak.date()} to {m.trough.date()} ({m.days_to_trough} calendar days), {recovery}.</p>
<div class="grid">
<div>{_html_table(["date", "return"], worst)}</div>
<figure>{svg.line_chart({"63-day rolling volatility": data.rolling_vol}, "Rolling annualised volatility", width=440, height=260, y_fmt=lambda v: f"{v:.0%}")}</figure>
</div>
{relative_html}
<h2>Monthly returns</h2>
{_calendar(data.monthly)}
<h2>Assets</h2>
<figure>{svg.line_chart(asset_growth, "Growth of 100, each asset", y_fmt=lambda v: f"{v:,.0f}", log_scale=True)}</figure>
<p class="note">Source: {escape(p.source)}. Indices are price-only and not investable; a yield series is converted to a total-return proxy by repricing a par bond daily. Weights are targets, rebalanced {escape(p.rebalance)}.</p>
</main></body></html>
"""
