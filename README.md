# portfolio-risk-report

Performance and risk analytics for a portfolio of daily prices — returns,
drawdowns, Sharpe/Sortino/Calmar, three flavours of value-at-risk, Euler
risk contributions, benchmark-relative statistics — rendered as one
self-contained HTML file. Numerics in numpy/pandas as pure functions, every
one of them tested against a hand-computed answer; charts drawn in inline
SVG with no plotting library.

[![ci](https://github.com/davrod-dev/portfolio-risk-report/actions/workflows/ci.yml/badge.svg)](https://github.com/davrod-dev/portfolio-risk-report/actions/workflows/ci.yml)

Data comes from FRED's keyless CSV endpoint (daily index levels), from any
wide CSV of prices you provide, or from a deterministic synthetic generator
so the whole thing runs — and is tested in CI — without a network.

## Quickstart

```bash
git clone https://github.com/davrod-dev/portfolio-risk-report
cd portfolio-risk-report
pip install -e ".[dev]"

python -m prr sample --out examples          # synthetic portfolio, no network, ~1 s
python -m prr report --markdown              # config/portfolio.json via FRED -> reports/report.html + .md
pytest                                       # 35 tests, all offline
```

Open `reports/report.html` in a browser. It has no external dependencies:
inline CSS, inline SVG, no scripts.

**See one now:** [`examples/sample_report.html`](examples/sample_report.html)
is the synthetic sample committed to the repo
([rendered via htmlpreview](https://htmlpreview.github.io/?https://github.com/davrod-dev/portfolio-risk-report/blob/main/examples/sample_report.html)).

## What the report contains

| section | what is computed |
|---|---|
| headline tiles | annualised return and volatility, Sharpe, max drawdown, 95 % one-day historical VaR |
| growth of 100 | portfolio vs benchmark, log scale; drawdown-from-peak area chart |
| performance | total and annualised return, volatility, Sharpe, Sortino, max drawdown, Calmar, skew, excess kurtosis, best/worst day, share of positive days — portfolio beside benchmark |
| tail risk | historical VaR and CVaR, parametric (normal) VaR, Cornish-Fisher VaR at 95 % and 99 % |
| where the risk comes from | marginal and component contribution per asset (Euler decomposition: components sum exactly to portfolio volatility), share of risk vs share of weight, diversification ratio, correlation heatmap |
| drawdown and worst days | peak, trough, recovery dates and durations; five worst days; 63-day rolling volatility |
| relative to benchmark | beta, Jensen alpha, correlation, tracking error, information ratio, up/down capture |
| monthly returns | calendar table with yearly totals |
| assets | growth of 100 for each sleeve |

## The live portfolio

`config/portfolio.json` defines a multi-asset proxy book from FRED series.
Indices are price-only and not investable; the 10-year Treasury sleeve is a
total-return proxy built from the `DGS10` yield by repricing a par bond
every day (see *Design notes*). 2016-10-04 to 2026-09-15, 2,436 trading
days, daily rebalancing, 4 % risk-free rate:

| asset | weight |
|---|---|
| S&P 500 | 40 % |
| Nasdaq Composite | 15 % |
| Dow Jones Industrial | 10 % |
| 10y Treasury (par-bond proxy from DGS10) | 25 % |
| Brent crude | 5 % |
| EUR/USD | 5 % |

`python -m prr report --markdown` prints this (abridged):

| metric | portfolio | S&P 500 |
|---|---|---|
| Annualised return | 10.9% | 13.9% |
| Annualised volatility | 12.7% | 18.2% |
| Sharpe ratio | 0.57 | 0.59 |
| Sortino ratio | 0.75 | 0.76 |
| Max drawdown | -24.5% | -33.9% |
| Skewness | -0.54 | -0.40 |
| Excess kurtosis | 14.72 | 15.91 |
| Worst day | -8.06% | -11.98% |

| level | historical VaR | historical CVaR | parametric VaR | Cornish-Fisher VaR |
|---|---|---|---|---|
| 95% | 1.16% | 1.94% | 1.28% | 1.16% |
| 99% | 2.27% | 3.43% | 1.82% | 4.82% |

| asset | weight | stand-alone vol | component | share of risk |
|---|---|---|---|---|
| S&P 500 | 40.0% | 18.2% | 7.07% | 55.5% |
| Nasdaq Composite | 15.0% | 22.3% | 3.14% | 24.6% |
| Dow Jones Industrial | 10.0% | 17.7% | 1.61% | 12.6% |
| 10y Treasury (par-bond proxy) | 25.0% | 7.3% | 0.01% | 0.1% |
| Brent crude | 5.0% | 50.5% | 0.85% | 6.7% |
| EUR/USD | 5.0% | 7.2% | 0.06% | 0.5% |

Max drawdown −24.5 % from 2020-02-19 to 2020-03-23 (33 calendar days),
recovered 2020-06-05 after 74 days. Beta to the S&P 500 0.68, up capture
69 %, down capture 67 %.

Three things in those numbers are worth a second look, and the report is
built so that they are visible rather than averaged away:

- **The 99 % VaRs disagree by a factor of 2.6.** Parametric says 1.8 %,
  Cornish-Fisher says 4.8 %, history says 2.3 %. With excess kurtosis of
  14.7 (March 2020 is in the sample), the normal assumption is the wrong
  one, and the report says so under the table.
- **A quarter of the book contributes 0.1 % of the risk.** The Treasury
  sleeve's marginal contribution is zero because its correlation with
  equities over this window is roughly zero — negative before 2022,
  positive after. Weight is not risk.
- **5 % in Brent contributes 6.7 % of the risk**, more than its weight,
  because its stand-alone volatility is 50 %.

## How it is put together

```
 config/portfolio.json          assets, weights, benchmark, risk-free rate, rebalancing
        │  src/prr/data.py     FRED (cached CSV) | wide CSV | synthetic GBM; transforms; inner-join alignment
        ▼
 prices: DataFrame of daily levels, one column per asset
        │  src/prr/returns.py  simple/log returns, daily-rebalanced or buy-and-hold portfolio, growth, drawdowns
        ▼
        │  src/prr/metrics.py  annualised return/vol, Sharpe, Sortino, Calmar, max drawdown with dates, monthly table
        │  src/prr/risk.py     historical/parametric/Cornish-Fisher VaR, CVaR, Euler risk contributions,
        │                       diversification ratio, beta, alpha, tracking error, capture ratios
        ▼
 src/prr/report.py             compute() once -> ReportData; render_html() and render_markdown()
 src/prr/svg.py                line, area, bar and heatmap charts as SVG strings
```

Every function in `returns`, `metrics` and `risk` is pure: pandas in,
numbers out, annualisation and the risk-free rate always explicit. That is
what makes them testable against a known answer.

## Design notes

1. **A 10-year bond from a yield series.** FRED has no daily Treasury
   total-return index, so `data.bond_total_return_index` makes one: each
   day the holder owns a par bond issued at yesterday's yield, earns one
   day of coupon, and marks it at today's yield with exact semi-annual
   discounting. Tests check that flat yields earn exactly the carry and a
   50 bp rise on a ten-year loses about 4 %. The one-day roll down the
   curve is ignored; for a ten-year bond it is noise.
2. **Inner join, not forward fill.** Equities, FX and commodities keep
   different holidays. Forward-filling a closed market gives it a zero
   return that day, which quietly deflates its volatility and its
   correlation with everything else. The panel keeps only days on which
   every asset printed.
3. **Three VaRs, deliberately.** Any one of them looks authoritative on
   its own; side by side they show whether the tails are normal. On this
   data they are not.
4. **Risk contributions are Euler contributions.** `w_i · ∂σ/∂w_i` sums
   exactly to σ, so "share of risk" is a real decomposition, not a
   heuristic. The test asserts the sum, and that two uncorrelated
   equal-volatility assets split it 50/50 with a diversification ratio of
   √2.
5. **Capture ratios use average returns.** The first version compounded
   the up-days and printed 5 % up capture for a portfolio with beta 0.68 —
   the ratio of two numbers like 10^40 is not a statistic. Averages give
   69 % / 67 %.
6. **Floating-point zero is not zero.** A hundred identical returns have a
   standard deviation of about 1e-19, and a Sharpe ratio of mean/1e-19 is
   nonsense. Ratios treat dispersion below 1e-12 as zero and return NaN.
7. **Sources that were rejected.** Stooq now fronts its CSVs with a
   JavaScript proof-of-work challenge; it was dropped rather than worked
   around. FRED's LBMA gold series is discontinued (404), hence the
   Treasury sleeve. WTI printed −$37.63 on 20 April 2020, which makes
   percentage returns meaningless, hence Brent.

## Testing

35 tests, all offline, on Python 3.11 and 3.12 in CI:

- **Designed series with known answers** — a 100→120→90→108→130 path has a
  25 % drawdown from day 2 to day 3 that recovers on day 5; a portfolio
  that is exactly twice its benchmark has beta 2.0, alpha 0, and capture
  ratios of 2.0; a sample of ten losses from −10 % to −1 % plus ninety
  small gains has a 95 % historical VaR of 5.05 % by NumPy's interpolation.
- **Identities** — component risks sum to portfolio volatility; buy-and-hold
  and daily-rebalanced returns agree on day one and diverge on day two as
  designed; monthly returns compound within the month.
- **Data layer with a fake session** — FRED parsing of "." as missing,
  on-disk caching, 404 handling, CSV validation, config validation.
- **End to end** — the synthetic portfolio through `compute`, `render_html`
  and the CLI, checking the engineered crash shows up in the drawdown and
  that the HTML references nothing external.

## Not done, on purpose

- **No optimisation.** Weights are inputs. Mean-variance and risk-parity
  solvers are a natural next module and would sit on `risk.py` as it is.
- **No transaction costs or dividends.** Daily rebalancing is frictionless
  and indices are price-only; both flatter the numbers slightly and both
  are stated in the report footer.
- **No factor model.** Beta is to a single benchmark.

## Built with Claude Code

Written in one session with Claude Code doing the typing and first drafts;
the human chose the scope, the output format and the sources, and read
every formula. The three source problems in note 7 and the capture-ratio
bug in note 5 were found by running the live portfolio, not by reading.
[`CLAUDE.md`](CLAUDE.md) has the conventions and the process.

## License

MIT. Index data is published by the Federal Reserve Bank of St. Louis
(FRED) under its own terms; this repository stores none of it.

---

David Rodriguez · [github.com/davrod-dev](https://github.com/davrod-dev)
