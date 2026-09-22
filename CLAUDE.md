# CLAUDE.md

Project rules for Claude Code, and how this repo was built with it.

## Conventions the tool must keep

- **Numerics are pure functions on pandas objects** in `returns.py`,
  `metrics.py` and `risk.py`. They take a Series or DataFrame and explicit
  parameters (periods per year, risk-free rate, confidence level) and
  return numbers or frames. No I/O, no globals, no printing.
- **Every metric has a hand-computed test.** A designed series with a known
  answer (a 100-120-90-108-130 path has a 25 % drawdown that recovers in
  two days), not a regression snapshot. A metric that only "runs" is not
  tested.
- **Annualisation is explicit.** Nothing assumes 252 without being told;
  the risk-free rate is compounded to a per-period rate, never divided.
- **The HTML report stands alone.** Inline CSS, inline SVG, no scripts, no
  CDN. `svg.py` draws the charts; do not add a plotting dependency for a
  chart that a few dozen lines of SVG can draw.
- **Data sources are behind `data.build_price_panel`.** FRED, CSV and the
  synthetic generator all produce the same wide DataFrame of daily levels
  on the dates where every asset printed. Inner join, no forward fill.
- **`sample` must work offline** and CI runs it. The synthetic generator is
  deterministic for a seed and contains an engineered crash so drawdown
  and tail statistics have something to find.
- **No keys, no contact details, no data in git.** FRED's CSV endpoint
  needs no key; `data/` and `reports/` are ignored. `examples/` holds the
  synthetic sample only.

## How this was built

One Claude Code session. The human chose the scope (textbook performance
and risk analytics, nothing proprietary), the output (a single HTML file a
recruiter can open), and the rule that this repository shares nothing with
any other project. The model wrote the numerics first with tests, then the
data layer, then the charts and report, and finally ran the live portfolio.

Things the data decided:

- **Stooq**, the intended price source, now fronts its CSV endpoint with a
  JavaScript proof-of-work challenge. It was dropped rather than worked
  around. FRED's `fredgraph.csv` endpoint serves daily index levels with
  no key and no challenge.
- **FRED's LBMA gold series is discontinued** (404). The gold sleeve became
  a 10-year Treasury total-return proxy built from the `DGS10` yield by
  repricing a par bond daily: a more useful sleeve and a more interesting
  transform.
- **WTI printed −$37.63 on 20 April 2020.** A negative price makes
  percentage returns meaningless, so the oil sleeve is Brent.
- **Capture ratios must use average returns.** Compounding 1,300 up-days
  and dividing two enormous numbers produced "5 % up capture" for a
  portfolio with a beta of 0.68. The fix and the reason are in
  `risk.capture_ratios`.
