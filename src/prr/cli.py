"""``python -m prr <command>``."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from prr import config, data, report


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="prr", description="portfolio performance and risk report")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("report", help="build the report for a portfolio definition")
    r.add_argument("--config", default=str(config.DEFAULT_CONFIG))
    r.add_argument("--out", default=str(config.REPORTS_DIR / "report.html"))
    r.add_argument("--markdown", action="store_true", help="also write a .md summary next to the HTML")
    r.add_argument("--force", action="store_true", help="re-download cached series")

    s = sub.add_parser("sample", help="render the synthetic sample portfolio; no network")
    s.add_argument("--out", default="examples", help="directory for sample_report.html and sample_summary.md")

    f = sub.add_parser("fetch", help="download and cache the series a portfolio needs")
    f.add_argument("--config", default=str(config.DEFAULT_CONFIG))
    f.add_argument("--force", action="store_true")
    return p


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"[write] {path} ({len(text.encode('utf-8')) // 1024} KB)")


def build(portfolio: config.Portfolio, *, force: bool = False) -> report.ReportData:
    client = data.FredClient(max_age_days=0 if force else 1.0)
    prices = data.build_price_panel(portfolio, client=client)
    print(f"[data]  {len(prices):,} common trading days, {prices.index[0].date()} to {prices.index[-1].date()}")
    return report.compute(portfolio, prices)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "sample":
            rd = report.compute(data.synthetic_portfolio(), data.build_price_panel(data.synthetic_portfolio()))
            out = Path(args.out)
            _write(out / "sample_report.html", report.render_html(rd))
            _write(out / "sample_summary.md", report.render_markdown(rd))
        elif args.command == "fetch":
            portfolio = config.load_portfolio(Path(args.config))
            client = data.FredClient(max_age_days=0 if args.force else 1.0)
            for asset in portfolio.assets:
                client.fetch(asset.id, force=args.force)
        elif args.command == "report":
            portfolio = config.load_portfolio(Path(args.config))
            rd = build(portfolio, force=args.force)
            out = Path(args.out)
            _write(out, report.render_html(rd))
            if args.markdown:
                _write(out.with_suffix(".md"), report.render_markdown(rd))
            print(report.render_markdown(rd))
    except (config.ConfigError, data.DataError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0
