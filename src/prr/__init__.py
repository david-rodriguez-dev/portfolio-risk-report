"""portfolio-risk-report: performance and risk analytics from daily prices.

    python -m prr report  --config config/portfolio.json --out reports/report.html
    python -m prr sample  --out examples/            synthetic data, no network
    python -m prr fetch   --config config/portfolio.json

Numerics live in ``returns``, ``metrics`` and ``risk``; ``report`` turns
them into a single self-contained HTML file with inline SVG charts.
"""

__version__ = "0.1.0"
