"""Self-contained HTML report generation.

One file, no CDN, no JavaScript frameworks -- all charts are inline SVG
generated from the actual computed distributions, so the report works
offline, in an email attachment, or in a GitHub preview, forever.
"""

from __future__ import annotations

import html
from pathlib import Path

import numpy as np

from .data import equity_curve
from .verdict import StrategyReport

VERDICT_COLORS = {"PASS": "#22c55e", "FAIL": "#ef4444", "UNPROVEN": "#f59e0b"}


def _pf(value: float) -> str:
    """Profit factor display: a fold with no losing trades shows as infinity."""
    return "inf" if np.isinf(value) else f"{value:.2f}"
ACCENTS = ["#38bdf8", "#a78bfa", "#f472b6", "#34d399"]


def _histogram_svg(
    data: np.ndarray,
    marker: float | None,
    title: str,
    color: str = "#38bdf8",
    width: int = 680,
    height: int = 240,
    xlabel: str = "",
) -> str:
    data = np.asarray(data, dtype=float)
    lo, hi = float(data.min()), float(data.max())
    if marker is not None:
        lo, hi = min(lo, marker), max(hi, marker)
    pad = (hi - lo) * 0.05 or 1.0
    lo, hi = lo - pad, hi + pad
    counts, edges = np.histogram(data, bins=30, range=(lo, hi))
    top = counts.max() or 1
    plot_w, plot_h = width - 70, height - 70
    bw = plot_w / len(counts)
    bars = []
    for i, c in enumerate(counts):
        x = 50 + i * bw
        h = (c / top) * plot_h
        bars.append(
            f'<rect x="{x:.1f}" y="{30 + plot_h - h:.1f}" width="{bw - 1:.1f}" '
            f'height="{h:.1f}" fill="{color}" opacity="0.75"/>'
        )
    marker_svg = ""
    if marker is not None:
        mx = 50 + (marker - lo) / (hi - lo) * plot_w
        marker_svg = (
            f'<line x1="{mx:.1f}" y1="22" x2="{mx:.1f}" y2="{30 + plot_h}" '
            f'stroke="#fbbf24" stroke-width="2.5"/>'
            f'<text x="{mx:.1f}" y="16" fill="#fbbf24" font-size="12" '
            f'text-anchor="middle" font-weight="700">observed {marker:.2f}</text>'
        )
    return (
        f'<svg viewBox="0 0 {width} {height}" class="chart" role="img">'
        f'<text x="50" y="14" fill="#94a3b8" font-size="12">{html.escape(title)}</text>'
        + "".join(bars)
        + marker_svg
        + f'<text x="{50 + plot_w / 2:.0f}" y="{height - 8}" fill="#64748b" '
          f'font-size="11" text-anchor="middle">{html.escape(xlabel)}</text>'
        + f'<text x="50" y="{height - 24}" fill="#64748b" font-size="10">{lo:.2f}</text>'
        + f'<text x="{50 + plot_w}" y="{height - 24}" fill="#64748b" font-size="10" '
          f'text-anchor="end">{hi:.2f}</text>'
        + "</svg>"
    )


def _equity_svg(reports: list[StrategyReport], width: int = 680, height: int = 260) -> str:
    curves = []
    all_vals = []
    for rep in reports:
        eq = equity_curve(rep.net_returns)
        if eq.size > 400:  # downsample for a light file
            idx = np.linspace(0, eq.size - 1, 400).astype(int)
            eq = eq[idx]
        curves.append(eq)
        all_vals.append(eq)
    joined = np.concatenate(all_vals)
    lo, hi = float(joined.min()), float(joined.max())
    pad = (hi - lo) * 0.08 or 1.0
    lo, hi = lo - pad, hi + pad
    plot_w, plot_h = width - 70, height - 60
    parts = [
        f'<svg viewBox="0 0 {width} {height}" class="chart" role="img">',
        '<text x="50" y="14" fill="#94a3b8" font-size="12">'
        "Equity curves (cumulative net return, %)</text>",
    ]
    zero_y = 26 + (hi - 0) / (hi - lo) * plot_h if lo < 0 < hi else None
    if zero_y is not None:
        parts.append(
            f'<line x1="50" y1="{zero_y:.1f}" x2="{50 + plot_w}" y2="{zero_y:.1f}" '
            'stroke="#334155" stroke-dasharray="4 4"/>'
        )
    legend_x = 50
    for i, (rep, eq) in enumerate(zip(reports, curves)):
        color = ACCENTS[i % len(ACCENTS)]
        pts = []
        for j, v in enumerate(eq):
            x = 50 + j / max(1, len(eq) - 1) * plot_w
            y = 26 + (hi - v) / (hi - lo) * plot_h
            pts.append(f"{x:.1f},{y:.1f}")
        parts.append(
            f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" '
            f'stroke-width="2"/>'
        )
        parts.append(
            f'<rect x="{legend_x}" y="{height - 22}" width="10" height="10" fill="{color}"/>'
            f'<text x="{legend_x + 15}" y="{height - 12}" fill="#cbd5e1" '
            f'font-size="12">{html.escape(rep.name)}</text>'
        )
        legend_x += 15 + 9 * len(rep.name) + 40
    parts.append("</svg>")
    return "".join(parts)


def _strategy_section(rep: StrategyReport, accent: str) -> str:
    vc = VERDICT_COLORS[rep.verdict]
    gate_rows = "".join(
        f'<tr><td>{html.escape(g.name)}</td>'
        f'<td><span class="pill" style="background:{"#14532d" if g.passed else "#450a0a"};'
        f'color:{"#4ade80" if g.passed else "#f87171"}">{"PASS" if g.passed else "FAIL"}</span></td>'
        f"<td>{html.escape(g.detail)}</td></tr>"
        for g in rep.gates
    )
    fold_rows = "".join(
        f"<tr><td>Fold {f.fold}</td><td>{f.n_trades}</td>"
        f"<td>{f.total_return:+.2f}%</td><td>{_pf(f.profit_factor)}</td>"
        f"<td>{f.sharpe:+.3f}</td></tr>"
        for f in rep.walkforward.folds
    )
    cost_rows = "".join(
        f"<tr><td>{r.multiplier:.0f}x</td><td>{r.total_return:+.2f}%</td>"
        f"<td>{_pf(r.profit_factor)}</td><td>{r.sharpe:+.3f}</td></tr>"
        for r in rep.costs.rows
    )
    perm_chart = _histogram_svg(
        rep.permutation.null_sharpes,
        rep.permutation.observed_sharpe,
        "MCPT null distribution (sign-flipped Sharpe ratios)",
        color=accent,
        xlabel="per-trade Sharpe under the null (no edge)",
    )
    dd_chart = _histogram_svg(
        rep.bootstrap.drawdowns,
        rep.bootstrap.realised_max_drawdown,
        "Bootstrap distribution of maximum drawdown (%)",
        color="#f472b6",
        xlabel="maximum drawdown (%) over resampled equity curves",
    )
    return f"""
<section class="card">
  <div class="card-head">
    <h2>{html.escape(rep.name)}</h2>
    <span class="verdict" style="background:{vc}">{rep.verdict}</span>
  </div>
  <div class="metrics">
    <div><span>Trades</span><strong>{rep.n_trades}</strong></div>
    <div><span>Profit factor (net)</span><strong>{rep.profit_factor:.2f}</strong></div>
    <div><span>Sharpe / trade</span><strong>{rep.sharpe:.3f}</strong></div>
    <div><span>Total return (net)</span><strong>{rep.total_return:+.1f}%</strong></div>
    <div><span>MCPT p-value</span><strong>{rep.permutation.p_value:.4f}</strong></div>
    <div><span>Deflated Sharpe prob.</span><strong>{rep.dsr_probability:.3f}</strong></div>
    <div><span>Realised max DD</span><strong>{rep.realised_max_drawdown:.1f}%</strong></div>
    <div><span>Bootstrap 95th pct DD</span><strong>{rep.bootstrap.p95_max_drawdown:.1f}%</strong></div>
  </div>
  <h3>The four gates</h3>
  <table><tr><th>Gate</th><th>Result</th><th>Detail</th></tr>{gate_rows}</table>
  <div class="charts">{perm_chart}{dd_chart}</div>
  <div class="two-col">
    <div><h3>Walk-forward folds</h3>
      <table><tr><th>Fold</th><th>n</th><th>Total</th><th>PF</th><th>Sharpe</th></tr>{fold_rows}</table></div>
    <div><h3>Cost stress</h3>
      <table><tr><th>Cost</th><th>Total</th><th>PF</th><th>Sharpe</th></tr>{cost_rows}</table>
      <p class="note">Base cost {rep.base_cost_pct:.2f}% per trade; gates use the 1x-cost series, survival is judged at 2x.</p></div>
  </div>
</section>"""


CSS = """
:root{color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;background:#0b1120;color:#e2e8f0;
font-family:'Segoe UI',system-ui,-apple-system,Roboto,'Helvetica Neue',Arial,sans-serif}
.wrap{max-width:1060px;margin:0 auto;padding:40px 22px 64px}
header{text-align:center;padding:28px 0 8px}
.logo{font-size:44px;font-weight:800;letter-spacing:-1px}
.logo span{color:#38bdf8}
.tagline{color:#94a3b8;font-size:18px;margin-top:6px}
.tagline strong{color:#fbbf24}
.card{background:#111a2e;border:1px solid #1e293b;border-radius:16px;
padding:28px;margin:28px 0;box-shadow:0 10px 30px rgba(0,0,0,.35)}
.card-head{display:flex;justify-content:space-between;align-items:center;gap:16px;flex-wrap:wrap}
h2{margin:0;font-size:26px}h3{color:#cbd5e1;margin:22px 0 10px;font-size:16px}
.verdict{color:#0b1120;font-weight:800;font-size:18px;padding:8px 22px;border-radius:999px;letter-spacing:1px}
.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-top:18px}
.metrics div{background:#0b1120;border:1px solid #1e293b;border-radius:12px;padding:12px}
.metrics span{display:block;color:#64748b;font-size:12px;text-transform:uppercase;letter-spacing:.5px}
.metrics strong{font-size:21px}
table{width:100%;border-collapse:collapse;font-size:14px}
th{color:#64748b;text-align:left;font-weight:600;padding:6px 8px;border-bottom:1px solid #1e293b}
td{padding:6px 8px;border-bottom:1px solid #16213a}
.pill{padding:2px 10px;border-radius:999px;font-weight:700;font-size:12px}
.charts{display:grid;grid-template-columns:1fr;gap:8px;margin-top:10px}
.chart{width:100%;height:auto;background:#0b1120;border:1px solid #1e293b;border-radius:12px}
.two-col{display:grid;grid-template-columns:1fr 1fr;gap:24px}
@media(max-width:760px){.two-col{grid-template-columns:1fr}}
.note{color:#64748b;font-size:13px}
footer{color:#475569;text-align:center;font-size:13px;margin-top:36px;line-height:1.7}
"""


def build_report(reports: list[StrategyReport], out_path: str | Path) -> Path:
    out = Path(out_path)
    sections = "".join(
        _strategy_section(rep, ACCENTS[i % len(ACCENTS)])
        for i, rep in enumerate(reports)
    )
    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>EdgeGate -- Strategy Validation Report</title>
<style>{CSS}</style></head>
<body><div class="wrap">
<header>
  <div class="logo">Edge<span>Gate</span></div>
  <p class="tagline">Your backtest is lying to you. <strong>EdgeGate proves it.</strong></p>
  <p class="note">Monte Carlo permutation test &middot; Deflated Sharpe Ratio &middot;
  walk-forward consistency &middot; bootstrap drawdowns &middot; cost stress.
  A strategy must pass <em>every</em> gate before it deserves real money.</p>
</header>
{_equity_svg(reports)}
{sections}
<footer>Generated by EdgeGate &middot; Demo strategies are synthetic by design
(see README for the disclosed generating processes) &middot;
Research software -- not financial advice.</footer>
</div></body></html>"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")
    return out
