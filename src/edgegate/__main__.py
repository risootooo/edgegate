"""EdgeGate CLI.

Usage:
    python -m edgegate                      # run both bundled demos
    python -m edgegate --csv my_trades.csv --trials 50 --cost 0.08
    python -m edgegate --output docs/report.html --permutations 5000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import demo_strategies, load_returns_csv, profit_factor
from .report import build_report
from .verdict import StrategyReport, evaluate_strategy


def _print_summary(rep: StrategyReport) -> None:
    print(f"\n{rep.name}")
    print(f"  trades={rep.n_trades}  PF(net)={rep.profit_factor:.2f}  "
          f"Sharpe/trade={rep.sharpe:.3f}  total(net)={rep.total_return:+.2f}%")
    for g in rep.gates:
        print(f"  [{'PASS' if g.passed else 'FAIL'}] {g.name}: {g.detail}")
    print(f"  bootstrap max-DD: realised {rep.bootstrap.realised_max_drawdown:.1f}%  "
          f"median {rep.bootstrap.median_max_drawdown:.1f}%  "
          f"95th pct {rep.bootstrap.p95_max_drawdown:.1f}%")
    print(f"  VERDICT: {rep.verdict}")


def _report_dict(rep: StrategyReport) -> dict:
    return {
        "name": rep.name,
        "verdict": rep.verdict,
        "n_trades": rep.n_trades,
        "n_trials": rep.n_trials,
        "base_cost_pct": rep.base_cost_pct,
        "profit_factor_net": rep.profit_factor,
        "sharpe_net": rep.sharpe,
        "total_return_net_pct": rep.total_return,
        "mcpt_p_value": rep.permutation.p_value,
        "mcpt_p_value_pf": rep.permutation.p_value_profit_factor,
        "dsr_probability": rep.dsr_probability,
        "walkforward_consistency": rep.walkforward.consistency,
        "walkforward_positive_folds": rep.walkforward.positive_folds,
        "walkforward_folds": [
            {"fold": f.fold, "total_return": f.total_return,
             "profit_factor": f.profit_factor, "sharpe": f.sharpe}
            for f in rep.walkforward.folds
        ],
        "cost_stress": [
            {"multiplier": r.multiplier, "total_return": r.total_return,
             "profit_factor": r.profit_factor, "sharpe": r.sharpe}
            for r in rep.costs.rows
        ],
        "profit_factor_gross": profit_factor(rep.gross_returns),
        "bootstrap_dd_median_pct": rep.bootstrap.median_max_drawdown,
        "bootstrap_dd_p95_pct": rep.bootstrap.p95_max_drawdown,
        "realised_max_dd_pct": rep.realised_max_drawdown,
        "gates": [{"name": g.name, "passed": g.passed, "detail": g.detail}
                  for g in rep.gates],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="edgegate", description=__doc__)
    ap.add_argument("--csv", help="CSV of your own trade returns (percent per trade)")
    ap.add_argument("--name", default="Custom strategy", help="name for --csv strategy")
    ap.add_argument("--trials", type=int, default=1,
                    help="how many variants you tried before this one (DSR)")
    ap.add_argument("--cost", type=float, default=0.05,
                    help="1x transaction cost, percent per trade (for --csv)")
    ap.add_argument("--output", default="docs/report.html",
                    help="where to write the HTML report")
    ap.add_argument("--json", default=None, help="also write results JSON here")
    ap.add_argument("--permutations", type=int, default=5000)
    ap.add_argument("--resamples", type=int, default=2000)
    args = ap.parse_args(argv)

    reports: list[StrategyReport] = []
    if args.csv:
        returns = load_returns_csv(args.csv)
        reports.append(evaluate_strategy(
            args.name, returns, n_trials=args.trials, base_cost_pct=args.cost,
            n_permutations=args.permutations, n_resamples=args.resamples))
    else:
        for demo in demo_strategies():
            reports.append(evaluate_strategy(
                demo.name, demo.returns, n_trials=demo.n_trials,
                base_cost_pct=demo.base_cost_pct,
                n_permutations=args.permutations, n_resamples=args.resamples))

    for rep in reports:
        _print_summary(rep)

    out = build_report(reports, Path(args.output))
    print(f"\nReport written to {out}")
    if args.json:
        Path(args.json).write_text(
            json.dumps([_report_dict(r) for r in reports], indent=2), encoding="utf-8")
        print(f"Results written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
