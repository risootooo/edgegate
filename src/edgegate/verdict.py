"""The gate verdict engine.

Four gates, all evaluated on the **net (1x-cost)** series unless stated:

1. **MCPT**          -- sign-flip permutation p-value < 0.05
2. **Deflated Sharpe** -- DSR probability > 0.95 (selection-corrected)
3. **Walk-forward**  -- at least 60% of folds profitable
4. **Cost stress**   -- still profitable with PF > 1 at 2x costs

Verdict logic
-------------
* ``UNPROVEN`` if the sample has fewer than 30 trades -- no statistic can
  rescue a sample that small, and pretending otherwise is how accounts die.
* ``PASS`` if **all four** gates pass.
* ``FAIL`` if the strategy is net-negative, passes at most one gate, or
  fails both the MCPT and the cost-stress gate (no detectable edge *and*
  the edge that appears on paper cannot survive realistic execution).
* ``UNPROVEN`` otherwise -- mixed evidence; collect more trades, do not
  trade it yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .bootstrap import BootstrapResult, bootstrap_drawdowns
from .costs import CostStressResult, apply_costs, cost_stress
from .data import max_drawdown, profit_factor, sharpe_ratio, total_return
from .deflated_sharpe import deflated_sharpe_for_returns
from .permutation import PermutationResult, sign_flip_permutation_test
from .walkforward import WalkForwardResult, walk_forward

MIN_TRADES = 30
MCPT_ALPHA = 0.05
DSR_THRESHOLD = 0.95


@dataclass
class GateOutcome:
    name: str
    passed: bool
    detail: str


@dataclass
class StrategyReport:
    name: str
    n_trades: int
    n_trials: int
    base_cost_pct: float
    # headline (net of 1x costs)
    profit_factor: float
    sharpe: float
    total_return: float
    realised_max_drawdown: float
    # gates
    permutation: PermutationResult
    dsr_probability: float
    walkforward: WalkForwardResult
    costs: CostStressResult
    bootstrap: BootstrapResult
    gates: list[GateOutcome] = field(default_factory=list)
    verdict: str = "UNPROVEN"
    net_returns: np.ndarray = field(default_factory=lambda: np.array([]))
    gross_returns: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def gates_passed(self) -> int:
        return sum(1 for g in self.gates if g.passed)


def evaluate_strategy(
    name: str,
    gross_returns: np.ndarray,
    n_trials: int = 1,
    base_cost_pct: float = 0.05,
    n_permutations: int = 5000,
    n_resamples: int = 2000,
    seed: int = 2024,
) -> StrategyReport:
    gross = np.asarray(gross_returns, dtype=float)
    net = apply_costs(gross, base_cost_pct, 1.0)

    perm = sign_flip_permutation_test(net, n_permutations=n_permutations, seed=seed)
    dsr = deflated_sharpe_for_returns(net, n_trials=n_trials)
    wf = walk_forward(net, n_folds=5)
    cs = cost_stress(gross, base_cost_pct)
    boot = bootstrap_drawdowns(net, n_resamples=n_resamples, seed=seed + 1)

    gates = [
        GateOutcome(
            "Monte Carlo Permutation Test",
            perm.p_value < MCPT_ALPHA,
            f"p = {perm.p_value:.4f} (need < {MCPT_ALPHA})",
        ),
        GateOutcome(
            "Deflated Sharpe Ratio",
            dsr > DSR_THRESHOLD,
            f"P(true SR > 0) = {dsr:.3f} after {n_trials} trials (need > {DSR_THRESHOLD})",
        ),
        GateOutcome(
            "Walk-Forward Consistency",
            wf.passes,
            f"{wf.positive_folds}/{wf.n_folds} folds profitable "
            f"(consistency {wf.consistency:.0%}, need >= 60%)",
        ),
        GateOutcome(
            "Cost Stress (2x)",
            cs.survives_2x,
            f"at 2x costs: total {cs.row(2.0).total_return:+.2f}%, "
            f"PF {cs.row(2.0).profit_factor:.2f} (need total > 0 and PF > 1)",
        ),
    ]

    report = StrategyReport(
        name=name,
        n_trades=int(net.size),
        n_trials=int(n_trials),
        base_cost_pct=float(base_cost_pct),
        profit_factor=profit_factor(net),
        sharpe=sharpe_ratio(net),
        total_return=total_return(net),
        realised_max_drawdown=max_drawdown(net),
        permutation=perm,
        dsr_probability=dsr,
        walkforward=wf,
        costs=cs,
        bootstrap=boot,
        gates=gates,
        net_returns=net,
        gross_returns=gross,
    )

    passed = report.gates_passed
    mcpt_ok, cost_ok = gates[0].passed, gates[3].passed
    if report.n_trades < MIN_TRADES:
        report.verdict = "UNPROVEN"
    elif passed == 4:
        report.verdict = "PASS"
    elif report.total_return <= 0 or passed <= 1 or (not mcpt_ok and not cost_ok):
        report.verdict = "FAIL"
    else:
        report.verdict = "UNPROVEN"
    return report
