"""EdgeGate -- a strategy-validation lab.

Runs any strategy's trade list through four statistical gates (Monte
Carlo permutation test, Deflated Sharpe Ratio, walk-forward consistency,
cost stress) plus a bootstrap drawdown distribution, and issues a single
honest verdict: PASS, FAIL, or UNPROVEN.
"""

from .bootstrap import bootstrap_drawdowns
from .costs import apply_costs, cost_stress
from .data import (
    demo_strategies,
    generate_mirage,
    generate_modest_edge,
    load_returns_csv,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
)
from .deflated_sharpe import deflated_sharpe_for_returns, deflated_sharpe_probability
from .permutation import sign_flip_permutation_test
from .verdict import StrategyReport, evaluate_strategy
from .walkforward import walk_forward

__version__ = "1.0.0"
__all__ = [
    "apply_costs",
    "bootstrap_drawdowns",
    "cost_stress",
    "deflated_sharpe_for_returns",
    "deflated_sharpe_probability",
    "demo_strategies",
    "evaluate_strategy",
    "generate_mirage",
    "generate_modest_edge",
    "load_returns_csv",
    "max_drawdown",
    "profit_factor",
    "sharpe_ratio",
    "sign_flip_permutation_test",
    "StrategyReport",
    "walk_forward",
]
