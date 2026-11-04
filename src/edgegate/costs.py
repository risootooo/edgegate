"""Cost-stress test.

Backtests routinely assume costs that real execution laughs at.  This
gate re-runs the strategy net of 1x, 2x and 3x the stated per-trade
transaction cost (spread + slippage + commission, in percent per trade)
and asks the only question that matters: **does the edge survive paying
twice what you hoped?**

Costs are charged per trade, so high-turnover strategies with tiny
average wins -- the classic curve-fit scalper profile -- are hit hardest,
exactly as they are live.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import profit_factor, sharpe_ratio, total_return


def apply_costs(returns: np.ndarray, base_cost_pct: float, multiplier: float) -> np.ndarray:
    """Net returns after ``multiplier`` x the base per-trade cost."""
    return np.asarray(returns, dtype=float) - multiplier * base_cost_pct


@dataclass
class CostRow:
    multiplier: float
    total_return: float
    profit_factor: float
    sharpe: float


@dataclass
class CostStressResult:
    rows: list[CostRow]
    base_cost_pct: float

    def row(self, multiplier: float) -> CostRow:
        for r in self.rows:
            if r.multiplier == multiplier:
                return r
        raise KeyError(multiplier)

    @property
    def survives_2x(self) -> bool:
        """Edge survives 2x costs: still profitable with PF above 1."""
        r2 = self.row(2.0)
        return r2.total_return > 0 and r2.profit_factor > 1.0


def cost_stress(
    returns: np.ndarray,
    base_cost_pct: float,
    multipliers: tuple[float, ...] = (1.0, 2.0, 3.0),
) -> CostStressResult:
    rows = []
    for m in multipliers:
        net = apply_costs(returns, base_cost_pct, m)
        rows.append(
            CostRow(
                multiplier=float(m),
                total_return=total_return(net),
                profit_factor=profit_factor(net),
                sharpe=sharpe_ratio(net),
            )
        )
    return CostStressResult(rows=rows, base_cost_pct=float(base_cost_pct))
