"""Walk-forward analysis.

The series is cut into ``n_folds`` consecutive, non-overlapping folds
(the order of trades is never shuffled -- time structure is the point).
For each fold we compute profit factor, per-trade Sharpe and total return.
A real edge should show up in *most* folds; an edge that lives in a
single lucky window is a regime artefact, not a strategy.

Consistency score = fraction of folds with a positive total return.
The gate passes when that fraction is at least 0.6 (a strict majority,
with slack for one bad regime in five folds).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import profit_factor, sharpe_ratio


@dataclass
class FoldResult:
    fold: int
    n_trades: int
    total_return: float
    profit_factor: float
    sharpe: float


@dataclass
class WalkForwardResult:
    folds: list[FoldResult]
    consistency: float        # fraction of folds with positive total return
    positive_folds: int
    n_folds: int

    @property
    def passes(self) -> bool:
        return self.consistency >= 0.6


def walk_forward(returns: np.ndarray, n_folds: int = 5) -> WalkForwardResult:
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        return WalkForwardResult([], 0.0, 0, 0)
    n_folds = int(min(n_folds, r.size))
    chunks = np.array_split(r, n_folds)
    folds: list[FoldResult] = []
    for i, chunk in enumerate(chunks):
        folds.append(
            FoldResult(
                fold=i + 1,
                n_trades=int(chunk.size),
                total_return=float(chunk.sum()),
                profit_factor=profit_factor(chunk),
                sharpe=sharpe_ratio(chunk),
            )
        )
    positive = sum(1 for f in folds if f.total_return > 0)
    consistency = positive / len(folds) if folds else 0.0
    return WalkForwardResult(folds, float(consistency), positive, len(folds))
