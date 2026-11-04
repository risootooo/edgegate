"""Bootstrap drawdown distribution.

A single realised equity curve shows exactly one drawdown -- the one that
happened to occur.  Circular **block bootstrap** (Politis & Romano, 1992)
resamples the trade series in blocks (preserving short-range dependence
between neighbouring trades), rebuilds thousands of alternative equity
curves the same strategy could plausibly have produced, and records the
maximum drawdown of each.  The 95th percentile is the honest answer to
"how bad could this reasonably get?" -- the number to size positions
against, not the rosy realised one.

Reference: Politis, D. N. & Romano, J. P. (1992), "A Circular Moving
Blocks Bootstrap for Stationary Data".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import max_drawdown


@dataclass
class BootstrapResult:
    realised_max_drawdown: float
    median_max_drawdown: float
    p95_max_drawdown: float
    mean_max_drawdown: float
    drawdowns: np.ndarray     # full distribution (for the report chart)
    n_resamples: int
    block_length: int


def _resample_blocks(rng: np.random.Generator, r: np.ndarray, block: int) -> np.ndarray:
    n = r.size
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=n_blocks)
    idx = (starts[:, None] + np.arange(block)[None, :]) % n   # circular
    return r[idx.ravel()[:n]]


def bootstrap_drawdowns(
    returns: np.ndarray,
    n_resamples: int = 2000,
    block_length: int = 5,
    seed: int = 999,
) -> BootstrapResult:
    r = np.asarray(returns, dtype=float)
    rng = np.random.default_rng(seed)
    dds = np.empty(n_resamples)
    for i in range(n_resamples):
        dds[i] = max_drawdown(_resample_blocks(rng, r, block_length))
    return BootstrapResult(
        realised_max_drawdown=max_drawdown(r),
        median_max_drawdown=float(np.percentile(dds, 50)),
        p95_max_drawdown=float(np.percentile(dds, 95)),
        mean_max_drawdown=float(dds.mean()),
        drawdowns=dds,
        n_resamples=n_resamples,
        block_length=block_length,
    )
