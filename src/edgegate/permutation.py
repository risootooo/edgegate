"""Monte Carlo Permutation Test (MCPT).

In the spirit of Timothy Masters' permutation testing for trading systems:
if a strategy has no edge, the *signs* of its trade returns are arbitrary --
knowing a trade made 0.8% tells you nothing about whether it should have
been +0.8% or -0.8%.  So we build the null distribution by randomly
flipping the sign of every trade return (magnitudes -- i.e. how the system
sizes and where it trades -- are kept; only the claimed direction of the
edge is destroyed) and ask: how often does pure sign-luck produce a result
at least as good as the one observed?

The test statistic is the per-trade Sharpe ratio.  Note that sign flips
leave the standard deviation unchanged, so the Sharpe ordering is exactly
the ordering of total return; we also record the permutation p-value for
profit factor for the report.

p-value uses the standard +1 correction:  p = (1 + #{null >= obs}) / (1 + N).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import profit_factor, sharpe_ratio


@dataclass
class PermutationResult:
    observed_sharpe: float
    observed_profit_factor: float
    p_value: float            # primary gate statistic (Sharpe / total return)
    p_value_profit_factor: float
    null_sharpes: np.ndarray  # full null distribution (for the report chart)
    n_permutations: int


def sign_flip_permutation_test(
    returns: np.ndarray,
    n_permutations: int = 5000,
    seed: int = 12345,
) -> PermutationResult:
    r = np.asarray(returns, dtype=float)
    obs_sharpe = sharpe_ratio(r)
    obs_pf = profit_factor(r)

    rng = np.random.default_rng(seed)
    null_sharpes = np.empty(n_permutations)
    ge_sharpe = 0
    ge_pf = 0
    # Chunked so a 600-trade series x 5000 permutations stays memory-light.
    chunk = 500
    done = 0
    while done < n_permutations:
        m = min(chunk, n_permutations - done)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(m, r.size))
        flipped = signs * r
        means = flipped.mean(axis=1)
        # std is invariant under sign flips; use the observed std.
        null_sharpes[done:done + m] = means / r.std(ddof=1) if r.std(ddof=1) else 0.0
        ge_sharpe += int((null_sharpes[done:done + m] >= obs_sharpe).sum())
        for row in flipped:
            if profit_factor(row) >= obs_pf:
                ge_pf += 1
        done += m

    p_sharpe = (1 + ge_sharpe) / (1 + n_permutations)
    p_pf = (1 + ge_pf) / (1 + n_permutations)
    return PermutationResult(
        observed_sharpe=obs_sharpe,
        observed_profit_factor=obs_pf,
        p_value=float(p_sharpe),
        p_value_profit_factor=float(p_pf),
        null_sharpes=null_sharpes,
        n_permutations=n_permutations,
    )
