"""Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).

An observed Sharpe ratio is inflated when (a) it was selected as the best
of many trials and (b) returns are non-normal.  The Deflated Sharpe Ratio
(DSR) returns the *probability* that the true Sharpe is greater than zero
after correcting for both:

    DSR = Phi( (SR - SR0) * sqrt(T - 1)
               / sqrt(1 - skew * SR + ((kurt - 1) / 4) * SR^2) )

where SR0 is the Sharpe expected from selection alone -- the expected
maximum of ``n_trials`` Sharpe ratios estimated under the null:

    SR0 = sqrt(V) * [ (1 - gamma) * Z^-1(1 - 1/N)
                      + gamma * Z^-1(1 - 1 / (N * e)) ]

with V = 1 / (T - 1) the null variance of a Sharpe estimate, gamma the
Euler-Mascheroni constant, and Z the standard normal CDF.

Reference: Bailey, D. H. & Lopez de Prado, M. (2014), "The Deflated
Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and
Non-Normality", Journal of Portfolio Management 40(5).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis as _kurtosis
from scipy.stats import norm, skew as _skew

EULER_GAMMA = 0.5772156649015329


def expected_max_sharpe(n_trials: int, sample_length: int) -> float:
    """Sharpe expected purely from picking the best of ``n_trials`` tries."""
    if n_trials <= 1:
        return 0.0
    if sample_length <= 1:
        return 0.0
    v = 1.0 / (sample_length - 1)
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(v) * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def deflated_sharpe_probability(
    observed_sharpe: float,
    sample_length: int,
    skew: float,
    kurtosis: float,
    n_trials: int,
) -> float:
    """DSR as a probability in [0, 1].

    ``kurtosis`` is the *non-excess* (Pearson) kurtosis -- 3.0 for a normal
    distribution -- matching the (kurt - 1) term in the original formula.
    """
    sr = float(observed_sharpe)
    t = int(sample_length)
    if t <= 2:
        return 0.0
    sr0 = expected_max_sharpe(n_trials, t)
    denom = np.sqrt(max(1e-12, 1.0 - skew * sr + ((kurtosis - 1.0) / 4.0) * sr**2))
    z = (sr - sr0) * np.sqrt(t - 1) / denom
    return float(norm.cdf(z))


def deflated_sharpe_for_returns(returns, n_trials: int) -> float:
    """Convenience wrapper: DSR probability straight from a return series."""
    from .data import sharpe_ratio

    r = np.asarray(returns, dtype=float)
    if r.size > 3 and r.std(ddof=1) > 0:
        sk = float(_skew(r))
        ku = float(_kurtosis(r, fisher=False))
    else:  # constant / tiny series: fall back to normal moments
        sk, ku = 0.0, 3.0
    return deflated_sharpe_probability(
        observed_sharpe=sharpe_ratio(r),
        sample_length=r.size,
        skew=sk,
        kurtosis=ku,
        n_trials=n_trials,
    )
