"""Demo data generators and basic strategy statistics.

Units
-----
Everywhere in EdgeGate a *trade return* is expressed in **percent per trade**
(``0.5`` means the trade made +0.5% of capital at risk).  This keeps the
demo numbers human-readable and makes the per-trade transaction cost, also
expressed in percent, directly subtractable.

The two bundled demo strategies are **synthetic by design**.  Their
generating processes are documented honestly below and in the README --
nothing here is market data, and the demos exist to show that the gates can
tell a real (injected) edge from a curve-fit mirage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Basic statistics on a trade-return series
# ---------------------------------------------------------------------------

def profit_factor(returns: np.ndarray) -> float:
    """Gross profits / gross losses.  ``inf`` if there are no losses."""
    r = np.asarray(returns, dtype=float)
    wins = r[r > 0].sum()
    losses = -r[r < 0].sum()
    if losses == 0:
        return float("inf") if wins > 0 else 0.0
    return float(wins / losses)


def sharpe_ratio(returns: np.ndarray) -> float:
    """Per-trade Sharpe ratio: mean / std (ddof=1).  Not annualised."""
    r = np.asarray(returns, dtype=float)
    if r.size < 2:
        return 0.0
    sd = r.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(r.mean() / sd)


def total_return(returns: np.ndarray) -> float:
    """Sum of per-trade returns (percentage points)."""
    return float(np.asarray(returns, dtype=float).sum())


def equity_curve(returns: np.ndarray) -> np.ndarray:
    """Cumulative PnL curve starting at 0, length ``len(returns) + 1``."""
    r = np.asarray(returns, dtype=float)
    return np.concatenate([[0.0], np.cumsum(r)])


def max_drawdown(returns: np.ndarray) -> float:
    """Maximum peak-to-trough decline of the cumulative PnL curve (>= 0)."""
    eq = equity_curve(returns)
    peak = np.maximum.accumulate(eq)
    return float((peak - eq).max()) if eq.size else 0.0


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

_CANDIDATE_COLUMNS = ("return_pct", "return", "returns", "pnl", "pnl_pct")


def load_returns_csv(path: str) -> np.ndarray:
    """Load a trade-return series from a CSV file.

    Looks for a column named one of ``return_pct``, ``return``, ``returns``,
    ``pnl`` or ``pnl_pct`` (case-insensitive).  If none is present, the first
    numeric column is used.  Values are interpreted as percent per trade.
    """
    df = pd.read_csv(path)
    lower = {c.lower(): c for c in df.columns}
    for cand in _CANDIDATE_COLUMNS:
        if cand in lower:
            return df[lower[cand]].dropna().to_numpy(dtype=float)
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] == 0:
        raise ValueError(f"No numeric column found in {path!r}")
    return numeric.iloc[:, 0].dropna().to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# Demo strategies (synthetic, seeded, honestly documented)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DemoStrategy:
    name: str
    returns: np.ndarray          # gross returns, percent per trade
    n_trials: int                # variants the author (honestly) evaluated
    base_cost_pct: float         # 1x transaction cost, percent per trade
    story: str


#: Mirage seed -- chosen (and verified by the test-suite) so the sample
#: really does show PF ~ 1.9 in-sample while failing the gates.
MIRAGE_SEED = 7
MODEST_SEED = 6


def generate_mirage(seed: int = MIRAGE_SEED) -> DemoStrategy:
    """The curve-fit mirage.

    Generating process (this *is* the trick, disclosed):

    * A high-win-rate scalper: ~85% of trades win a small amount
      (+0.50% +/- 0.12), the rest lose a larger amount (-1.35% +/- 0.25).
      That asymmetry prints a seductive profit factor around 1.8 while the
      mean edge per trade is tiny relative to the dispersion.
    * The win rate **decays fold by fold** (0.92 -> 0.70): the pattern was
      fitted to the early regime and is already rotting out-of-sample.
    * The author "researched" it by trying **500 parameter variants** and
      keeping the best -- the multiple-testing debt the Deflated Sharpe
      gate collects.
    * It trades small moves, so a realistic 0.10%-per-trade cost (1x)
      already eats most of the edge; at 2x costs it is underwater.
    """
    rng = np.random.default_rng(seed)
    n = 44
    folds = 5
    win_probs = np.linspace(0.92, 0.70, folds)
    returns = np.empty(n)
    per_fold = n // folds
    for i in range(n):
        fold = min(i // per_fold, folds - 1)
        if rng.random() < win_probs[fold]:
            returns[i] = rng.normal(0.50, 0.12)
        else:
            returns[i] = rng.normal(-1.35, 0.25)
    return DemoStrategy(
        name="Mirage (curve-fit scalper)",
        returns=returns,
        n_trials=500,
        base_cost_pct=0.10,
        story="In-sample PF ~1.9 on 44 trades after trying 500 variants.",
    )


def generate_modest_edge(seed: int = MODEST_SEED) -> DemoStrategy:
    """The modest real edge.

    Generating process: trade returns are drawn i.i.d. from
    Normal(mean = +0.18%, sd = 1.10%) -- a small but genuine positive drift
    injected into the process.  Proving a per-trade Sharpe of ~0.16 after
    correcting for 25 tried variants honestly takes a large sample, so
    the sample is **1,500 trades** (at 600 trades this same process would
    -- correctly -- come back UNPROVEN, not PASS).
    The author evaluated **25 variants** (an honest, modest search), and
    the 1x cost is 0.05% per trade, small relative to the moves traded.
    Unglamorous: PF only ~1.5.  But every gate should pass.
    """
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.18, 1.10, size=1500)
    return DemoStrategy(
        name="Modest Edge (small real drift)",
        returns=returns,
        n_trials=25,
        base_cost_pct=0.05,
        story="Per-trade Sharpe ~0.13 (net) with PF ~1.5 over 1,500 trades; 25 variants tried.",
    )


def demo_strategies() -> list[DemoStrategy]:
    return [generate_mirage(), generate_modest_edge()]
