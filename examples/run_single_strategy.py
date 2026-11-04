"""Minimal library usage: evaluate one return series with EdgeGate.

Run from the repo root:  python examples/run_single_strategy.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from edgegate import evaluate_strategy  # noqa: E402

# Your own trades would come from a CSV (edgegate.load_returns_csv);
# here we synthesise a borderline series on the spot.
rng = np.random.default_rng(0)
my_returns = rng.normal(0.10, 1.0, size=180)   # percent per trade

report = evaluate_strategy(
    "My backtest", my_returns, n_trials=40, base_cost_pct=0.06,
    n_permutations=2000, n_resamples=1000,
)
for gate in report.gates:
    print(f"[{'PASS' if gate.passed else 'FAIL'}] {gate.name}: {gate.detail}")
print(f"VERDICT: {report.verdict}")
