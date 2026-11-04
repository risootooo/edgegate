"""Generate the two demo strategies as CSVs and print their headline stats.

Run from the repo root:  python examples/make_demo_data.py
Writes examples/mirage_trades.csv and examples/modest_edge_trades.csv --
feed either back through EdgeGate, e.g.:

    python main.py --csv examples/modest_edge_trades.csv --trials 25 --cost 0.05
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from edgegate.data import (  # noqa: E402
    generate_mirage,
    generate_modest_edge,
    profit_factor,
    sharpe_ratio,
)

HERE = Path(__file__).parent

for demo in (generate_mirage(), generate_modest_edge()):
    fname = "mirage_trades.csv" if "Mirage" in demo.name else "modest_edge_trades.csv"
    df = pd.DataFrame({"trade": range(1, len(demo.returns) + 1),
                       "return_pct": demo.returns})
    df.to_csv(HERE / fname, index=False)
    print(f"{demo.name}: n={len(demo.returns)}  gross PF={profit_factor(demo.returns):.2f}  "
          f"Sharpe/trade={sharpe_ratio(demo.returns):.3f}  -> {fname}")
    print(f"   {demo.story}")
