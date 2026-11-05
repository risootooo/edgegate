# EdgeGate

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Tests](https://img.shields.io/badge/tests-30%20passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)
![Dependencies](https://img.shields.io/badge/deps-numpy%20%C2%B7%20pandas%20%C2%B7%20scipy-blue)
![Report](https://img.shields.io/badge/report-self--contained%20HTML-purple)

## Your backtest is lying to you. EdgeGate proves it in 60 seconds.

EdgeGate is a **strategy-validation lab**. Feed it a strategy's trade list
and it runs a battery of statistical gates — Monte Carlo permutation test,
Deflated Sharpe Ratio, walk-forward consistency, bootstrap drawdowns and a
cost-stress test — then issues a single honest verdict:

**PASS** · **FAIL** · **UNPROVEN**

No verdict is bought with a pretty equity curve. A strategy must pass
*every* gate before EdgeGate will say it deserves real money.

> 📊 **See it now:** open [`docs/report.html`](docs/report.html) — the full
> demo report (dark, self-contained, works offline, charts are inline SVG
> generated from the real computed distributions).

---

## The killer demo: Mirage vs Modest Edge

Two synthetic strategies are bundled, with their generating processes
**disclosed in the code** (`src/edgegate/data.py`). One is a trap. One is
real. Your eye cannot tell them apart from their headline stats — the
gates can.

| | **Mirage** (curve-fit scalper) | **Modest Edge** (small real drift) |
|---|---|---|
| Trades | 44 | 1,500 |
| Variants the author tried | 500 | 25 |
| Profit factor (gross) | **1.93** — looks *better* | 1.56 |
| Profit factor (net of 1× costs) | 1.41 | 1.39 |
| Sharpe per trade (net) | 0.137 | 0.133 |
| **MCPT p-value** | **0.182** ❌ | **0.0002** ✅ |
| **Deflated Sharpe probability** | **0.028** ❌ | **0.999** ✅ |
| **Walk-forward** | 3/5 folds profitable (60%) ✅ | **5/5 folds (100%)** ✅ |
| **At 2× costs** | total −0.41%, PF 0.96 ❌ | total +145.37%, PF 1.24 ✅ |
| Realised max drawdown | 3.8% | 15.6% |
| Bootstrap 95th-pct max drawdown | 8.9% | 27.4% |
| **VERDICT** | 🔴 **FAIL** (1/4 gates) | 🟢 **PASS** (4/4 gates) |

The Mirage has the *higher* gross profit factor. It is also worthless:
its result is indistinguishable from sign-flip luck (p = 0.18), its
Sharpe evaporates once you charge it for the 500 variants that were tried
before it (DSR probability 0.028), and doubling transaction costs puts it
underwater. The Modest Edge is uglier on every vanity metric — and passes
everything, because its edge is actually in the generating process
(i.i.d. Normal(+0.18%, 1.10%) per trade, disclosed in the source).

Notice the honest detail the gates surface about the *passing* strategy
too: its realised 15.6% drawdown understates the bootstrap 95th
percentile of 27.4%. EdgeGate is not a rubber stamp for strategies it
likes; it prices the risk of the ones it passes.

*All numbers above are reproduced by `python main.py` and asserted by the
test suite — if the Mirage ever passes or the Modest Edge ever fails,
the tests go red.*

---

## Quickstart

```bash
git clone https://github.com/risootooo/edgegate.git
cd edgegate
pip install -r requirements.txt        # numpy, pandas, scipy — that's all

python main.py                         # both demos → docs/report.html
open docs/report.html                  # (or xdg-open on Linux)
```

Validate **your own** strategy (CSV with a `return_pct` column, percent
per trade):

```bash
python main.py --csv my_trades.csv --trials 50 --cost 0.08 --name "My strategy"
```

`--trials` is the question nobody wants to answer honestly: *how many
variants did you try before this one?* The Deflated Sharpe gate charges
you for every one of them. `--cost` is your 1× per-trade transaction
cost in percent; the stress gate re-runs you at 2× and 3×.

As a library:

```python
from edgegate import evaluate_strategy, load_returns_csv

report = evaluate_strategy(
    "My strategy", load_returns_csv("my_trades.csv"),
    n_trials=50, base_cost_pct=0.08,
)
print(report.verdict)   # PASS / FAIL / UNPROVEN
```

---

## How it works — the five analyses

### 1. Monte Carlo Permutation Test (`permutation.py`)
If a strategy has no edge, the *signs* of its trade returns are
arbitrary. EdgeGate flips every trade's sign at random (5,000 times),
keeping magnitudes — position sizing and trade selection survive, only
the claimed direction is destroyed — and builds the null distribution of
Sharpe ratios. The p-value is the fraction of sign-luck worlds that do
at least as well as you did, with the standard +1 correction. Because
sign flips preserve Σ|x|, the profit-factor and Sharpe orderings
coincide, and both p-values are reported.
*After Timothy Masters' Monte Carlo permutation testing for trading systems.*

### 2. Deflated Sharpe Ratio (`deflated_sharpe.py`)
The Sharpe you selected is the maximum of every variant you tried, and
maxima are biased upward. The DSR (Bailey & López de Prado, 2014)
computes the probability that the true Sharpe exceeds zero after
correcting for the number of trials, sample length, skewness and
kurtosis:

```
SR0 = sqrt(1/(T-1)) · [ (1-γ)·Z⁻¹(1 − 1/N) + γ·Z⁻¹(1 − 1/(N·e)) ]
DSR = Φ( (SR − SR0)·sqrt(T−1) / sqrt(1 − skew·SR + ((kurt−1)/4)·SR²) )
```

### 3. Walk-Forward Consistency (`walkforward.py`)
The series is cut into 5 consecutive folds — **never shuffled**, time
order is the point. A real edge appears in most folds; a curve-fit lives
in one lucky window. Gate: ≥ 60% of folds profitable.

### 4. Bootstrap Drawdown Distribution (`bootstrap.py`)
One realised equity curve shows one drawdown: the lucky one. Circular
block bootstrap (Politis & Romano, 1992; block length 5) rebuilds 2,000
plausible alternative curves and reports the median and **95th
percentile** maximum drawdown — the number to size positions against.

### 5. Cost-Stress Test (`costs.py`)
Net results are recomputed at 1×, 2× and 3× the stated per-trade cost.
Gate: still profitable with PF > 1 at **2× costs**. High-turnover
strategies with tiny average wins — the classic scalper mirage — are hit
hardest, exactly as they are live.

### The verdict engine (`verdict.py`)
| Verdict | Meaning |
|---|---|
| 🔴 **FAIL** | Net-negative, or ≤ 1 gate passed, or fails both MCPT *and* cost stress — no detectable, executable edge |
| 🟢 **PASS** | All four gates passed on an adequate sample |
| 🟡 **UNPROVEN** | Fewer than 30 trades (no statistic rescues that sample), or mixed gate evidence — collect more trades, trade nothing yet |

All gates are evaluated on the net (1×-cost) series; cost survival is
judged at 2×.

---

## Architecture

```
src/edgegate/
├── data.py             # demo generators (disclosed processes) + core statistics
├── permutation.py      # MCPT (sign-flip null distribution)
├── deflated_sharpe.py  # Bailey & López de Prado DSR
├── walkforward.py      # fold consistency
├── bootstrap.py        # circular block-bootstrap drawdowns
├── costs.py            # 1×/2×/3× cost stress
├── verdict.py          # gate engine → StrategyReport
├── report.py           # self-contained dark HTML report (inline SVG)
└── __main__.py         # CLI
examples/               # demo CSV generator + minimal library usage
tests/                  # 23 pytest tests, incl. the Mirage/Modest honesty contract
docs/report.html        # the generated demo report — open it
```

## Limitations (stated, not buried)

- **Gates test a trade series, not a trading system.** Look-ahead bias,
  survivorship bias and bad fills live upstream of the trade list, and no
  statistic here can detect them. Garbage trade list in, rigorous
  garbage out.
- **The DSR is only as honest as your `--trials` number.** If you tried
  500 variants and report 5, the gate believes you. EdgeGate makes
  dishonesty *visible*, not impossible.
- **The permutation null assumes trade signs are exchangeable under no
  edge.** Strongly autocorrelated trade streams weaken the test; the
  block bootstrap partially compensates for drawdowns, not for the MCPT.
- **Passing is not a promise.** A PASS means the evidence in this sample
  survives these corrections — edges decay, regimes change, and the
  bootstrap 95th-percentile drawdown is a planning number, not a bound.
- Demo data is **synthetic by design**; the demo verdicts demonstrate the
  gates, not any market claim.

## References

- Bailey, D. H. & López de Prado, M. (2014). *The Deflated Sharpe Ratio:
  Correcting for Selection Bias, Backtest Overfitting and Non-Normality.*
  Journal of Portfolio Management 40(5), 94–107.
- Masters, T. (2006). *Monte Carlo evaluation of trading systems* and
  related work on permutation testing for strategy validation.
- Politis, D. N. & Romano, J. P. (1992). *A Circular Moving Blocks
  Bootstrap for Stationary Data.* 

## License

MIT — see [LICENSE](LICENSE).

---

*Research software for strategy validation. Nothing here is financial
advice, and a PASS from EdgeGate is not a recommendation to trade
anything.*
