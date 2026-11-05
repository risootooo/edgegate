"""EdgeGate test suite.

The demo tests double as the project's honesty contract: if the Mirage
ever starts passing, or the Modest Edge ever starts failing, this suite
goes red.
"""

import numpy as np
import pytest
from scipy.stats import norm

from edgegate import (
    apply_costs,
    bootstrap_drawdowns,
    cost_stress,
    evaluate_strategy,
    generate_mirage,
    generate_modest_edge,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    sign_flip_permutation_test,
    walk_forward,
)
from edgegate.deflated_sharpe import (
    deflated_sharpe_probability,
    expected_max_sharpe,
)
from edgegate.report import build_report


# ----------------------------------------------------------------------
# Basic statistics (hand-checked)
# ----------------------------------------------------------------------

def test_profit_factor_hand_check():
    # wins 2 + 3 = 5, losses 1 + 1 = 2  ->  PF = 2.5
    assert profit_factor(np.array([2.0, -1.0, 3.0, -1.0])) == pytest.approx(2.5)


def test_profit_factor_no_losses_is_infinite():
    assert profit_factor(np.array([1.0, 2.0])) == float("inf")
    assert profit_factor(np.array([-1.0, -2.0])) == 0.0


def test_sharpe_hand_check():
    r = np.array([1.0, 2.0, 3.0])           # mean 2, std(ddof=1) = 1
    assert sharpe_ratio(r) == pytest.approx(2.0)
    assert sharpe_ratio(np.array([1.0, 1.0, 1.0])) == 0.0   # zero variance


def test_max_drawdown_hand_check():
    assert max_drawdown(np.array([5.0, -3.0, -4.0, 2.0])) == pytest.approx(7.0)
    assert max_drawdown(np.array([1.0, 1.0, 1.0])) == 0.0


# ----------------------------------------------------------------------
# Monte Carlo Permutation Test
# ----------------------------------------------------------------------

def test_permutation_detects_injected_edge():
    rng = np.random.default_rng(3)
    edge = rng.normal(0.30, 1.0, size=300)
    res = sign_flip_permutation_test(edge, n_permutations=2000, seed=1)
    assert res.p_value < 0.05


def test_permutation_rejects_pure_noise():
    rng = np.random.default_rng(11)
    noise = rng.normal(0.0, 1.0, size=300)
    res = sign_flip_permutation_test(noise, n_permutations=2000, seed=1)
    assert res.p_value > 0.05


def test_permutation_pvalue_bounds_and_determinism():
    r = np.random.default_rng(5).normal(0.1, 1.0, size=80)
    a = sign_flip_permutation_test(r, n_permutations=500, seed=42)
    b = sign_flip_permutation_test(r, n_permutations=500, seed=42)
    assert 0.0 < a.p_value <= 1.0
    assert a.p_value == b.p_value
    assert a.null_sharpes.shape == (500,)
    # Sign flips keep sum|x| fixed, so the PF and Sharpe orderings coincide.
    assert a.p_value_profit_factor == a.p_value


# ----------------------------------------------------------------------
# Deflated Sharpe Ratio
# ----------------------------------------------------------------------

def test_deflated_sharpe_hand_check_single_trial():
    # n_trials = 1 -> SR0 = 0, so DSR = Phi( SR * sqrt(T-1) / sqrt(1 + (kurt-1)/4 * SR^2) )
    # SR=0.5, T=101, skew=0, kurt=3:  z = 0.5*10 / sqrt(1.125) = 4.7140
    p = deflated_sharpe_probability(0.5, 101, skew=0.0, kurtosis=3.0, n_trials=1)
    assert p == pytest.approx(norm.cdf(4.7140), abs=1e-3)
    assert p > 0.9999


def test_deflated_sharpe_penalises_more_trials():
    p_few = deflated_sharpe_probability(0.20, 250, 0.0, 3.0, n_trials=5)
    p_many = deflated_sharpe_probability(0.20, 250, 0.0, 3.0, n_trials=500)
    assert p_many < p_few


def test_deflated_sharpe_rewards_more_data():
    p_short = deflated_sharpe_probability(0.20, 100, 0.0, 3.0, n_trials=10)
    p_long = deflated_sharpe_probability(0.20, 1000, 0.0, 3.0, n_trials=10)
    assert p_long > p_short


def test_expected_max_sharpe_grows_with_trials_and_is_zero_for_one():
    assert expected_max_sharpe(1, 500) == 0.0
    assert expected_max_sharpe(100, 500) > expected_max_sharpe(10, 500) > 0


# ----------------------------------------------------------------------
# Walk-forward
# ----------------------------------------------------------------------

def test_walkforward_all_positive_consistency_one():
    wf = walk_forward(np.full(100, 0.5), n_folds=5)
    assert wf.consistency == 1.0 and wf.passes and len(wf.folds) == 5


def test_walkforward_all_negative_consistency_zero():
    wf = walk_forward(np.full(100, -0.5), n_folds=5)
    assert wf.consistency == 0.0 and not wf.passes


# ----------------------------------------------------------------------
# Cost stress
# ----------------------------------------------------------------------

def test_cost_stress_monotonic_and_identity_at_zero():
    r = np.random.default_rng(2).normal(0.2, 1.0, size=120)
    cs = cost_stress(r, base_cost_pct=0.10)
    totals = [row.total_return for row in cs.rows]
    assert totals[0] > totals[1] > totals[2]
    assert np.allclose(apply_costs(r, 0.10, 0.0), r)


# ----------------------------------------------------------------------
# Bootstrap drawdowns
# ----------------------------------------------------------------------

def test_bootstrap_percentiles_ordered():
    r = np.random.default_rng(4).normal(0.05, 1.0, size=150)
    boot = bootstrap_drawdowns(r, n_resamples=300, seed=7)
    assert boot.p95_max_drawdown >= boot.median_max_drawdown >= 0.0
    assert boot.drawdowns.shape == (300,)


def test_bootstrap_constant_wins_have_zero_drawdown():
    boot = bootstrap_drawdowns(np.full(50, 0.4), n_resamples=50, seed=7)
    assert boot.realised_max_drawdown == 0.0
    assert boot.p95_max_drawdown == 0.0


# ----------------------------------------------------------------------
# Demo strategies -- the honesty contract
# ----------------------------------------------------------------------

def test_mirage_looks_great_gross():
    mirage = generate_mirage()
    assert len(mirage.returns) == 44
    assert profit_factor(mirage.returns) > 1.7          # the seductive bait


def test_modest_edge_is_positive_and_large_sample():
    modest = generate_modest_edge()
    assert len(modest.returns) >= 1000
    assert 1.3 < profit_factor(modest.returns) < 1.8    # unglamorous, real


def test_mirage_fails_the_gates():
    mirage = generate_mirage()
    rep = evaluate_strategy(mirage.name, mirage.returns, n_trials=mirage.n_trials,
                            base_cost_pct=mirage.base_cost_pct,
                            n_permutations=2000, n_resamples=500)
    assert rep.verdict == "FAIL"
    assert rep.permutation.p_value > 0.05
    assert rep.dsr_probability < 0.95
    assert not rep.costs.survives_2x


def test_modest_edge_passes_the_gates():
    modest = generate_modest_edge()
    rep = evaluate_strategy(modest.name, modest.returns, n_trials=modest.n_trials,
                            base_cost_pct=modest.base_cost_pct,
                            n_permutations=2000, n_resamples=500)
    assert rep.verdict == "PASS"
    assert rep.gates_passed == 4


def test_small_sample_is_unproven_no_matter_how_good():
    great_but_tiny = np.full(20, 1.0)          # 20 trades, zero losses
    rep = evaluate_strategy("Tiny sample", great_but_tiny, n_trials=1,
                            base_cost_pct=0.0, n_permutations=200, n_resamples=100)
    assert rep.verdict == "UNPROVEN"


def test_losing_strategy_fails():
    loser = np.random.default_rng(9).normal(-0.3, 1.0, size=200)
    rep = evaluate_strategy("Loser", loser, n_trials=1, base_cost_pct=0.05,
                            n_permutations=500, n_resamples=200)
    assert rep.verdict == "FAIL"


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------

def test_report_is_self_contained_html(tmp_path):
    mirage = generate_mirage()
    rep = evaluate_strategy(mirage.name, mirage.returns, n_trials=mirage.n_trials,
                            base_cost_pct=mirage.base_cost_pct,
                            n_permutations=200, n_resamples=100)
    out = build_report([rep], tmp_path / "report.html")
    text = out.read_text(encoding="utf-8")
    assert "FAIL" in text and "<svg" in text
    assert "http://" not in text and "https://" not in text   # fully offline


# ----------------------------------------------------------------------
# CSV loading (load_returns_csv -- the public entry point for own trades)
# ----------------------------------------------------------------------

from edgegate import load_returns_csv


def _write_csv(tmp_path, text):
    path = tmp_path / "trades.csv"
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_load_returns_csv_named_column(tmp_path):
    path = _write_csv(tmp_path, "trade,return_pct\n1,0.5\n2,-0.25\n3,1.0\n")
    assert np.allclose(load_returns_csv(path), [0.5, -0.25, 1.0])


def test_load_returns_csv_column_name_is_case_insensitive(tmp_path):
    path = _write_csv(tmp_path, "PnL\n0.5\n-0.25\n")
    assert np.allclose(load_returns_csv(path), [0.5, -0.25])


def test_load_returns_csv_candidate_priority(tmp_path):
    # return_pct outranks pnl in the candidate list, wherever it appears.
    path = _write_csv(tmp_path, "pnl,return_pct\n9.9,0.5\n8.8,-0.25\n")
    assert np.allclose(load_returns_csv(path), [0.5, -0.25])


def test_load_returns_csv_falls_back_to_first_numeric_column(tmp_path):
    path = _write_csv(tmp_path, "label,amount\nwin,0.5\nloss,-0.25\n")
    assert np.allclose(load_returns_csv(path), [0.5, -0.25])


def test_load_returns_csv_drops_missing_values(tmp_path):
    path = _write_csv(tmp_path, "return_pct\n0.5\n\n-0.25\n")
    assert np.allclose(load_returns_csv(path), [0.5, -0.25])


def test_load_returns_csv_no_numeric_column_raises(tmp_path):
    path = _write_csv(tmp_path, "label,note\nwin,good\nloss,bad\n")
    with pytest.raises(ValueError):
        load_returns_csv(path)


def test_load_returns_csv_bundled_examples():
    from pathlib import Path

    examples = Path(__file__).resolve().parents[1] / "examples"
    mirage = load_returns_csv(str(examples / "mirage_trades.csv"))
    modest = load_returns_csv(str(examples / "modest_edge_trades.csv"))
    assert len(mirage) == 44
    assert len(modest) == 1500
    # The bundled CSVs are the same series the generators produce.
    assert np.allclose(mirage, generate_mirage().returns)
    assert np.allclose(modest, generate_modest_edge().returns)
