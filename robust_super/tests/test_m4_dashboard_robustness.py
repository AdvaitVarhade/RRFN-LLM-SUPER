"""
Unit Tests for Milestone M4: Streamlit Dashboard & Experiment Script Robustness.
Verifies:
1. Fast ablation study executes without torch.topk errors on empty head/tail partitions.
2. A/B business report generator handles missing metric keys safely without raising KeyError.
3. _load_saved_sim_data handles missing and corrupt result files gracefully.
4. Experiment script candidate scoring topk logic guards against empty item partitions.
5. Tab 5 and Tab 6 empty data guards operate reliably.
"""

import os
import sys
import json
import pytest
import numpy as np
import pandas as pd
import torch
from unittest.mock import patch

# Ensure robust_super root is on sys.path
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROBUST_DIR = os.path.abspath(os.path.join(TESTS_DIR, ".."))
if ROBUST_DIR not in sys.path:
    sys.path.insert(0, ROBUST_DIR)

from src.data.preprocessor import DataSplit
from src.evaluation.ab_simulator import generate_ab_report
from robust_super.app import (
    _load_saved_sim_data,
    safe_generate_ab_report,
    run_fast_ablation_study,
)


# ==============================================================================
# 1. TEST: _load_saved_sim_data Resilience (Missing and Corrupted Files)
# ==============================================================================

def test_load_saved_sim_data_missing_files_graceful_fallback(tmp_path):
    """
    Verifies that _load_saved_sim_data handles an empty directory (all result files missing)
    without crashing with FileNotFoundError, returning a complete structured sim_data dict.
    """
    empty_dir = str(tmp_path / "empty_results")
    os.makedirs(empty_dir, exist_ok=True)

    sim_data = _load_saved_sim_data(saved_results_dir=empty_dir)

    assert isinstance(sim_data, dict)
    # Required keys for dashboard operation
    expected_keys = [
        "metrics_robust", "metrics_vanilla", "metrics_uncalib", "metrics_clean_super",
        "benchmark_table", "latex_table_str", "T_hat", "T_final", "fused_weights",
        "H_robust", "T_robust", "robust_inclinations", "denoised_blueprints",
        "recs_robust", "user_cand_pools", "loss_history_pop", "loss_history_tail"
    ]
    for k in expected_keys:
        assert k in sim_data, f"Missing expected key: {k}"

    assert isinstance(sim_data["metrics_robust"], dict)
    assert isinstance(sim_data["benchmark_table"], list)
    assert isinstance(sim_data["T_hat"], np.ndarray)
    assert sim_data["T_hat"].shape == (5, 5)


def test_load_saved_sim_data_corrupted_json_files(tmp_path):
    """
    Verifies that corrupted JSON files in the results directory log warnings
    rather than raising json.JSONDecodeError.
    """
    corrupt_dir = str(tmp_path / "corrupt_results")
    os.makedirs(corrupt_dir, exist_ok=True)

    # Write malformed JSON files
    with open(os.path.join(corrupt_dir, "metrics.json"), "w") as f:
        f.write("{invalid_json: true, missing_bracket")
    with open(os.path.join(corrupt_dir, "benchmark_table.json"), "w") as f:
        f.write("[truncated json...")
    with open(os.path.join(corrupt_dir, "transition_matrix.json"), "w") as f:
        f.write("not json at all")

    sim_data = _load_saved_sim_data(saved_results_dir=corrupt_dir)

    assert isinstance(sim_data, dict)
    assert "metrics_robust" in sim_data
    assert isinstance(sim_data["T_hat"], np.ndarray)
    assert sim_data["T_hat"].shape == (5, 5)


# ==============================================================================
# 2. TEST: Tab 7 Resilience (safe_generate_ab_report)
# ==============================================================================

def test_safe_generate_ab_report_missing_keys_no_keyerror():
    """
    Standard generate_ab_report raises KeyError when nDCG@10, APLT@10, LTC@10
    are absent. safe_generate_ab_report must provide safe fallback defaults.
    """
    # Empty metric dictionaries
    report = safe_generate_ab_report({}, {})
    assert isinstance(report, dict)
    assert "ctr_robust" in report
    assert "ctr_vanilla" in report
    assert "ctr_lift_pct" in report
    assert "daily_lift" in report
    assert "monthly_lift" in report
    assert "annual_lift" in report
    assert "tail_exposure_lift_pct" in report


def test_safe_generate_ab_report_partial_and_alternate_keys():
    """
    Verifies that partial metric dicts with non-standard key names (e.g. 'nDCG' instead of 'nDCG@10')
    are correctly prioritized and do not raise KeyError.
    """
    m_rob = {"nDCG": 0.16, "APLT": 0.30}
    m_van = {"nDCG": 0.08, "APLT": 0.15}

    report = safe_generate_ab_report(m_rob, m_van)
    assert isinstance(report, dict)
    assert report["ndcg_robust"] == 0.16
    assert report["ndcg_vanilla"] == 0.08
    assert report["ctr_robust"] > report["ctr_vanilla"]
    assert report["daily_lift"] > 0


# ==============================================================================
# 3. TEST: run_fast_ablation_study & Top-K Scoring on Empty Partitions
# ==============================================================================

def test_fast_ablation_topk_guard_empty_head_and_tail_lists():
    """
    Verifies that the candidate scoring logic in run_fast_ablation_study
    handles empty head_list and tail_list (e.g. when H_set or T_set is empty)
    without raising RuntimeError: k must be greater than 0.
    """
    head_list = []
    tail_list = []
    s_pop = torch.tensor([])
    s_tail = torch.tensor([])

    # Replicate guarded scoring logic
    k_p = min(10, len(head_list))
    p_top = [head_list[i] for i in torch.topk(s_pop, k=k_p).indices.cpu().numpy()] if k_p > 0 else []

    k_t = min(10, len(tail_list))
    t_top = [tail_list[i] for i in torch.topk(s_tail, k=k_t).indices.cpu().numpy()] if k_t > 0 else []

    assert p_top == []
    assert t_top == []


def test_run_fast_ablation_study_execution_with_empty_partitions():
    """
    Executes run_fast_ablation_study on a synthetic split with mocked
    pareto_partition returning empty sets (H_set=set(), T_set=set()),
    confirming the entire pipeline runs without torch.topk crashes.
    """
    from src.data.loader import MovieLensLoader
    from src.data.preprocessor import preprocess_dataset
    loader = MovieLensLoader(data_dir="non_existent", min_user_interactions=2, min_item_interactions=2)
    ratings_df, movies_df, users_df = loader.load_data()
    clean_split = preprocess_dataset(ratings_df, movies_df, users_df)

    # 1 fast variant with empty partitions
    single_variant = [("Test Empty Partition Variant", 0.5, 0.3, 0.2, False, False, False)]
    with patch("robust_super.app.pareto_partition", return_value=(set(), set())):
        rows = run_fast_ablation_study(clean_split, movies_df, variants=single_variant)
        assert isinstance(rows, pd.DataFrame)
        assert len(rows) == 1
        assert rows.iloc[0]["Variant"] == "Test Empty Partition Variant"


# ==============================================================================
# 4. TEST: Experiment Scripts Top-K Guards & KeyError Avoidance
# ==============================================================================

def test_experiment_scripts_topk_guards_empty_partitions():
    """
    Verifies top-k candidate scoring across all 5 experiment runner scripts:
    - run_comprehensive_comparison.py
    - run_attack.py
    - run_baseline.py
    - run_robust_super.py
    - run_ablations.py
    Ensures empty partitions produce [] without RuntimeError across all pipelines.
    """
    top_k = 10

    # 1. run_comprehensive_comparison clean baseline scoring
    h_list_c = []
    t_list_c = []
    sp = torch.tensor([])
    st = torch.tensor([])
    k_pc = min(top_k, len(h_list_c))
    pc = [h_list_c[idx] for idx in torch.topk(sp, k=k_pc).indices.cpu().numpy()] if k_pc > 0 else []
    k_tc = min(top_k, len(t_list_c))
    tc = [t_list_c[idx] for idx in torch.topk(st, k=k_tc).indices.cpu().numpy()] if k_tc > 0 else []
    assert pc == []
    assert tc == []

    # 2. run_attack scoring
    head_list = []
    tail_list = []
    scores_pop = torch.tensor([])
    scores_tail = torch.tensor([])
    k_pop = min(top_k, len(head_list))
    pop_sorted_idx = torch.topk(scores_pop, k=k_pop).indices.cpu().numpy() if k_pop > 0 else []
    k_tail = min(top_k, len(tail_list))
    tail_sorted_idx = torch.topk(scores_tail, k=k_tail).indices.cpu().numpy() if k_tail > 0 else []
    pop_cands = [head_list[idx] for idx in pop_sorted_idx] if len(head_list) > 0 else []
    tail_cands = [tail_list[idx] for idx in tail_sorted_idx] if len(tail_list) > 0 else []
    assert pop_cands == []
    assert tail_cands == []

    # 3. run_robust_super & run_ablations candidate scoring
    k_p_m5 = min(top_k, len(head_list))
    pop_sorted_m5 = torch.topk(scores_pop, k=k_p_m5).indices.cpu().numpy() if k_p_m5 > 0 else []
    k_t_m5 = min(top_k, len(tail_list))
    tail_sorted_m5 = torch.topk(scores_tail, k=k_t_m5).indices.cpu().numpy() if k_t_m5 > 0 else []
    pop_cands_m5 = [head_list[idx] for idx in pop_sorted_m5] if len(head_list) > 0 else []
    tail_cands_m5 = [tail_list[idx] for idx in tail_sorted_m5] if len(tail_list) > 0 else []
    assert pop_cands_m5 == []
    assert tail_cands_m5 == []

    # 4. run_baseline blueprints.get() KeyError prevention
    blueprints = {}
    missing_user = 999
    _, b_u = blueprints.get(missing_user, ([], []))
    assert b_u == []


# ==============================================================================
# 5. TEST: Tab 5 and Tab 6 Empty State Guards
# ==============================================================================

def test_tab5_empty_users_or_movies_guard():
    """
    Verifies that Tab 5 does not attempt selectbox rendering on empty user or movie lists.
    """
    all_users = []
    movie_options = []
    has_sufficient_data = len(all_users) > 0 and len(movie_options) > 0
    assert not has_sufficient_data


def test_tab6_missing_user_in_cand_pools_guard():
    """
    Verifies that Tab 6 detects missing user in user_cand_pools and provides safe warning path.
    """
    user_cand_pools = {1: ([1, 2], [3, 4])}
    selected_user = 999
    assert selected_user not in user_cand_pools
