"""
test_m5_adversarial_challenger.py
---------------------------------
Adversarial Stress Test Suite for Milestone M5 Verification & Audit.
Authored by Empirical Challenger teamwork_preview_challenger_m5_2.

Verifies:
1. Stage 1: Theorem 1 Tikhonov spectral bound analytical proof & adversarial stress test.
2. Stage 2: Theorem 2 Clean surrogate risk consistency bound & optimization behavior.
3. Stage 3: Multi-domain benchmark tables (MovieLens, Amazon, Yelp) empirical integrity.
4. Stage 4: 2D BLAS review-bombing burst detection vectorization speedup (>10x).
5. Stage 5: Table V 6-variant strict monotonic ablation invariants & mutation resilience.
6. Master reproduction script: Subprocess execution asserting exit code 0.
"""

import os
import sys
import json
import subprocess
import pytest
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "robust_super", "results")
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "robust_super", "experiments")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from robust_super.experiments.run_ablations import verify_monotonicity
from robust_super.src.bombing_detector.temporal_density import compute_temporal_acceleration


# ==============================================================================
# STAGE 1: THEOREM 1 TIKHONOV SPECTRAL BOUND PROOF & ADVERSARIAL STRESS TEST
# ==============================================================================

def test_stage1_theorem1_analytical_exact_bound():
    """
    Theorem 1: For any matrix T and regularization parameter lambda > 0,
    the Tikhonov regularized pseudo-inverse T_dagger = (T^T T + lambda I)^{-1} T^T
    satisfies:
        ||T_dagger||_2 <= 1 / (2 * sqrt(lambda))
    
    Analytical proof:
    Singular values of T_dagger are f(sigma) = sigma / (sigma^2 + lambda).
    d/d(sigma) [f(sigma)] = (lambda - sigma^2) / (sigma^2 + lambda)^2.
    Critical point is sigma = sqrt(lambda).
    f(sqrt(lambda)) = sqrt(lambda) / (lambda + lambda) = 1 / (2 * sqrt(lambda)).
    Since f(0) = 0 and lim_{sigma->inf} f(sigma) = 0, this is a global maximum on [0, inf).
    """
    for lam in [1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0]:
        sigmas = np.linspace(0.0, 100.0, 100000)
        f_vals = sigmas / (sigmas ** 2 + lam)
        max_f = np.max(f_vals)
        theoretical_bound = 1.0 / (2.0 * np.sqrt(lam))
        
        # Must be strictly bounded by 1 / (2 * sqrt(lambda))
        assert max_f <= theoretical_bound + 1e-9
        # Maximum must match theoretical bound within fine discretization tolerance
        assert np.isclose(max_f, theoretical_bound, rtol=1e-4)


def test_stage1_theorem1_adversarial_matrices():
    """
    Adversarially stress-tests Theorem 1 against:
    - Random stochastic transition matrices
    - Singular / rank-deficient matrices
    - Ill-conditioned matrices with condition number > 1e12
    - All-zeros degenerate matrix
    - Exact worst-case matrix with singular value exactly equal to sqrt(lambda)
    """
    rng = np.random.RandomState(42)
    lambdas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5]

    for lam in lambdas:
        theoretical_bound = 1.0 / (2.0 * np.sqrt(lam))

        # 1. Random stochastic matrices
        for _ in range(5):
            T = rng.dirichlet(np.ones(5), size=5)
            reg_inv = np.linalg.inv(T.T @ T + lam * np.eye(5))
            T_dagger = reg_inv @ T.T
            norm_2 = np.linalg.norm(T_dagger, 2)
            assert norm_2 <= theoretical_bound + 1e-6, f"Failed on random matrix: {norm_2} > {theoretical_bound}"

        # 2. Degenerate rank-1 matrix (all rows equal)
        T_rank1 = np.ones((5, 5)) * 0.2
        reg_inv = np.linalg.inv(T_rank1.T @ T_rank1 + lam * np.eye(5))
        T_dagger = reg_inv @ T_rank1.T
        norm_2 = np.linalg.norm(T_dagger, 2)
        assert norm_2 <= theoretical_bound + 1e-6

        # 3. All-zeros matrix
        T_zero = np.zeros((5, 5))
        reg_inv = np.linalg.inv(T_zero.T @ T_zero + lam * np.eye(5))
        T_dagger = reg_inv @ T_zero.T
        norm_2 = np.linalg.norm(T_dagger, 2)
        assert norm_2 == 0.0 <= theoretical_bound

        # 4. Ill-conditioned matrix (singular values spanning 1e12)
        U, _ = np.linalg.qr(rng.randn(5, 5))
        V, _ = np.linalg.qr(rng.randn(5, 5))
        S_ill = np.diag([1.0, 1e-2, 1e-5, 1e-8, 1e-12])
        T_ill = U @ S_ill @ V.T
        reg_inv = np.linalg.inv(T_ill.T @ T_ill + lam * np.eye(5))
        T_dagger = reg_inv @ T_ill.T
        norm_2 = np.linalg.norm(T_dagger, 2)
        assert norm_2 <= theoretical_bound + 1e-6

        # 5. Exact worst-case: one singular value exactly equals sqrt(lambda)
        S_worst = np.diag([np.sqrt(lam), 0.1, 0.05, 0.01, 0.001])
        T_worst = U @ S_worst @ V.T
        reg_inv = np.linalg.inv(T_worst.T @ T_worst + lam * np.eye(5))
        T_dagger = reg_inv @ T_worst.T
        norm_2 = np.linalg.norm(T_dagger, 2)
        # Should achieve exact equality up to machine precision
        assert np.isclose(norm_2, theoretical_bound, atol=1e-5)


# ==============================================================================
# STAGE 2: THEOREM 2 CLEAN SURROGATE RISK CONSISTENCY VERIFICATION
# ==============================================================================

def test_stage2_theorem2_consistency_error_bound():
    """
    Theorem 2: Given transition estimation error ||T_hat - T||_2 <= delta,
    the clean surrogate risk consistency upper bound is:
        E(lambda, delta) = delta / (2 * sqrt(lambda)) + sqrt(lambda)
    
    Verifies:
    1. Upper bound is strictly positive and finite for all delta > 0, lambda > 0.
    2. Optimum regularizer lambda* = delta / 2 minimizes the upper bound.
    3. Monotonic behavior on both sides of lambda*.
    """
    delta = 0.05
    for lam in [1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
        bound = (delta / (2.0 * np.sqrt(lam))) + np.sqrt(lam)
        assert bound > 0.0 and np.isfinite(bound)

    # Theoretical minimum is at lambda* = delta / 2 = 0.025
    lam_star = delta / 2.0
    min_bound = (delta / (2.0 * np.sqrt(lam_star))) + np.sqrt(lam_star)
    
    # Check that nearby lambdas have higher upper bounds
    for perturb in [-0.01, 0.01, 0.05]:
        test_lam = lam_star + perturb
        if test_lam > 0:
            test_bound = (delta / (2.0 * np.sqrt(test_lam))) + np.sqrt(test_lam)
            assert test_bound >= min_bound - 1e-7


# ==============================================================================
# STAGE 3: MULTI-DOMAIN BENCHMARK TABLES VERIFICATION
# ==============================================================================

def test_stage3_multidomain_benchmark_tables_integrity():
    """
    Verifies that all 3 domain benchmark tables exist, are non-empty, valid JSON,
    and fulfill performance claims.
    """
    # 1. MovieLens
    ml_file = os.path.join(RESULTS_DIR, "run_15epoch_gpu", "benchmark_table.json")
    assert os.path.exists(ml_file), f"Missing {ml_file}"
    with open(ml_file) as f:
        ml_data = json.load(f)
    assert len(ml_data) >= 4
    for row in ml_data:
        assert 0.0 <= row["Recall@10"] <= 1.0
        assert 0.0 <= row["nDCG@10"] <= 1.0
        assert 0.0 <= row["LTC@10"] <= 1.0
        assert row["Novelty"] > 0.0

    # MovieLens ROC-PR
    roc_file = os.path.join(RESULTS_DIR, "run_15epoch_gpu", "roc_pr.json")
    assert os.path.exists(roc_file), f"Missing {roc_file}"
    with open(roc_file) as f:
        roc_data = json.load(f)
    assert roc_data["roc_auc"] > 0.65
    assert roc_data["avg_precision"] > 0.95
    assert np.isclose(roc_data["roc_auc"], 0.7028, atol=1e-3)
    assert np.isclose(roc_data["avg_precision"], 0.9985, atol=1e-3)

    # 2. Amazon Reviews (SimGCL)
    amz_file = os.path.join(RESULTS_DIR, "run_amazon_simgcl", "benchmark_table.json")
    assert os.path.exists(amz_file), f"Missing {amz_file}"
    with open(amz_file) as f:
        amz_data = json.load(f)
    assert len(amz_data) >= 3
    amz_base = next(r for r in amz_data if "Base" in r["Method"])
    amz_ours = next(r for r in amz_data if "RRFN-LLM-SUPER" in r["Method"])
    # Assert improvement over base
    assert amz_ours["Recall@10"] > amz_base["Recall@10"]
    assert amz_ours["nDCG@10"] > amz_base["nDCG@10"]
    assert amz_ours["LTC@10"] > amz_base["LTC@10"]

    # 3. Yelp Academic (SGL)
    yelp_file = os.path.join(RESULTS_DIR, "run_yelp_sgl", "benchmark_table.json")
    assert os.path.exists(yelp_file), f"Missing {yelp_file}"
    with open(yelp_file) as f:
        yelp_data = json.load(f)
    assert len(yelp_data) >= 3
    yelp_base = next(r for r in yelp_data if "Base" in r["Method"])
    yelp_ours = next(r for r in yelp_data if "RRFN-LLM-SUPER" in r["Method"])
    # Assert improvement over base
    assert yelp_ours["Recall@10"] > yelp_base["Recall@10"]
    assert yelp_ours["nDCG@10"] > yelp_base["nDCG@10"]


# ==============================================================================
# STAGE 4: 2D BLAS BURST DETECTION SPEEDUP (>10X) VERIFICATION
# ==============================================================================

def test_stage4_vectorization_benchmark_artifact():
    """
    Verifies vectorization_benchmark.json records latency and >10x speedup.
    """
    vec_path = os.path.join(RESULTS_DIR, "run_15epoch_gpu", "vectorization_benchmark.json")
    assert os.path.exists(vec_path), f"Missing {vec_path}"
    with open(vec_path) as f:
        v = json.load(f)
    
    assert "vectorized_latency_ms" in v
    assert "unvectorized_latency_ms" in v
    assert "speedup_factor" in v
    assert v["speedup_factor"] > 10.0, f"Expected >10x speedup, got {v['speedup_factor']}x"
    assert v["speedup_factor"] > 500.0  # Actually 539.1x


def test_stage4_vectorization_live_execution_correctness():
    """
    Live test running compute_temporal_acceleration on synthetic burst data:
    - Verifies execution completes in < 100ms
    - Verifies burst items receive significantly higher acceleration scores
    """
    import time
    rng = np.random.RandomState(42)
    num_users = 100
    base_ts = 1700000000

    train_dict = {}
    for u in range(num_users):
        interactions = []
        # Item 0 is burst-attacked (all within 1 hour)
        interactions.append((0, 5, base_ts + int(rng.randint(0, 3600))))
        # Items 1..10 are normal organic interactions spread over 60 days
        for i in range(1, 10):
            interactions.append((i, int(rng.randint(1, 6)), base_ts + int(rng.randint(0, 3600 * 24 * 60))))
        train_dict[u] = interactions

    t0 = time.perf_counter()
    scores = compute_temporal_acceleration(train_dict, time_window_hours=24)
    elapsed = time.perf_counter() - t0

    assert elapsed < 0.10, f"Vectorized burst detection took {elapsed:.4f}s, expected < 0.10s"
    
    # Burst item 0: all interactions should have maximum or high acceleration
    burst_scores = [scores[(u, 0)] for u in range(num_users)]
    assert np.mean(burst_scores) > 5.0, f"Burst item 0 should have high average acceleration, got {np.mean(burst_scores)}"

    # Organic items should have significantly lower acceleration
    organic_scores = [scores[(u, i)] for u in range(num_users) for i in range(1, 10) if (u, i) in scores]
    assert np.mean(burst_scores) > np.mean(organic_scores) * 2.0


# ==============================================================================
# STAGE 5: TABLE V 6-VARIANT STRICT MONOTONIC ABLATION INVARIANTS
# ==============================================================================

def test_stage5_table_v_ablation_invariants():
    """
    Verifies that Table V ablation_table.json has 6 variants and passes
    verify_monotonicity.
    """
    abl_path = os.path.join(RESULTS_DIR, "run_15epoch_gpu", "ablation_table.json")
    assert os.path.exists(abl_path), f"Missing {abl_path}"
    with open(abl_path) as f:
        abl = json.load(f)

    assert len(abl) == 6, f"Expected 6 variants, found {len(abl)}"
    
    # Must pass strict monotonicity verification
    verify_monotonicity(abl)


def test_stage5_ablation_adversarial_mutant_rejection():
    """
    Adversarial mutation test: construct 5 invalid mutant ablation tables
    and assert that verify_monotonicity rejects EVERY single mutant.
    """
    abl_path = os.path.join(RESULTS_DIR, "run_15epoch_gpu", "ablation_table.json")
    with open(abl_path) as f:
        clean_rows = json.load(f)

    # Mutant 1: w/o RRFN does NOT drop ROC-AUC (invariant 2 violated)
    import copy
    m1 = copy.deepcopy(clean_rows)
    for r in m1:
        if "w/o RRFN" in r["Variant"]:
            r["Denoising_ROC"] = 0.7028  # No drop!
    with pytest.raises(AssertionError):
        verify_monotonicity(m1)

    # Mutant 2: w/o LLM does not drop ROC-AUC (invariant 3 violated)
    m2 = copy.deepcopy(clean_rows)
    for r in m2:
        if "w/o LLM" in r["Variant"]:
            r["Denoising_ROC"] = 0.7500  # Higher than Full!
    with pytest.raises(AssertionError):
        verify_monotonicity(m2)

    # Mutant 3: w/o Bombing does not drop ROC-AUC (invariant 4 violated)
    m3 = copy.deepcopy(clean_rows)
    for r in m3:
        if "w/o Bombing" in r["Variant"]:
            r["Denoising_ROC"] = 0.6800  # Less than 15% drop
    with pytest.raises(AssertionError):
        verify_monotonicity(m3)

    # Mutant 4: w/o GCL does not drop LTC@10 (invariant 5 violated)
    m4 = copy.deepcopy(clean_rows)
    for r in m4:
        if "w/o GCL" in r["Variant"]:
            r["LTC@10"] = 0.2000  # Higher than Full!
    with pytest.raises(AssertionError):
        verify_monotonicity(m4)

    # Mutant 5: w/o SUPER does not collapse LTC@10 (invariant 6 violated)
    m5 = copy.deepcopy(clean_rows)
    for r in m5:
        if "w/o SUPER" in r["Variant"]:
            r["LTC@10"] = 0.1500  # No collapse
    with pytest.raises(AssertionError):
        verify_monotonicity(m5)


# ==============================================================================
# SUBPROCESS EXECUTION OF MASTER REPRODUCTION SCRIPT
# ==============================================================================

def test_master_reproduction_script_subprocess_exit_0():
    """
    Executes 'python paper/reproduce_all_experiments.py' via subprocess.
    Asserts:
    1. Exit code == 0.
    2. All 5 stage headers and PASSED confirmations appear in stdout.
    """
    script_path = os.path.join(PROJECT_ROOT, "paper", "reproduce_all_experiments.py")
    assert os.path.exists(script_path), f"Missing {script_path}"

    result = subprocess.run(
        [sys.executable, script_path],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True
    )

    assert result.returncode == 0, f"Script failed with code {result.returncode}:\n{result.stderr}"
    stdout = result.stdout

    # Assert Stage 1 & 2
    assert "MATHEMATICAL AUDIT: THEOREM 1 & THEOREM 2 PROOFS" in stdout
    assert "Theorems 1 and 2 are strictly verified mathematically." in stdout

    # Assert Stage 3
    assert "EMPIRICAL DATA VERIFICATION: MOVIELENS, AMAZON, YELP BENCHMARKS" in stdout
    assert "All empirical benchmark tables successfully loaded and verified with zero discrepancies." in stdout

    # Assert Stage 4
    assert "VERIFICATION: VECTORIZATION SPEEDUP & A/B BUSINESS SIMULATION" in stdout
    assert "Speedup: 539.1x over loop" in stdout

    # Assert Stage 5
    assert "VERIFICATION: COMPONENT ABLATION STUDY" in stdout
    assert "All 6 strict monotonic invariants verified successfully!" in stdout
    assert "Ablation study verified successfully with strict monotonic invariants." in stdout
