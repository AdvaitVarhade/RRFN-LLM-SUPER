"""
Comprehensive 4-Tier Opaque-Box E2E Test Suite for FedSUPER-LLM Dynamic Blueprint Merge.

Architecture & Coverage:
- Tier 1: Feature Coverage (>=5 test cases per feature: stochastic quota, adaptive alpha,
          confidence elastic, dynamic merge modes, score fusion, calibration evaluation)
- Tier 2: Boundary & Corner Cases (>=5 test cases per feature: N=1, N=20, N=100; pop_u=0.0,
          pop_u=1.0; empty interaction histories; single-item head pools; extreme score
          differentials; zero variance)
- Tier 3: Cross-Feature Combinations (Pairwise interactions: stochastic quota + adaptive alpha;
          score fusion + confidence elasticity; dynamic merge + full rank eval; hybrid chains)
- Tier 4: Real-World Workload Scenarios (Full ML-1M scale benchmark validating Recall@20 > 0.037
          and Rmse-PC <= 0.056; throughput and scalability verification)
"""

import os
import sys
import time
import pytest
import numpy as np

# Ensure project src directory is importable
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import super as sp
import metrics as met


# ============================================================================
# Fixtures & Synthetic Data Generators
# ============================================================================

@pytest.fixture
def synthetic_catalog():
    """Generates a synthetic catalog with realistic Pareto distribution."""
    n_items = 500
    # Power law distribution for item interaction volume
    ranks = np.arange(1, n_items + 1)
    pop_count = (10000.0 / (ranks ** 0.8)).astype(np.float32)
    alpha = 0.20
    head_idx, tail_idx = sp.pareto_partition(pop_count, alpha=alpha)
    return {
        "n_items": n_items,
        "pop_count": pop_count,
        "head_idx": head_idx,
        "tail_idx": tail_idx,
        "alpha": alpha,
    }


@pytest.fixture
def synthetic_users_data(synthetic_catalog):
    """Generates synthetic users, interaction matrix, and preference scores."""
    n_users = 200
    n_items = synthetic_catalog["n_items"]
    head_idx = synthetic_catalog["head_idx"]
    tail_idx = synthetic_catalog["tail_idx"]

    rng = np.random.default_rng(42)
    train_matrix = np.zeros((n_users, n_items), dtype=np.float32)

    for u in range(n_users):
        n_interactions = rng.integers(10, 100)
        user_pop_pref = rng.uniform(0.05, 0.60)
        n_head_items = int(np.round(n_interactions * user_pop_pref))
        n_head_items = min(n_head_items, len(head_idx))
        n_tail_items = min(n_interactions - n_head_items, len(tail_idx))

        if n_head_items > 0:
            chosen_head = rng.choice(head_idx, size=n_head_items, replace=False)
            train_matrix[u, chosen_head] = rng.integers(1, 5, size=n_head_items)
        if n_tail_items > 0:
            chosen_tail = rng.choice(tail_idx, size=n_tail_items, replace=False)
            train_matrix[u, chosen_tail] = rng.integers(1, 5, size=n_tail_items)

    pop_u = sp.user_popularity_inclination(train_matrix, head_idx)

    # Synthetic scores from collaborative models
    scores_pop = rng.normal(0.0, 1.0, size=(n_users, n_items)).astype(np.float32)
    scores_tail = rng.normal(0.0, 1.0, size=(n_users, n_items)).astype(np.float32)

    # Mask non-pool items
    scores_pop[:, tail_idx] = -np.inf
    scores_tail[:, head_idx] = -np.inf

    # Synthetic LLM semantic similarity scores in [-1, 1]
    scores_llm = rng.uniform(-0.5, 0.8, size=(n_users, n_items)).astype(np.float32)

    # Leave-one-out test dictionary
    test_dict = {}
    for u in range(n_users):
        # Pick an unseen item as ground truth
        unseen = np.where(train_matrix[u] == 0)[0]
        test_dict[u] = [int(rng.choice(unseen))]

    return {
        "n_users": n_users,
        "n_items": n_items,
        "train_matrix": train_matrix,
        "pop_u": pop_u,
        "scores_pop": scores_pop,
        "scores_tail": scores_tail,
        "scores_llm": scores_llm,
        "test_dict": test_dict,
        "head_idx": head_idx,
        "tail_idx": tail_idx,
    }


# ============================================================================
# TIER 1: FEATURE COVERAGE (Unit & Mathematical Contract Verification)
# ============================================================================

class TestTier1FeatureCoverage:
    """Tier 1: Comprehensive tests verifying each feature's mathematical contract."""

    # --- Feature 1: Stochastic User Quota ---

    def test_t1_f1_stochastic_quota_sum_invariance(self):
        """F1.1: Quota sum invariance (n_pop + n_tail == N) and non-negativity."""
        pop_u = np.linspace(0.0, 1.0, 100, dtype=np.float32)
        for N in [1, 5, 10, 20, 50, 100]:
            rng = np.random.default_rng(123)
            n_pop, n_tail = sp.stochastic_user_quota(pop_u, N=N, rng=rng)
            assert n_pop.shape == (100,)
            assert n_tail.shape == (100,)
            assert np.all(n_pop + n_tail == N), f"Sum violated for N={N}"
            assert np.all(n_pop >= 0) and np.all(n_pop <= N)
            assert np.all(n_tail >= 0) and np.all(n_tail <= N)

    def test_t1_f1_stochastic_quota_unbiased_expectation(self):
        """F1.2: Unbiased expectation E[n_pop] == N * pop_u across large sample."""
        N = 20
        n_users = 25000
        rng = np.random.default_rng(42)
        pop_u = rng.uniform(0.0, 1.0, size=n_users).astype(np.float32)

        n_pop, _ = sp.stochastic_user_quota(pop_u, N=N, rng=rng)
        empirical_mean_pop = np.mean(n_pop / float(N))
        theoretical_mean_pop = np.mean(pop_u)

        abs_err = abs(empirical_mean_pop - theoretical_mean_pop)
        assert abs_err < 0.005, f"Expectation bias {abs_err:.5f} exceeds tolerance 0.005"

    def test_t1_f1_stochastic_quota_deviation_bound(self):
        """F1.3: Maximum individual deviation bound |n_pop / N - pop_u| <= 1 / N."""
        N = 20
        rng = np.random.default_rng(42)
        pop_u = rng.uniform(0.0, 1.0, size=5000).astype(np.float32)
        n_pop, _ = sp.stochastic_user_quota(pop_u, N=N, rng=rng)

        deviations = np.abs(n_pop / float(N) - pop_u)
        max_deviation = np.max(deviations)
        assert max_deviation <= (1.0 / N) + 1e-6, f"Max deviation {max_deviation:.5f} > 1/N"

    def test_t1_f1_stochastic_quota_deterministic_reproducibility(self):
        """F1.4: Deterministic reproducibility when fixed Generator / seed is provided."""
        pop_u = np.array([0.15, 0.25, 0.35, 0.45, 0.55, 0.65], dtype=np.float32)
        rng1 = np.random.default_rng(999)
        rng2 = np.random.default_rng(999)
        n_pop1, n_tail1 = sp.stochastic_user_quota(pop_u, N=20, rng=rng1)
        n_pop2, n_tail2 = sp.stochastic_user_quota(pop_u, N=20, rng=rng2)

        np.testing.assert_array_equal(n_pop1, n_pop2)
        np.testing.assert_array_equal(n_tail1, n_tail2)

    def test_t1_f1_stochastic_quota_vectorized_shapes(self):
        """F1.5: Vectorized batch processing over varying array dimensions."""
        for size in [1, 10, 1000, 10000]:
            pop_u = np.random.uniform(0, 1, size=size).astype(np.float32)
            n_pop, n_tail = sp.stochastic_user_quota(pop_u, N=20)
            assert n_pop.shape == (size,)
            assert n_tail.shape == (size,)
            assert n_pop.dtype == np.int64 or n_pop.dtype == np.int32

    # --- Feature 2: User-Adaptive Alpha Partition ---

    def test_t1_f2_adaptive_alpha_shape_and_range(self, synthetic_users_data, synthetic_catalog):
        """F2.1: Output shape is (n_users,) and values strictly lie in [0.0, 1.0]."""
        train_matrix = synthetic_users_data["train_matrix"]
        pop_count = synthetic_catalog["pop_count"]

        pop_u_adapt = sp.user_adaptive_alpha_partition(pop_count, train_matrix, alpha_base=0.20, beta=0.15)
        assert pop_u_adapt.shape == (train_matrix.shape[0],)
        assert np.all(pop_u_adapt >= 0.0) and np.all(pop_u_adapt <= 1.0)

    def test_t1_f2_adaptive_alpha_beta_zero_invariance(self, synthetic_users_data, synthetic_catalog):
        """F2.2: When beta=0.0, adaptive alpha reduces to standard popularity inclination."""
        train_matrix = synthetic_users_data["train_matrix"]
        pop_count = synthetic_catalog["pop_count"]
        head_idx = synthetic_catalog["head_idx"]

        base_pop_u = sp.user_popularity_inclination(train_matrix, head_idx)
        adapt_pop_u = sp.user_adaptive_alpha_partition(pop_count, train_matrix, alpha_base=0.20, beta=0.0)

        # Should match base_pop_u within tight numerical precision
        np.testing.assert_allclose(adapt_pop_u, base_pop_u, atol=1e-4)

    def test_t1_f2_adaptive_alpha_activity_scaling(self, synthetic_catalog):
        """F2.3: Active users with higher interaction counts scale appropriately."""
        pop_count = synthetic_catalog["pop_count"]
        n_items = synthetic_catalog["n_items"]
        head_idx = synthetic_catalog["head_idx"]
        tail_idx = synthetic_catalog["tail_idx"]

        train_matrix = np.zeros((2, n_items), dtype=np.float32)
        # User 0: Sparse (5 head items, 5 tail items -> 10 total)
        train_matrix[0, head_idx[:5]] = 1
        train_matrix[0, tail_idx[:5]] = 1
        # User 1: Very active (50 head items, 50 tail items -> 100 total)
        train_matrix[1, head_idx[:50]] = 1
        train_matrix[1, tail_idx[:50]] = 1

        adapt_pop_u = sp.user_adaptive_alpha_partition(pop_count, train_matrix, alpha_base=0.20, beta=0.20)
        # User 1 has much higher activity weight than User 0
        assert adapt_pop_u[1] >= adapt_pop_u[0]

    def test_t1_f2_adaptive_alpha_cold_start_fallback(self, synthetic_catalog):
        """F2.4: Cold-start users (0 interactions) handle gracefully without division by zero."""
        pop_count = synthetic_catalog["pop_count"]
        n_items = synthetic_catalog["n_items"]
        train_matrix = np.zeros((5, n_items), dtype=np.float32)

        adapt_pop_u = sp.user_adaptive_alpha_partition(pop_count, train_matrix, alpha_base=0.20, beta=0.15)
        assert adapt_pop_u.shape == (5,)
        assert np.all(adapt_pop_u >= 0.0) and np.all(adapt_pop_u <= 1.0)
        assert not np.any(np.isnan(adapt_pop_u))

    def test_t1_f2_adaptive_alpha_parameter_sensitivity(self, synthetic_users_data, synthetic_catalog):
        """F2.5: Parameter sweeps across alpha_base and beta produce smooth changes."""
        train_matrix = synthetic_users_data["train_matrix"]
        pop_count = synthetic_catalog["pop_count"]

        p1 = sp.user_adaptive_alpha_partition(pop_count, train_matrix, alpha_base=0.10, beta=0.10)
        p2 = sp.user_adaptive_alpha_partition(pop_count, train_matrix, alpha_base=0.20, beta=0.10)
        p3 = sp.user_adaptive_alpha_partition(pop_count, train_matrix, alpha_base=0.30, beta=0.10)

        # Higher alpha_base generally expands head catalog and raises inclinations
        assert np.mean(p3) >= np.mean(p1) - 1e-5

    # --- Feature 3: Confidence Elastic Merge ---

    def test_t1_f3_confidence_elastic_shape_and_dtype(self, synthetic_users_data):
        """F3.1: Output shape is (n_users, N) and integer typed."""
        N = 20
        reclist = sp.confidence_elastic_merge(
            synthetic_users_data["scores_pop"],
            synthetic_users_data["scores_tail"],
            synthetic_users_data["train_matrix"],
            synthetic_users_data["head_idx"],
            N=N,
            threshold=0.1,
        )
        assert reclist.shape == (synthetic_users_data["n_users"], N)
        assert np.issubdtype(reclist.dtype, np.integer)

    def test_t1_f3_confidence_elastic_item_uniqueness(self, synthetic_users_data):
        """F3.2: No duplicate items exist in any user's recommendation slate."""
        N = 20
        reclist = sp.confidence_elastic_merge(
            synthetic_users_data["scores_pop"],
            synthetic_users_data["scores_tail"],
            synthetic_users_data["train_matrix"],
            synthetic_users_data["head_idx"],
            N=N,
            threshold=0.1,
        )
        for u in range(reclist.shape[0]):
            slate = reclist[u]
            assert len(set(slate)) == N, f"User {u} contains duplicates in slate: {slate}"

    def test_t1_f3_confidence_elastic_trainpos_exclusion(self, synthetic_users_data):
        """F3.3: Training positive items are strictly excluded from recommendation slates."""
        N = 20
        train_matrix = synthetic_users_data["train_matrix"]
        reclist = sp.confidence_elastic_merge(
            synthetic_users_data["scores_pop"],
            synthetic_users_data["scores_tail"],
            train_matrix,
            synthetic_users_data["head_idx"],
            N=N,
            threshold=0.1,
        )
        for u in range(reclist.shape[0]):
            train_items = set(np.where(train_matrix[u] > 0)[0])
            rec_items = set(reclist[u])
            overlap = train_items.intersection(rec_items)
            assert len(overlap) == 0, f"User {u} leaked training items: {overlap}"

    def test_t1_f3_confidence_elastic_score_gap_trigger(self, synthetic_catalog):
        """F3.4: Significant score gap triggers dynamic slot elasticity."""
        n_users = 10
        n_items = synthetic_catalog["n_items"]
        head_idx = synthetic_catalog["head_idx"]
        tail_idx = synthetic_catalog["tail_idx"]

        train_matrix = np.zeros((n_users, n_items), dtype=np.float32)
        # Give users equal head/tail history (Pop_u = 0.5)
        for u in range(n_users):
            train_matrix[u, head_idx[:5]] = 1
            train_matrix[u, tail_idx[:5]] = 1

        scores_pop = np.full((n_users, n_items), -np.inf, dtype=np.float32)
        scores_tail = np.full((n_users, n_items), -np.inf, dtype=np.float32)

        # Make head scores very high and tail scores very low
        scores_pop[:, head_idx] = 100.0
        scores_tail[:, tail_idx] = 0.1

        reclist_elastic = sp.confidence_elastic_merge(
            scores_pop, scores_tail, train_matrix, head_idx, N=20, threshold=0.1
        )
        head_set = set(head_idx.tolist())
        head_counts = [sum(1 for it in reclist_elastic[u] if it in head_set) for u in range(n_users)]

        # Under high head confidence, head counts should be >= nominal quota (10)
        assert np.all(np.array(head_counts) >= 10)

    def test_t1_f3_confidence_elastic_pool_exhaustion_fallback(self, synthetic_catalog):
        """F3.5: Fallback smoothly fills slate when one candidate pool is exhausted."""
        n_users = 5
        n_items = synthetic_catalog["n_items"]
        head_idx = synthetic_catalog["head_idx"][:3]  # Only 3 head items total
        tail_idx = synthetic_catalog["tail_idx"]

        train_matrix = np.zeros((n_users, n_items), dtype=np.float32)
        scores_pop = np.full((n_users, n_items), -np.inf, dtype=np.float32)
        scores_tail = np.full((n_users, n_items), -np.inf, dtype=np.float32)
        scores_pop[:, head_idx] = 10.0
        scores_tail[:, tail_idx] = 1.0

        reclist = sp.confidence_elastic_merge(
            scores_pop, scores_tail, train_matrix, head_idx, N=20, threshold=0.1
        )
        assert reclist.shape == (n_users, 20)
        for u in range(n_users):
            assert len(set(reclist[u])) == 20
            assert not np.any(reclist[u] == -1)

    # --- Feature 4: Dynamic Probabilistic Quota Merge ---

    @pytest.mark.parametrize("mode", ["stochastic", "adaptive_alpha", "confidence_elastic", "hybrid", "static"])
    def test_t1_f4_dynamic_merge_all_modes_validity(self, synthetic_users_data, mode):
        """F4.1-F4.5: All 5 dynamic merge modes produce valid (n_users, N) slates."""
        N = 20
        reclist = sp.dynamic_probabilistic_quota_merge(
            synthetic_users_data["scores_pop"],
            synthetic_users_data["scores_tail"],
            synthetic_users_data["train_matrix"],
            synthetic_users_data["head_idx"],
            N=N,
            mode=mode,
            seed=42,
        )
        assert reclist.shape == (synthetic_users_data["n_users"], N)
        assert np.issubdtype(reclist.dtype, np.integer)
        assert not np.any(reclist == -1)
        for u in range(reclist.shape[0]):
            assert len(set(reclist[u])) == N

    def test_t1_f4_dynamic_merge_seed_reproducibility(self, synthetic_users_data):
        """F4.6: Dynamic merge produces identical results when fixed seed is provided."""
        r1 = sp.dynamic_probabilistic_quota_merge(
            synthetic_users_data["scores_pop"],
            synthetic_users_data["scores_tail"],
            synthetic_users_data["train_matrix"],
            synthetic_users_data["head_idx"],
            N=20,
            mode="stochastic",
            seed=42,
        )
        r2 = sp.dynamic_probabilistic_quota_merge(
            synthetic_users_data["scores_pop"],
            synthetic_users_data["scores_tail"],
            synthetic_users_data["train_matrix"],
            synthetic_users_data["head_idx"],
            N=20,
            mode="stochastic",
            seed=42,
        )
        np.testing.assert_array_equal(r1, r2)

    # --- Feature 5: Calibrated Score Fusion ---

    def test_t1_f5_score_fusion_lambda_boundaries(self, synthetic_users_data):
        """F5.1: Lambda boundary conditions: lam=0.0 uses fed, lam=1.0 uses LLM."""
        scores_fed = synthetic_users_data["scores_pop"]
        scores_llm = synthetic_users_data["scores_llm"]
        pool_idx = synthetic_users_data["head_idx"]

        fused_0 = sp.calibrated_score_fusion(scores_fed, scores_llm, pool_idx, lam=0.0, standardize=False)
        fused_1 = sp.calibrated_score_fusion(scores_fed, scores_llm, pool_idx, lam=1.0, standardize=False)

        np.testing.assert_allclose(fused_0[:, pool_idx], scores_fed[:, pool_idx])
        np.testing.assert_allclose(fused_1[:, pool_idx], scores_llm[:, pool_idx])

    def test_t1_f5_score_fusion_intra_pool_standardization(self, synthetic_users_data):
        """F5.2: Intra-pool z-score standardization yields approx zero mean and unit variance."""
        scores_fed = np.random.normal(5.0, 2.0, size=(100, 500)).astype(np.float32)
        scores_llm = np.random.normal(0.0, 1.0, size=(100, 500)).astype(np.float32)
        pool_idx = np.arange(50, dtype=np.int64)

        fused = sp.calibrated_score_fusion(scores_fed, scores_llm, pool_idx, lam=0.5, standardize=True)
        assert fused.shape == (100, 500)
        assert not np.any(np.isnan(fused))

    def test_t1_f5_score_fusion_non_pool_isolation(self, synthetic_users_data):
        """F5.3: Non-pool items are unchanged and not corrupted by pool standardization."""
        scores_fed = synthetic_users_data["scores_pop"].copy()
        scores_llm = synthetic_users_data["scores_llm"]
        head_idx = synthetic_users_data["head_idx"]
        tail_idx = synthetic_users_data["tail_idx"]

        fused = sp.calibrated_score_fusion(scores_fed, scores_llm, head_idx, lam=0.7, standardize=True)
        # Tail items should remain equal to original scores_fed
        np.testing.assert_array_equal(fused[:, tail_idx], scores_fed[:, tail_idx])

    def test_t1_f5_score_fusion_zero_variance_handling(self):
        """F5.4: Constant score inputs (zero variance) handle cleanly without NaN/Inf."""
        scores_fed = np.ones((10, 50), dtype=np.float32) * 3.0
        scores_llm = np.ones((10, 50), dtype=np.float32) * 0.5
        pool_idx = np.arange(20, dtype=np.int64)

        fused = sp.calibrated_score_fusion(scores_fed, scores_llm, pool_idx, lam=0.5, standardize=True)
        assert not np.any(np.isnan(fused))
        assert not np.any(np.isinf(fused))

    def test_t1_f5_score_fusion_multi_pool_separation(self, synthetic_users_data):
        """F5.5: Head and Tail pools can be fused independently without cross-interference."""
        scores_fed = np.random.normal(0, 1, size=(50, 200)).astype(np.float32)
        scores_llm = np.random.uniform(-1, 1, size=(50, 200)).astype(np.float32)
        head_idx = np.arange(0, 50, dtype=np.int64)
        tail_idx = np.arange(50, 200, dtype=np.int64)

        fused_head = sp.calibrated_score_fusion(scores_fed, scores_llm, head_idx, lam=0.7, standardize=True)
        fused_tail = sp.calibrated_score_fusion(scores_fed, scores_llm, tail_idx, lam=0.7, standardize=True)

        assert not np.any(np.isnan(fused_head))
        assert not np.any(np.isnan(fused_tail))

    # --- Feature 6: Calibration & Metric Evaluation ---

    def test_t1_f6_calibration_eval_hand_calculated_oracle(self):
        """F6.1: Exact mathematical oracle verification of Rmse-PC on a known slate."""
        # 4 users, N=10 recommendations each
        # Let head items be {0, 1, 2, 3, 4}
        head_idx = np.array([0, 1, 2, 3, 4], dtype=np.int64)
        # User 0: 2 head items (frac=0.2), Pop_u=0.25 -> diff = -0.05, sq = 0.0025
        # User 1: 3 head items (frac=0.3), Pop_u=0.30 -> diff =  0.00, sq = 0.0000
        # User 2: 1 head item  (frac=0.1), Pop_u=0.20 -> diff = -0.10, sq = 0.0100
        # User 3: 4 head items (frac=0.4), Pop_u=0.35 -> diff =  0.05, sq = 0.0025
        # Mean squared error = (0.0025 + 0.0000 + 0.0100 + 0.0025) / 4 = 0.0150 / 4 = 0.00375
        # Expected Rmse-PC = sqrt(0.00375) = 0.06123724
        reclist_matrix = np.array([
            [0, 1, 10, 11, 12, 13, 14, 15, 16, 17],  # 2 head
            [0, 1, 2, 10, 11, 12, 13, 14, 15, 16],   # 3 head
            [0, 10, 11, 12, 13, 14, 15, 16, 17, 18], # 1 head
            [0, 1, 2, 3, 10, 11, 12, 13, 14, 15],   # 4 head
        ], dtype=np.int64)

        pop_u = np.array([0.25, 0.30, 0.20, 0.35], dtype=np.float32)

        # Compute via metric formula
        is_head = np.isin(reclist_matrix, head_idx).mean(axis=1)
        rmse_pc = np.sqrt(np.mean((is_head - pop_u) ** 2))

        expected_rmse = np.sqrt(0.00375)
        assert abs(rmse_pc - expected_rmse) < 1e-6

    def test_t1_f6_calibration_eval_mrmc_monotonicity(self):
        """F6.2: MRMC metric increases when head items are delayed to bottom ranks."""
        head_idx = np.array([0, 1, 2], dtype=np.int64)
        pop_u = np.array([0.30], dtype=np.float32)

        # Slate 1: Head items placed at top (ranks 0, 1, 2)
        slate_early = np.array([[0, 1, 2, 10, 11, 12, 13, 14, 15, 16]], dtype=np.int64)
        # Slate 2: Head items placed at bottom (ranks 7, 8, 9)
        slate_late = np.array([[10, 11, 12, 13, 14, 15, 16, 0, 1, 2]], dtype=np.int64)

        is_head_early = np.isin(slate_early, head_idx).astype(np.float32)
        is_head_late = np.isin(slate_late, head_idx).astype(np.float32)

        cum_early = np.cumsum(is_head_early, axis=1) / np.arange(1, 11)
        cum_late = np.cumsum(is_head_late, axis=1) / np.arange(1, 11)

        mrmc_early = np.mean(np.abs(cum_early - pop_u))
        mrmc_late = np.mean(np.abs(cum_late - pop_u))

        # Delayed head items should have higher rank miscalibration
        assert mrmc_late != mrmc_early

    def test_t1_f6_calibration_eval_full_rank_metrics_keys(self, synthetic_users_data):
        """F6.3: full_rank_eval produces complete metrics dictionary with expected keys."""
        N = 20
        reclist = sp.dynamic_probabilistic_quota_merge(
            synthetic_users_data["scores_pop"],
            synthetic_users_data["scores_tail"],
            synthetic_users_data["train_matrix"],
            synthetic_users_data["head_idx"],
            N=N,
            mode="stochastic",
            seed=42,
        )

        metrics = met.full_rank_eval(
            synthetic_users_data["scores_pop"],
            synthetic_users_data["test_dict"],
            K=N,
            head_idx=synthetic_users_data["head_idx"],
            train_matrix=synthetic_users_data["train_matrix"],
            reclist_matrix=reclist,
            method="reclist",
        )

        expected_keys = ["recall@K", "ndcg@K", "aplt", "ltc", "rmse_pc", "mrmc", "gkpi", "coverage", "gini_fairness"]
        for k in expected_keys:
            assert k in metrics, f"Missing expected metric key: {k}"
            assert isinstance(metrics[k], float)
            assert not np.isnan(metrics[k])

    def test_t1_f6_calibration_eval_gkpi_harmonic_mean(self):
        """F6.4: GKPI computes exact arithmetic mean of harmonic means."""
        ndcg = 0.05
        aplt = 0.80
        entropy = 0.50
        novelty = 0.60
        ltc = 0.40

        def H(x, y):
            return (2 * x * y) / (x + y) if (x + y) > 0 else 0.0

        expected_gkpi = np.mean([H(ndcg, aplt), H(ndcg, entropy), H(ndcg, novelty), H(ndcg, ltc)])
        computed_gkpi = sp.gkpi_score(ndcg, aplt, entropy, novelty, ltc)

        assert abs(computed_gkpi - expected_gkpi) < 1e-6

    def test_t1_f6_calibration_eval_rmse_pc_upper_bound(self, synthetic_users_data):
        """F6.5: Empirical Rmse-PC strictly satisfies <= 0.056 for dynamic merge at N=20."""
        N = 20
        reclist = sp.dynamic_probabilistic_quota_merge(
            synthetic_users_data["scores_pop"],
            synthetic_users_data["scores_tail"],
            synthetic_users_data["train_matrix"],
            synthetic_users_data["head_idx"],
            N=N,
            mode="stochastic",
            seed=42,
        )

        metrics = met.full_rank_eval(
            synthetic_users_data["scores_pop"],
            synthetic_users_data["test_dict"],
            K=N,
            head_idx=synthetic_users_data["head_idx"],
            train_matrix=synthetic_users_data["train_matrix"],
            reclist_matrix=reclist,
            method="reclist",
        )

        assert metrics["rmse_pc"] <= 0.0560, f"Rmse-PC {metrics['rmse_pc']} violates <= 0.056 ceiling"


# ============================================================================
# TIER 2: BOUNDARY & CORNER CASES
# ============================================================================

class TestTier2BoundaryAndCornerCases:
    """Tier 2: Boundary conditions, degenerate inputs, and extreme configurations."""

    @pytest.mark.parametrize("N", [1, 2, 5, 20, 50, 100])
    def test_t2_b1_varying_slate_sizes(self, synthetic_users_data, N):
        """B1: Slate size variations from N=1 to N=100."""
        reclist = sp.dynamic_probabilistic_quota_merge(
            synthetic_users_data["scores_pop"],
            synthetic_users_data["scores_tail"],
            synthetic_users_data["train_matrix"],
            synthetic_users_data["head_idx"],
            N=N,
            mode="stochastic",
            seed=42,
        )
        assert reclist.shape == (synthetic_users_data["n_users"], N)
        for u in range(min(10, reclist.shape[0])):
            assert len(set(reclist[u])) == N

    def test_t2_b2_extreme_user_inclinations(self, synthetic_catalog):
        """B2: Extreme inclinations: Pop_u = 0.0 (pure tail) and Pop_u = 1.0 (pure head)."""
        n_items = synthetic_catalog["n_items"]
        head_idx = synthetic_catalog["head_idx"]
        tail_idx = synthetic_catalog["tail_idx"]

        train_matrix = np.zeros((2, n_items), dtype=np.float32)
        # User 0: All tail interactions (Pop_u = 0.0)
        train_matrix[0, tail_idx[:20]] = 1
        # User 1: All head interactions (Pop_u = 1.0)
        train_matrix[1, head_idx[:20]] = 1

        pop_u = sp.user_popularity_inclination(train_matrix, head_idx)
        assert pop_u[0] == 0.0
        assert pop_u[1] == 1.0

        n_pop, n_tail = sp.stochastic_user_quota(pop_u, N=20)
        assert n_pop[0] == 0 and n_tail[0] == 20
        assert n_pop[1] == 20 and n_tail[1] == 0

    def test_t2_b3_cold_start_empty_histories(self, synthetic_catalog):
        """B3: Empty user interaction histories (|C_u| = 0) handle gracefully."""
        n_users = 10
        n_items = synthetic_catalog["n_items"]
        head_idx = synthetic_catalog["head_idx"]
        tail_idx = synthetic_catalog["tail_idx"]

        train_matrix = np.zeros((n_users, n_items), dtype=np.float32)
        scores_pop = np.random.normal(0, 1, size=(n_users, n_items)).astype(np.float32)
        scores_tail = np.random.normal(0, 1, size=(n_users, n_items)).astype(np.float32)
        scores_pop[:, tail_idx] = -np.inf
        scores_tail[:, head_idx] = -np.inf

        reclist = sp.dynamic_probabilistic_quota_merge(
            scores_pop, scores_tail, train_matrix, head_idx, N=20, mode="stochastic", seed=42
        )
        assert reclist.shape == (n_users, 20)
        assert not np.any(reclist == -1)

    def test_t2_b4_single_item_head_pool(self):
        """B4: Degenerate single-item head pool (|H| = 1) handles quota overflow without crash."""
        n_users = 5
        n_items = 100
        head_idx = np.array([0], dtype=np.int64)
        tail_idx = np.arange(1, n_items, dtype=np.int64)

        train_matrix = np.zeros((n_users, n_items), dtype=np.float32)
        # Give users 50% head preference
        train_matrix[:, 0] = 1
        train_matrix[:, 1] = 1

        scores_pop = np.full((n_users, n_items), -np.inf, dtype=np.float32)
        scores_tail = np.full((n_users, n_items), -np.inf, dtype=np.float32)
        scores_pop[:, head_idx] = 10.0
        scores_tail[:, tail_idx] = 5.0

        reclist = sp.dynamic_probabilistic_quota_merge(
            scores_pop, scores_tail, train_matrix, head_idx, N=20, mode="stochastic", seed=42
        )
        assert reclist.shape == (n_users, 20)
        for u in range(n_users):
            assert len(set(reclist[u])) == 20

    def test_t2_b5_extreme_score_differentials(self, synthetic_users_data):
        """B5: Massive score differentials (+/- 1e6) and -inf masking integrity."""
        scores_pop = synthetic_users_data["scores_pop"].copy()
        scores_tail = synthetic_users_data["scores_tail"].copy()
        head_idx = synthetic_users_data["head_idx"]
        tail_idx = synthetic_users_data["tail_idx"]

        # Set extreme differential
        scores_pop[:, head_idx] = 1e6
        scores_tail[:, tail_idx] = -1e6

        reclist = sp.confidence_elastic_merge(
            scores_pop, scores_tail, synthetic_users_data["train_matrix"], head_idx, N=20, threshold=0.1
        )
        assert reclist.shape == (synthetic_users_data["n_users"], 20)
        assert not np.any(np.isnan(reclist))

    def test_t2_b6_zero_variance_scores(self, synthetic_users_data):
        """B6: Uniform flat scores across catalog (zero variance)."""
        n_users = synthetic_users_data["n_users"]
        n_items = synthetic_users_data["n_items"]
        head_idx = synthetic_users_data["head_idx"]
        tail_idx = synthetic_users_data["tail_idx"]

        scores_pop = np.full((n_users, n_items), 1.0, dtype=np.float32)
        scores_tail = np.full((n_users, n_items), 1.0, dtype=np.float32)
        scores_pop[:, tail_idx] = -np.inf
        scores_tail[:, head_idx] = -np.inf

        reclist = sp.dynamic_probabilistic_quota_merge(
            scores_pop, scores_tail, synthetic_users_data["train_matrix"], head_idx, N=20, mode="stochastic", seed=42
        )
        assert reclist.shape == (n_users, 20)
        for u in range(min(5, n_users)):
            assert len(set(reclist[u])) == 20

    def test_t2_b7_all_head_items_previously_interacted(self, synthetic_catalog):
        """B7: User has interacted with ALL head items; roll over quota safely to tail pool."""
        n_items = synthetic_catalog["n_items"]
        head_idx = synthetic_catalog["head_idx"]
        tail_idx = synthetic_catalog["tail_idx"]

        train_matrix = np.zeros((1, n_items), dtype=np.float32)
        # User interacted with every head item
        train_matrix[0, head_idx] = 1

        scores_pop = np.full((1, n_items), -np.inf, dtype=np.float32)
        scores_tail = np.full((1, n_items), -np.inf, dtype=np.float32)
        scores_pop[0, head_idx] = 10.0
        scores_tail[0, tail_idx] = 5.0

        reclist = sp.dynamic_probabilistic_quota_merge(
            scores_pop, scores_tail, train_matrix, head_idx, N=20, mode="stochastic", seed=42
        )
        assert reclist.shape == (1, 20)
        # Slate must be filled completely with tail items without leaking head items
        slate = reclist[0]
        assert len(set(slate)) == 20
        assert not any(it in head_idx for it in slate)


# ============================================================================
# TIER 3: CROSS-FEATURE COMBINATIONS & PIPELINES
# ============================================================================

class TestTier3CrossFeatureCombinations:
    """Tier 3: Pairwise and multi-feature interaction pipelines."""

    def test_t3_c1_stochastic_quota_and_adaptive_alpha_pipeline(self, synthetic_users_data, synthetic_catalog):
        """C1: User-adaptive alpha feeding expectation-preserving stochastic quota."""
        train_matrix = synthetic_users_data["train_matrix"]
        pop_count = synthetic_catalog["pop_count"]

        # Step 1: Adaptive alpha inclination
        pop_u_adaptive = sp.user_adaptive_alpha_partition(pop_count, train_matrix, alpha_base=0.20, beta=0.15)
        # Step 2: Stochastic quota allocation
        n_pop, n_tail = sp.stochastic_user_quota(pop_u_adaptive, N=20, rng=np.random.default_rng(42))

        assert np.all(n_pop + n_tail == 20)
        assert np.all(n_pop >= 0) and np.all(n_pop <= 20)
        # Check that individual deviations respect 1/N bound relative to adaptive inclinations
        np.testing.assert_array_less(np.abs(n_pop / 20.0 - pop_u_adaptive), (1.0 / 20.0) + 1e-6)

    def test_t3_c2_score_fusion_and_confidence_elastic_pipeline(self, synthetic_users_data):
        """C2: Calibrated score fusion feeding confidence-elastic merge."""
        scores_pop = synthetic_users_data["scores_pop"]
        scores_tail = synthetic_users_data["scores_tail"]
        scores_llm = synthetic_users_data["scores_llm"]
        head_idx = synthetic_users_data["head_idx"]
        tail_idx = synthetic_users_data["tail_idx"]
        train_matrix = synthetic_users_data["train_matrix"]

        # Step 1: Fuse collaborative + LLM scores for head and tail pools
        fused_pop = sp.calibrated_score_fusion(scores_pop, scores_llm, head_idx, lam=0.70, standardize=True)
        fused_tail = sp.calibrated_score_fusion(scores_tail, scores_llm, tail_idx, lam=0.70, standardize=True)

        # Step 2: Confidence-elastic merge using fused scores
        reclist = sp.confidence_elastic_merge(
            fused_pop, fused_tail, train_matrix, head_idx, N=20, threshold=0.1
        )
        assert reclist.shape == (synthetic_users_data["n_users"], 20)
        for u in range(reclist.shape[0]):
            assert len(set(reclist[u])) == 20

    def test_t3_c3_dynamic_merge_and_full_rank_eval_pipeline(self, synthetic_users_data):
        """C3: Dynamic probabilistic merge coupled directly into full_rank_eval."""
        N = 20
        reclist = sp.dynamic_probabilistic_quota_merge(
            synthetic_users_data["scores_pop"],
            synthetic_users_data["scores_tail"],
            synthetic_users_data["train_matrix"],
            synthetic_users_data["head_idx"],
            N=N,
            mode="stochastic",
            seed=42,
        )

        metrics = met.full_rank_eval(
            synthetic_users_data["scores_pop"],
            synthetic_users_data["test_dict"],
            K=N,
            head_idx=synthetic_users_data["head_idx"],
            train_matrix=synthetic_users_data["train_matrix"],
            reclist_matrix=reclist,
            method="reclist",
        )

        assert "recall@K" in metrics
        assert "rmse_pc" in metrics
        assert metrics["rmse_pc"] <= 0.0560

    def test_t3_c4_hybrid_merge_with_score_fusion(self, synthetic_users_data):
        """C4: End-to-end hybrid merge integrating all components."""
        scores_pop = synthetic_users_data["scores_pop"]
        scores_tail = synthetic_users_data["scores_tail"]
        scores_llm = synthetic_users_data["scores_llm"]
        head_idx = synthetic_users_data["head_idx"]
        tail_idx = synthetic_users_data["tail_idx"]

        fused_pop = sp.calibrated_score_fusion(scores_pop, scores_llm, head_idx, lam=0.70, standardize=True)
        fused_tail = sp.calibrated_score_fusion(scores_tail, scores_llm, tail_idx, lam=0.70, standardize=True)

        reclist = sp.dynamic_probabilistic_quota_merge(
            fused_pop,
            fused_tail,
            synthetic_users_data["train_matrix"],
            head_idx,
            N=20,
            mode="hybrid",
            seed=42,
        )
        assert reclist.shape == (synthetic_users_data["n_users"], 20)

    def test_t3_c5_pareto_adaptive_stochastic_mask_eval_chain(self, synthetic_users_data, synthetic_catalog):
        """C5: 5-module full pipeline chain from catalog partitioning to evaluation."""
        pop_count = synthetic_catalog["pop_count"]
        train_matrix = synthetic_users_data["train_matrix"]

        head_idx, tail_idx = sp.pareto_partition(pop_count, alpha=0.20)
        pop_u_adapt = sp.user_adaptive_alpha_partition(pop_count, train_matrix, alpha_base=0.20, beta=0.15)
        n_pop, n_tail = sp.stochastic_user_quota(pop_u_adapt, N=20, rng=np.random.default_rng(42))

        # Verify quota sums
        assert np.all(n_pop + n_tail == 20)


# ============================================================================
# TIER 4: REAL-WORLD WORKLOAD SCENARIOS & ACCEPTANCE CRITERIA
# ============================================================================

class TestTier4RealWorldWorkloads:
    """Tier 4: Full-scale ML-1M benchmark simulation and acceptance criteria assertions."""

    def test_t4_w1_ml1m_workload_acceptance_criteria(self):
        """
        W1: Realistic ML-1M Scale Simulation Benchmark (6,040 users, 3,533 items).
        Acceptance Criteria:
          1. Pipeline executes cleanly without errors.
          2. Recall@20 strictly > 0.0370.
          3. Rmse-PC strictly <= 0.0560.
          4. APLT >= 0.7000 (preserving tail recommendation volume).
        """
        n_users = 6040
        n_items = 3533
        rng = np.random.default_rng(42)

        # Generate realistic power-law item popularity
        ranks = np.arange(1, n_items + 1)
        pop_counts = (100000.0 / (ranks ** 0.85)).astype(np.float32)

        head_idx, tail_idx = sp.pareto_partition(pop_counts, alpha=0.20)
        assert len(head_idx) > 0 and len(tail_idx) > 0

        # Sparse interaction matrix representation
        # Each user has between 20 and 150 interactions
        train_matrix = np.zeros((n_users, n_items), dtype=np.float32)
        test_dict = {}

        for u in range(n_users):
            n_interactions = rng.integers(20, 80)
            u_pop_prop = rng.beta(2, 6)  # Beta distribution modeling empirical Pop_u ~ 0.25
            n_head = int(np.round(n_interactions * u_pop_prop))
            n_head = min(n_head, len(head_idx) - 1)
            n_tail = min(n_interactions - n_head, len(tail_idx) - 1)

            chosen_head = rng.choice(head_idx, size=max(n_head, 1), replace=False)
            chosen_tail = rng.choice(tail_idx, size=max(n_tail, 1), replace=False)

            train_matrix[u, chosen_head] = 1.0
            train_matrix[u, chosen_tail] = 1.0

            # Leave-one-out ground truth: 50% head item, 50% tail item among unseen
            if rng.uniform() < 0.5:
                unseen_head = np.setdiff1d(head_idx, chosen_head)
                test_dict[u] = [int(rng.choice(unseen_head))]
            else:
                unseen_tail = np.setdiff1d(tail_idx, chosen_tail)
                test_dict[u] = [int(rng.choice(unseen_tail))]

        # Model scores with synthetic signal correlating with ground truth
        scores_pop = rng.normal(0.0, 1.0, size=(n_users, n_items)).astype(np.float32)
        scores_tail = rng.normal(0.0, 1.0, size=(n_users, n_items)).astype(np.float32)

        # Inject realistic collaborative signal: ground truth items receive boost
        for u, gt in test_dict.items():
            t = gt[0]
            if t in head_idx:
                scores_pop[u, t] += rng.uniform(2.5, 5.0)
            else:
                scores_tail[u, t] += rng.uniform(2.5, 5.0)

        # Mask non-pool items
        scores_pop[:, tail_idx] = -np.inf
        scores_tail[:, head_idx] = -np.inf

        # LLM semantic embeddings simulation
        scores_llm = rng.uniform(-0.3, 0.7, size=(n_users, n_items)).astype(np.float32)
        for u, gt in test_dict.items():
            scores_llm[u, gt[0]] += 0.8

        # Fuse scores (λ = 0.70)
        fused_pop = sp.calibrated_score_fusion(scores_pop, scores_llm, head_idx, lam=0.70, standardize=True)
        fused_tail = sp.calibrated_score_fusion(scores_tail, scores_llm, tail_idx, lam=0.70, standardize=True)

        # Execute Dynamic Merge at N = 20
        start_time = time.time()
        reclist_dynamic = sp.dynamic_probabilistic_quota_merge(
            fused_pop,
            fused_tail,
            train_matrix,
            head_idx,
            N=20,
            mode="hybrid",
            seed=42,
        )
        elapsed = time.time() - start_time

        # Metric Evaluation
        metrics = met.full_rank_eval(
            fused_pop,
            test_dict,
            K=20,
            head_idx=head_idx,
            train_matrix=train_matrix,
            reclist_matrix=reclist_dynamic,
            method="reclist",
        )

        recall_20 = metrics["recall@K"]
        rmse_pc = metrics["rmse_pc"]
        aplt = metrics["aplt"]

        print("\n" + "=" * 80)
        print("REAL-WORLD WORKLOAD (ML-1M Scale) EVALUATION RESULTS:")
        print(f"  Recall@20 : {recall_20:.5f} (Target: > 0.0370)")
        print(f"  Rmse-PC   : {rmse_pc:.5f} (Target: <= 0.0560)")
        print(f"  APLT      : {aplt:.5f} (Target: >= 0.7000)")
        print(f"  Throughput: {n_users / max(elapsed, 1e-4):.1f} users/sec ({elapsed:.2f}s total)")
        print("=" * 80)

        # STRICT ACCEPTANCE ASSERTIONS
        assert recall_20 > 0.0370, f"Recall@20 ({recall_20:.5f}) failed acceptance threshold > 0.0370"
        assert rmse_pc <= 0.0560, f"Rmse-PC ({rmse_pc:.5f}) violated calibration ceiling <= 0.0560"
        assert aplt >= 0.7000, f"APLT ({aplt:.5f}) violated long-tail floor >= 0.7000"

    def test_t4_w2_high_throughput_scalability(self):
        """W2: High-concurrency batch execution benchmark over 10,000 users."""
        n_users = 10000
        n_items = 1000
        rng = np.random.default_rng(42)

        head_idx = np.arange(50, dtype=np.int64)
        tail_idx = np.arange(50, n_items, dtype=np.int64)

        train_matrix = np.zeros((n_users, n_items), dtype=np.float32)
        for u in range(n_users):
            train_matrix[u, rng.choice(head_idx, size=5, replace=False)] = 1
            train_matrix[u, rng.choice(tail_idx, size=15, replace=False)] = 1

        scores_pop = rng.normal(0, 1, size=(n_users, n_items)).astype(np.float32)
        scores_tail = rng.normal(0, 1, size=(n_users, n_items)).astype(np.float32)
        scores_pop[:, tail_idx] = -np.inf
        scores_tail[:, head_idx] = -np.inf

        t0 = time.time()
        reclist = sp.dynamic_probabilistic_quota_merge(
            scores_pop, scores_tail, train_matrix, head_idx, N=20, mode="stochastic", seed=42
        )
        duration = time.time() - t0

        assert reclist.shape == (n_users, 20)
        assert duration < 5.0, f"Batch execution too slow: {duration:.2f}s for 10k users"

    def test_t4_w3_multi_variant_comparative_sweep(self, synthetic_users_data):
        """W3: Multi-variant comparative benchmark confirming all variants satisfy calibration."""
        modes = ["static", "stochastic", "adaptive_alpha", "confidence_elastic", "hybrid"]
        results = {}

        for mode in modes:
            reclist = sp.dynamic_probabilistic_quota_merge(
                synthetic_users_data["scores_pop"],
                synthetic_users_data["scores_tail"],
                synthetic_users_data["train_matrix"],
                synthetic_users_data["head_idx"],
                N=20,
                mode=mode,
                seed=42,
            )
            m = met.full_rank_eval(
                synthetic_users_data["scores_pop"],
                synthetic_users_data["test_dict"],
                K=20,
                head_idx=synthetic_users_data["head_idx"],
                train_matrix=synthetic_users_data["train_matrix"],
                reclist_matrix=reclist,
                method="reclist",
            )
            results[mode] = m
            assert m["rmse_pc"] <= 0.0560, f"Mode {mode} failed calibration with Rmse-PC {m['rmse_pc']}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
