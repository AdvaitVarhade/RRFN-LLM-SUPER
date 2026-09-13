"""Comprehensive Unit Tests for FedSUPER-LLM Dynamic Merge Functions.

Tests cover:
  1. Quota validity: (n_pop + n_tail == N, n_pop >= 0, n_tail >= 0, integer types).
  2. Expectation preservation: (mean error across 10,000 users < 1e-3).
  3. Calibration bounds: (individual deviation <= 1/N, empirical Rmse-PC <= 0.035 for N=20).
  4. User adaptive alpha partition: (inclination range, beta scaling, cold-start handling).
  5. Calibrated score fusion: (intra-pool standardization, lambda blending, masking).
  6. Confidence elastic merge: (margin-aware slot adjustment, valid slate construction).
  7. Dynamic probabilistic quota merge: (all modes: stochastic, adaptive_alpha, confidence_elastic, hybrid, static).
  8. Reclist construction validity: (shape (n_users, N), no duplicates, no train-positive leakage).
  9. Evaluate calibration & regression tests on existing SUPER functions.
"""
import os
import sys
import unittest
import numpy as np

# Ensure src is on python path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from super import (
    pareto_partition,
    tri_tier_partition,
    user_popularity_inclination,
    tri_tier_user_inclinations,
    stochastic_user_quota,
    user_adaptive_alpha_partition,
    calibrated_score_fusion,
    confidence_elastic_merge,
    dynamic_probabilistic_quota_merge,
    evaluate_calibration,
    super_blueprint_merge,
    tri_tier_blueprint_merge,
    logit_adjusted_calibration,
    mask_user_trainpos,
    gkpi_score,
)


class TestStochasticUserQuota(unittest.TestCase):
    """Test suite for stochastic_user_quota randomized rounding."""

    def test_quota_validity_and_types(self):
        """Verify n_pop + n_tail == N, non-negativity, and integer dtype."""
        rng = np.random.default_rng(42)
        n_users = 500
        pop_u = rng.uniform(0.0, 1.0, size=n_users)

        for N in [5, 10, 20, 50]:
            n_pop, n_tail = stochastic_user_quota(pop_u, N=N, rng=rng)

            self.assertEqual(n_pop.shape, (n_users,))
            self.assertEqual(n_tail.shape, (n_users,))
            self.assertTrue(np.issubdtype(n_pop.dtype, np.integer))
            self.assertTrue(np.issubdtype(n_tail.dtype, np.integer))

            # Non-negativity & ceiling
            self.assertTrue(np.all(n_pop >= 0))
            self.assertTrue(np.all(n_pop <= N))
            self.assertTrue(np.all(n_tail >= 0))
            self.assertTrue(np.all(n_tail <= N))

            # Exact sum to N
            self.assertTrue(np.all(n_pop + n_tail == N))

    def test_boundary_values(self):
        """Test extreme boundary inclinations 0.0 and 1.0."""
        pop_u = np.array([0.0, 1.0, 0.0, 1.0, 0.5])
        n_pop, n_tail = stochastic_user_quota(pop_u, N=20, rng=42)

        self.assertEqual(n_pop[0], 0)
        self.assertEqual(n_tail[0], 20)
        self.assertEqual(n_pop[1], 20)
        self.assertEqual(n_tail[1], 0)
        self.assertEqual(n_pop[2], 0)
        self.assertEqual(n_tail[2], 20)
        self.assertEqual(n_pop[3], 20)
        self.assertEqual(n_tail[3], 0)
        self.assertEqual(n_pop[4], 10)
        self.assertEqual(n_tail[4], 10)

    def test_expectation_preservation_large_sample(self):
        """Test that sample expectation E[n_pop / N] == pop_u (mean error < 1e-3 across 10,000 users)."""
        rng = np.random.default_rng(12345)
        n_users = 10000
        pop_u = rng.uniform(0.0, 1.0, size=n_users)
        N = 20

        n_pop, n_tail = stochastic_user_quota(pop_u, N=N, rng=rng)
        allocated_ratio = n_pop / float(N)
        mean_error = float(np.mean(allocated_ratio - pop_u))

        self.assertLess(abs(mean_error), 1e-3, f"Mean error {mean_error} exceeds 1e-3")

    def test_individual_calibration_bounds(self):
        """Verify strict individual deviation bound |n_pop / N - pop_u| <= 1 / N."""
        rng = np.random.default_rng(999)
        n_users = 5000
        pop_u = rng.uniform(0.0, 1.0, size=n_users)

        for N in [10, 20]:
            n_pop, _ = stochastic_user_quota(pop_u, N=N, rng=rng)
            deviations = np.abs(n_pop / float(N) - pop_u)
            max_dev = np.max(deviations)
            bound = 1.0 / float(N) + 1e-7

            self.assertLessEqual(max_dev, bound, f"Max deviation {max_dev} exceeds bound {bound}")

    def test_empirical_rmse_pc_bound(self):
        """Empirical Rmse-PC for N=20 must be strictly <= 0.035 (theoretical ~0.025)."""
        rng = np.random.default_rng(2026)
        n_users = 6040
        # Realistic beta distribution matching MovieLens-1M Pop_u
        pop_u = rng.beta(a=2.0, b=6.0, size=n_users)
        N = 20

        n_pop, _ = stochastic_user_quota(pop_u, N=N, rng=rng)
        rmse_pc = float(np.sqrt(np.mean((n_pop / float(N) - pop_u) ** 2)))

        self.assertLessEqual(rmse_pc, 0.035, f"Empirical Rmse-PC {rmse_pc} exceeds 0.035")


class TestUserAdaptiveAlphaPartition(unittest.TestCase):
    """Test suite for user_adaptive_alpha_partition."""

    def setUp(self):
        self.n_users = 100
        self.n_items = 50
        rng = np.random.default_rng(42)
        # Power-law item popularity
        ranks = np.arange(1, self.n_items + 1)
        self.pop_count = (1000.0 / ranks).astype(np.float64)

        # Sparse interaction matrix
        self.train_matrix = np.zeros((self.n_users, self.n_items), dtype=np.int32)
        for u in range(self.n_users):
            # Varying user activity
            num_inter = rng.integers(1, 30)
            chosen_items = rng.choice(self.n_items, size=num_inter, replace=False)
            self.train_matrix[u, chosen_items] = 1

    def test_output_shape_and_bounds(self):
        """Output must have shape (n_users,) and values in [0, 1]."""
        pop_u_adapt = user_adaptive_alpha_partition(self.pop_count, self.train_matrix, alpha_base=0.20, beta=0.15)

        self.assertEqual(pop_u_adapt.shape, (self.n_users,))
        self.assertTrue(np.all(pop_u_adapt >= 0.0))
        self.assertTrue(np.all(pop_u_adapt <= 1.0))

    def test_beta_zero_matches_standard_inclination(self):
        """When beta=0, adaptive partition must match standard user_popularity_inclination."""
        head_idx, _ = pareto_partition(self.pop_count, alpha=0.20)
        pop_u_std = user_popularity_inclination(self.train_matrix, head_idx)
        pop_u_adapt = user_adaptive_alpha_partition(self.pop_count, self.train_matrix, alpha_base=0.20, beta=0.0)

        np.testing.assert_allclose(pop_u_adapt, pop_u_std, atol=1e-5)

    def test_cold_start_user_handling(self):
        """Users with 0 interactions should receive alpha_base without crashing."""
        train_matrix_cold = self.train_matrix.copy()
        train_matrix_cold[0, :] = 0  # Cold start user 0

        pop_u_adapt = user_adaptive_alpha_partition(self.pop_count, train_matrix_cold, alpha_base=0.25, beta=0.15)
        self.assertEqual(pop_u_adapt[0], 0.25)


class TestCalibratedScoreFusion(unittest.TestCase):
    """Test suite for calibrated_score_fusion."""

    def setUp(self):
        self.n_users = 20
        self.n_items = 40
        rng = np.random.default_rng(42)
        self.scores_fed = rng.normal(0.0, 1.0, size=(self.n_users, self.n_items)).astype(np.float32)
        self.scores_llm = rng.uniform(-1.0, 1.0, size=(self.n_users, self.n_items)).astype(np.float32)
        self.head_idx = np.array([0, 1, 2, 3, 4], dtype=np.int64)
        self.tail_idx = np.array(list(set(range(self.n_items)) - set(self.head_idx)), dtype=np.int64)

    def test_masking_and_shape(self):
        """Fused matrix must have exact shape and -inf on non-pool items."""
        fused_head = calibrated_score_fusion(self.scores_fed, self.scores_llm, self.head_idx, lam=0.70, standardize=True)

        self.assertEqual(fused_head.shape, (self.n_users, self.n_items))
        self.assertTrue(np.all(fused_head[:, self.head_idx] > -np.inf))
        self.assertTrue(np.all(fused_head[:, self.tail_idx] == -np.inf))

    def test_lambda_extremes(self):
        """Test lambda=1.0 (pure LLM) and lambda=0.0 (pure standardized Fed)."""
        fused_pure_llm = calibrated_score_fusion(self.scores_fed, self.scores_llm, self.head_idx, lam=1.0, standardize=True)
        np.testing.assert_allclose(fused_pure_llm[:, self.head_idx], self.scores_llm[:, self.head_idx], atol=1e-6)

        fused_pure_fed = calibrated_score_fusion(self.scores_fed, self.scores_llm, self.head_idx, lam=0.0, standardize=False)
        np.testing.assert_allclose(fused_pure_fed[:, self.head_idx], self.scores_fed[:, self.head_idx], atol=1e-6)

    def test_standardization_zero_variance(self):
        """Intra-pool standardization handles constant scores gracefully."""
        const_scores = np.ones((self.n_users, self.n_items), dtype=np.float32)
        fused = calibrated_score_fusion(const_scores, self.scores_llm, self.head_idx, lam=0.50, standardize=True)
        self.assertFalse(np.isnan(fused[:, self.head_idx]).any())


class TestDynamicProbabilisticMerge(unittest.TestCase):
    """Test suite for dynamic merge algorithms (all modes, reclist validity, leakage)."""

    def setUp(self):
        self.n_users = 50
        self.n_items = 100
        self.N = 20
        rng = np.random.default_rng(42)

        # Build mock interaction matrix and Pareto split
        ranks = np.arange(1, self.n_items + 1)
        self.pop_count = (5000.0 / ranks).astype(np.float64)
        self.head_idx, self.tail_idx = pareto_partition(self.pop_count, alpha=0.20)

        self.train_matrix = np.zeros((self.n_users, self.n_items), dtype=np.int32)
        for u in range(self.n_users):
            num_inter = rng.integers(5, 25)
            chosen = rng.choice(self.n_items, size=num_inter, replace=False)
            self.train_matrix[u, chosen] = 1

        self.scores_pop = rng.normal(0.5, 1.0, size=(self.n_users, self.n_items)).astype(np.float32)
        self.scores_tail = rng.normal(0.0, 1.0, size=(self.n_users, self.n_items)).astype(np.float32)

    def _verify_reclist(self, reclist, mode_name):
        """Helper to verify shape, unique items, no train-positive leakage, valid item IDs."""
        self.assertEqual(reclist.shape, (self.n_users, self.N), f"Mode {mode_name}: incorrect shape")
        self.assertTrue(np.issubdtype(reclist.dtype, np.integer), f"Mode {mode_name}: not integer dtype")

        for u in range(self.n_users):
            rec = reclist[u]
            # 1. No duplicates
            self.assertEqual(len(set(rec)), self.N, f"Mode {mode_name}: user {u} contains duplicates: {rec}")
            # 2. Valid range
            self.assertTrue(np.all(rec >= 0) and np.all(rec < self.n_items), f"Mode {mode_name}: invalid item indices")
            # 3. No train-positive leakage
            train_pos = set(np.where(self.train_matrix[u] > 0)[0].tolist())
            leakage = set(rec) & train_pos
            self.assertEqual(len(leakage), 0, f"Mode {mode_name}: user {u} has {len(leakage)} leaked train positives")

    def test_all_merge_modes(self):
        """Test reclist validity across all 5 merge modes."""
        modes = ["stochastic", "adaptive_alpha", "confidence_elastic", "hybrid", "static"]
        for mode in modes:
            reclist = dynamic_probabilistic_quota_merge(
                self.scores_pop,
                self.scores_tail,
                self.train_matrix,
                self.head_idx,
                N=self.N,
                mode=mode,
                seed=42,
                alpha_base=0.20,
                beta=0.15,
                threshold=0.10,
                pop_count=self.pop_count,
            )
            self._verify_reclist(reclist, mode)

    def test_confidence_elastic_merge_standalone(self):
        """Test confidence_elastic_merge function directly."""
        reclist = confidence_elastic_merge(
            self.scores_pop,
            self.scores_tail,
            self.train_matrix,
            self.head_idx,
            N=self.N,
            threshold=0.10,
        )
        self._verify_reclist(reclist, "confidence_elastic_standalone")

    def test_calibration_evaluation(self):
        """Test evaluate_calibration on generated dynamic merge recommendations."""
        reclist = dynamic_probabilistic_quota_merge(
            self.scores_pop,
            self.scores_tail,
            self.train_matrix,
            self.head_idx,
            N=self.N,
            mode="stochastic",
            seed=42,
        )
        cal_metrics = evaluate_calibration(reclist, self.train_matrix, self.head_idx)

        self.assertIn("rmse_pc", cal_metrics)
        self.assertIn("mrmc", cal_metrics)
        self.assertGreaterEqual(cal_metrics["rmse_pc"], 0.0)
        self.assertLessEqual(cal_metrics["rmse_pc"], 0.056)
        self.assertGreaterEqual(cal_metrics["mrmc"], 0.0)


class TestExistingSuperRegression(unittest.TestCase):
    """Regression tests for existing super.py functions to prevent regressions."""

    def test_pareto_partition(self):
        pop = np.array([100, 50, 25, 10, 5, 1, 0])
        head_idx, tail_idx = pareto_partition(pop, alpha=0.50)
        self.assertIn(0, head_idx)
        self.assertEqual(len(head_idx) + len(tail_idx), len(pop))
        self.assertIn(6, tail_idx)  # zero pop item in tail

    def test_tri_tier_partition(self):
        pop = np.array([100, 50, 25, 15, 10, 5, 2, 0])
        head_idx, torso_idx, tail_idx = tri_tier_partition(pop, alpha_head=0.30, alpha_torso=0.40)
        self.assertEqual(len(head_idx) + len(torso_idx) + len(tail_idx), len(pop))

    def test_gkpi_score(self):
        score = gkpi_score(ndcg=0.03, aplt=0.78, entropy=8.5, novelty=10.2, ltc=0.04)
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_logit_adjusted_calibration(self):
        scores = np.array([[1.0, 2.0], [3.0, 4.0]])
        pop_counts = np.array([100, 10])
        pop_u = np.array([0.8, 0.2])
        adj = logit_adjusted_calibration(scores, pop_counts, pop_u, tau=0.5)
        self.assertEqual(adj.shape, scores.shape)


if __name__ == "__main__":
    unittest.main(verbosity=2)
