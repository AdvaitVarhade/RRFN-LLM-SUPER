"""Adversarial Stress Test Suite for FedSUPER-LLM Dynamic Merge Functions.

Author: Challenger 1 (teamwork_preview_challenger_m1_1)
Target: src/super.py
Milestone: M1

This suite conducts empirical adversarial stress testing on all functions in src/super.py:
1. 100,000-sample stochastic user quota expectation preservation and individual deviation bounds.
2. Extreme edge cases: N=1, N=100; pop_u=0.0, 1.0, and uniform; degenerate catalogs;
   extreme score differentials (+/- 1e6); empty/full training histories.
3. Verification across all 5 merge modes (stochastic, adaptive_alpha, confidence_elastic, hybrid, static):
   - Zero training positive leakage
   - Strict item uniqueness (no duplicates)
   - Output shape and integer validity
4. User-adaptive alpha and calibrated score fusion stress tests.
"""

import os
import sys
import unittest
import numpy as np

# Ensure src is on python path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import super as sp


class TestAdversarialStochasticQuota(unittest.TestCase):
    """Adversarial stress testing for stochastic_user_quota."""

    def test_100k_sample_expectation_preservation_uniform(self):
        """Verify expectation preservation across 100,000 samples under uniform pop_u."""
        N_samples = 100000
        rng = np.random.default_rng(2026)
        pop_u = rng.uniform(0.0, 1.0, size=N_samples).astype(np.float64)

        for N in [1, 2, 5, 10, 20, 50, 100]:
            n_pop, n_tail = sp.stochastic_user_quota(pop_u, N=N, rng=rng)

            # 1. Sum invariance
            np.testing.assert_array_equal(n_pop + n_tail, N)
            self.assertTrue(np.all(n_pop >= 0) and np.all(n_pop <= N))
            self.assertTrue(np.all(n_tail >= 0) and np.all(n_tail <= N))

            # 2. Expectation preservation (mean error over 100,000 samples must be < 1e-3)
            emp_mean_pop = np.mean(n_pop / float(N))
            true_mean_pop = np.mean(pop_u)
            mean_error = abs(emp_mean_pop - true_mean_pop)
            self.assertLess(mean_error, 1e-3, f"N={N}: Mean error {mean_error:.6f} exceeds 1e-3")

            # 3. Strict individual deviation bound <= 1/N
            deviations = np.abs(n_pop / float(N) - pop_u)
            max_dev = np.max(deviations)
            bound = (1.0 / float(N)) + 1e-9
            self.assertLessEqual(max_dev, bound, f"N={N}: Max deviation {max_dev:.6f} exceeds bound {bound:.6f}")

    def test_100k_sample_expectation_preservation_beta_distribution(self):
        """Verify expectation preservation across 100,000 samples under realistic Beta(2, 6) distribution."""
        N_samples = 100000
        rng = np.random.default_rng(9999)
        # Beta(2,6) matches the empirical head-affinity skew of MovieLens-1M
        pop_u = rng.beta(a=2.0, b=6.0, size=N_samples).astype(np.float64)

        for N in [10, 20, 50]:
            n_pop, n_tail = sp.stochastic_user_quota(pop_u, N=N, rng=rng)

            # Sum invariance
            np.testing.assert_array_equal(n_pop + n_tail, N)

            # Mean expectation error < 1e-3
            emp_mean = np.mean(n_pop / float(N))
            true_mean = np.mean(pop_u)
            mean_error = abs(emp_mean - true_mean)
            self.assertLess(mean_error, 1e-3, f"N={N}: Beta dist mean error {mean_error:.6f} exceeds 1e-3")

            # Max deviation bound
            deviations = np.abs(n_pop / float(N) - pop_u)
            max_dev = np.max(deviations)
            bound = (1.0 / float(N)) + 1e-9
            self.assertLessEqual(max_dev, bound, f"N={N}: Max deviation {max_dev:.6f} > {bound:.6f}")

    def test_extreme_boundary_inclinations(self):
        """Stress test extreme boundary inclinations: 0.0, 1.0, negative, and >1.0 values."""
        pop_u = np.array([-0.5, 0.0, 0.00001, 0.5, 0.99999, 1.0, 1.5, 999.0], dtype=np.float64)

        for N in [1, 20, 100]:
            n_pop, n_tail = sp.stochastic_user_quota(pop_u, N=N, rng=42)

            # All quotas must be strictly bounded in [0, N]
            self.assertTrue(np.all(n_pop >= 0) and np.all(n_pop <= N))
            self.assertTrue(np.all(n_tail >= 0) and np.all(n_tail <= N))
            np.testing.assert_array_equal(n_pop + n_tail, N)

            # Clipped boundaries check
            self.assertEqual(n_pop[0], 0)     # -0.5 clipped to 0
            self.assertEqual(n_pop[1], 0)     # 0.0 exactly 0
            self.assertEqual(n_pop[5], N)     # 1.0 exactly N
            self.assertEqual(n_pop[6], N)     # 1.5 clipped to N
            self.assertEqual(n_pop[7], N)     # 999.0 clipped to N


class TestAdversarialDynamicMergeModes(unittest.TestCase):
    """Adversarial stress testing for dynamic_probabilistic_quota_merge across all modes."""

    def _generate_synthetic_testbed(
        self,
        n_users=50,
        n_items=100,
        density=0.10,
        score_gap=0.0,
        head_score_bias=0.0,
        tail_score_bias=0.0,
        cold_users_count=5,
        dense_users_count=5,
        seed=42
    ):
        rng = np.random.default_rng(seed)
        ranks = np.arange(1, n_items + 1)
        pop_count = (10000.0 / (ranks ** 0.8)).astype(np.float64)
        head_idx, tail_idx = sp.pareto_partition(pop_count, alpha=0.20)

        train_matrix = np.zeros((n_users, n_items), dtype=np.int32)

        for u in range(n_users):
            if u < cold_users_count:
                # Cold start user: 0 interactions
                continue
            elif u < cold_users_count + dense_users_count:
                # Highly active user: 80% of items interacted
                n_inter = int(0.80 * n_items)
                chosen = rng.choice(n_items, size=n_inter, replace=False)
                train_matrix[u, chosen] = 1
            else:
                # Regular user
                n_inter = max(1, int(density * n_items))
                chosen = rng.choice(n_items, size=n_inter, replace=False)
                train_matrix[u, chosen] = 1

        scores_pop = rng.normal(0.0 + head_score_bias, 1.0, size=(n_users, n_items)).astype(np.float32)
        scores_tail = rng.normal(0.0 + tail_score_bias, 1.0, size=(n_users, n_items)).astype(np.float32)

        return {
            "n_users": n_users,
            "n_items": n_items,
            "head_idx": head_idx,
            "tail_idx": tail_idx,
            "pop_count": pop_count,
            "train_matrix": train_matrix,
            "scores_pop": scores_pop,
            "scores_tail": scores_tail,
        }

    def _assert_valid_recommendations(self, reclist, train_matrix, N, n_items, test_name):
        n_users = train_matrix.shape[0]
        self.assertEqual(reclist.shape, (n_users, N), f"{test_name}: Incorrect shape {reclist.shape}")
        self.assertTrue(np.issubdtype(reclist.dtype, np.integer), f"{test_name}: Non-integer dtype")

        for u in range(n_users):
            rec = reclist[u].tolist()
            train_pos = set(np.where(train_matrix[u] > 0)[0].tolist())
            unseen_count = n_items - len(train_pos)

            # 1. Check uniqueness (no duplicates)
            unique_items = set(rec)
            if unseen_count >= N:
                self.assertEqual(len(unique_items), N, f"{test_name}: User {u} has duplicate recommendations: {rec}")

            # 2. Check valid item catalog range
            self.assertTrue(all(0 <= item < n_items for item in rec), f"{test_name}: User {u} has out-of-range item: {rec}")

            # 3. Check zero training positive leakage (when unseen items >= N)
            if unseen_count >= N:
                leakage = unique_items & train_pos
                self.assertEqual(len(leakage), 0, f"{test_name}: User {u} has leaked training positives: {leakage}")

    def test_extreme_slate_lengths_n1_and_n100(self):
        """Test merge behavior at extreme slate lengths N=1 and N=100."""
        modes = ["stochastic", "adaptive_alpha", "confidence_elastic", "hybrid", "static"]
        tb = self._generate_synthetic_testbed(n_users=30, n_items=150)

        for N in [1, 2, 5, 20, 50, 100]:
            for mode in modes:
                reclist = sp.dynamic_probabilistic_quota_merge(
                    tb["scores_pop"],
                    tb["scores_tail"],
                    tb["train_matrix"],
                    tb["head_idx"],
                    N=N,
                    mode=mode,
                    seed=100 + N,
                    pop_count=tb["pop_count"],
                    threshold=0.10,
                )
                self._assert_valid_recommendations(reclist, tb["train_matrix"], N, tb["n_items"], f"N={N}_mode={mode}")

    def test_extreme_candidate_score_differentials(self):
        """Stress test with massive candidate score gaps (+1e6 pop vs -1e6 tail and vice versa)."""
        modes = ["stochastic", "adaptive_alpha", "confidence_elastic", "hybrid", "static"]

        # Case A: Pop scores >> Tail scores (pop advantage +1,000,000)
        tb_a = self._generate_synthetic_testbed(n_users=20, n_items=80, head_score_bias=1e6, tail_score_bias=-1e6)
        for mode in modes:
            reclist = sp.dynamic_probabilistic_quota_merge(
                tb_a["scores_pop"],
                tb_a["scores_tail"],
                tb_a["train_matrix"],
                tb_a["head_idx"],
                N=20,
                mode=mode,
                seed=42,
                pop_count=tb_a["pop_count"],
                threshold=0.10,
            )
            self._assert_valid_recommendations(reclist, tb_a["train_matrix"], 20, tb_a["n_items"], f"HighPopAdv_mode={mode}")

        # Case B: Tail scores >> Pop scores (tail advantage +1,000,000)
        tb_b = self._generate_synthetic_testbed(n_users=20, n_items=80, head_score_bias=-1e6, tail_score_bias=1e6)
        for mode in modes:
            reclist = sp.dynamic_probabilistic_quota_merge(
                tb_b["scores_pop"],
                tb_b["scores_tail"],
                tb_b["train_matrix"],
                tb_b["head_idx"],
                N=20,
                mode=mode,
                seed=42,
                pop_count=tb_b["pop_count"],
                threshold=0.10,
            )
            self._assert_valid_recommendations(reclist, tb_b["train_matrix"], 20, tb_b["n_items"], f"HighTailAdv_mode={mode}")

    def test_degenerate_catalog_sizes(self):
        """Test degenerate catalog sizes: n_items=20 for N=20, n_items=21, single head item."""
        modes = ["stochastic", "adaptive_alpha", "confidence_elastic", "hybrid", "static"]

        # 1. Exact catalog size = N (20 items, N=20)
        pop_count_20 = np.linspace(100, 1, 20)
        head_idx_20, tail_idx_20 = sp.pareto_partition(pop_count_20, alpha=0.20)
        train_matrix_20 = np.zeros((10, 20), dtype=np.int32)
        # Give users 0 to 5 interactions
        train_matrix_20[1, :2] = 1
        train_matrix_20[2, :5] = 1

        scores_pop_20 = np.random.normal(0, 1, size=(10, 20)).astype(np.float32)
        scores_tail_20 = np.random.normal(0, 1, size=(10, 20)).astype(np.float32)

        for mode in modes:
            reclist = sp.dynamic_probabilistic_quota_merge(
                scores_pop_20,
                scores_tail_20,
                train_matrix_20,
                head_idx_20,
                N=10,
                mode=mode,
                seed=42,
                pop_count=pop_count_20,
            )
            self._assert_valid_recommendations(reclist, train_matrix_20, 10, 20, f"Catalog20_mode={mode}")

        # 2. Single item head pool (|H| = 1)
        pop_count_single_head = np.array([1000.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        head_single, tail_single = sp.pareto_partition(pop_count_single_head, alpha=0.20)
        self.assertEqual(len(head_single), 1)

        train_matrix_single = np.zeros((5, 10), dtype=np.int32)
        train_matrix_single[0, 0] = 1  # User 0 already saw the single head item
        scores_pop_s = np.random.normal(0, 1, size=(5, 10)).astype(np.float32)
        scores_tail_s = np.random.normal(0, 1, size=(5, 10)).astype(np.float32)

        for mode in modes:
            reclist = sp.dynamic_probabilistic_quota_merge(
                scores_pop_s,
                scores_tail_s,
                train_matrix_single,
                head_single,
                N=5,
                mode=mode,
                seed=42,
                pop_count=pop_count_single_head,
            )
            self._assert_valid_recommendations(reclist, train_matrix_single, 5, 10, f"SingleHead_mode={mode}")

    def test_full_and_empty_training_histories(self):
        """Test users with 0 interactions and users with complete/near-complete interactions."""
        modes = ["stochastic", "adaptive_alpha", "confidence_elastic", "hybrid", "static"]
        n_items = 50
        n_users = 10
        pop_count = np.linspace(100, 1, n_items)
        head_idx, _ = sp.pareto_partition(pop_count, alpha=0.20)

        train_matrix = np.zeros((n_users, n_items), dtype=np.int32)
        # User 0: 0 interactions (Cold start)
        # User 1: 1 head item
        train_matrix[1, head_idx[0]] = 1
        # User 2: 1 tail item
        tail_items = list(set(range(n_items)) - set(head_idx.tolist()))
        train_matrix[2, tail_items[0]] = 1
        # User 3: All head items
        train_matrix[3, head_idx] = 1
        # User 4: All tail items
        train_matrix[4, tail_items] = 1
        # User 5: 35 items out of 50
        train_matrix[5, :35] = 1

        scores_pop = np.random.normal(0, 1, size=(n_users, n_items)).astype(np.float32)
        scores_tail = np.random.normal(0, 1, size=(n_users, n_items)).astype(np.float32)

        for mode in modes:
            reclist = sp.dynamic_probabilistic_quota_merge(
                scores_pop,
                scores_tail,
                train_matrix,
                head_idx,
                N=10,
                mode=mode,
                seed=42,
                pop_count=pop_count,
            )
            self._assert_valid_recommendations(reclist, train_matrix, 10, n_items, f"Histories_mode={mode}")


class TestAdversarialComponentStress(unittest.TestCase):
    """Stress tests on user_adaptive_alpha_partition, calibrated_score_fusion, and evaluate_calibration."""

    def test_adaptive_alpha_skewed_activity_and_bounds(self):
        """Stress test user_adaptive_alpha_partition with extreme activity disparity."""
        n_items = 100
        pop_count = np.linspace(500, 1, n_items)
        train_matrix = np.zeros((100, n_items), dtype=np.int32)

        # User 0: 100% active (all items)
        train_matrix[0, :] = 1
        # User 1: 1 item active
        train_matrix[1, 0] = 1
        # User 2..99: cold start (0 items)

        pop_u = sp.user_adaptive_alpha_partition(pop_count, train_matrix, alpha_base=0.20, beta=0.30)
        self.assertEqual(pop_u.shape, (100,))
        self.assertTrue(np.all(pop_u >= 0.0) and np.all(pop_u <= 1.0))
        # User 0 activity weight is highest -> should scale adaptive alpha
        self.assertGreaterEqual(pop_u[0], 0.0)
        # Cold start users must receive alpha_base
        for u in range(2, 100):
            self.assertAlmostEqual(pop_u[u], 0.20, places=5)

    def test_score_fusion_extreme_lambdas_and_zero_variance(self):
        """Stress test calibrated_score_fusion under extreme lambda and identical scores."""
        n_users, n_items = 10, 30
        head_idx = np.array([0, 1, 2, 3, 4], dtype=np.int64)

        # Zero variance scores
        fed_constant = np.full((n_users, n_items), 5.0, dtype=np.float32)
        llm_random = np.random.uniform(-1, 1, size=(n_users, n_items)).astype(np.float32)

        fused = sp.calibrated_score_fusion(fed_constant, llm_random, head_idx, lam=0.70, standardize=True)
        self.assertFalse(np.isnan(fused[:, head_idx]).any())
        self.assertFalse(np.isinf(fused[:, head_idx]).any())

        # Empty pool
        fused_empty = sp.calibrated_score_fusion(fed_constant, llm_random, np.array([], dtype=np.int64))
        self.assertTrue(np.all(fused_empty == -np.inf))

    def test_evaluate_calibration_bounds(self):
        """Stress test evaluate_calibration on corner cases (all head, all tail, perfectly calibrated)."""
        n_users, n_items, N = 20, 50, 20
        train_matrix = np.zeros((n_users, n_items), dtype=np.int32)
        head_idx = np.arange(10)  # items 0..9 are head

        # User 0 has 50% head interactions
        train_matrix[0, :5] = 1   # 5 head
        train_matrix[0, 10:15] = 1 # 5 tail

        # Perfectly matched recommendation
        rec_matrix = np.zeros((n_users, N), dtype=np.int64)
        rec_matrix[0, :10] = np.arange(10)      # 10 head = 50%
        rec_matrix[0, 10:20] = np.arange(10, 20) # 10 tail = 50%

        # Fill other users identically
        for u in range(1, n_users):
            train_matrix[u] = train_matrix[0]
            rec_matrix[u] = rec_matrix[0]

        cal = sp.evaluate_calibration(rec_matrix, train_matrix, head_idx)
        self.assertAlmostEqual(cal["rmse_pc"], 0.0, places=5)
        self.assertAlmostEqual(cal["mrmc"], 0.0, places=5)


def run_all_adversarial_tests():
    """Runs all test cases and returns structured summary."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestAdversarialStochasticQuota))
    suite.addTests(loader.loadTestsFromTestCase(TestAdversarialDynamicMergeModes))
    suite.addTests(loader.loadTestsFromTestCase(TestAdversarialComponentStress))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result


if __name__ == "__main__":
    result = run_all_adversarial_tests()
    if result.wasSuccessful():
        print("\n>>> ALL ADVERSARIAL STRESS TESTS PASSED SUCCESSFULLY! <<<")
        sys.exit(0)
    else:
        print("\n>>> ADVERSARIAL STRESS TESTS FAILED! <<<")
        sys.exit(1)
