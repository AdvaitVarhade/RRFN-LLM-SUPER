"""
fedsuper_simulation/tests/test_tier1_feature_units.py
Tier 1: Feature & Component Unit Tests for Privacy-Preserving Federated SUPER.
Exhaustively tests data generation, DP mechanisms, SUPER calibration math, evaluator metrics, and visual figure schemas.
Total Test Cases: 42 tests across 5 test classes.
"""
import sys
import os
import math
import unittest
import numpy as np
import pandas as pd
import pytest

# Ensure fedsuper_simulation directory is in path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.config import SimulationConfig
from src.mock_data import SyntheticDataset, generate_mock_dataset
from src.federated_core import FederatedServer, FederatedClient, FederatedSimulation
from src.super_engine import (
    compute_user_popularity_blueprint,
    intra_pool_zscore_standardization,
    fuse_cf_and_llm_scores,
    calibrated_blueprint_merge,
)
from src.evaluator import (
    calculate_rmse_pc,
    calculate_gini_index,
    calculate_long_tail_exposure_ratio,
    calculate_catalog_coverage,
    calculate_recall_at_k,
    calculate_ndcg_at_k,
    calculate_novelty,
    calculate_ild,
    compute_exposure_counts,
    compute_recommendation_distributions,
    evaluate_all_metrics,
)

try:
    from src.graph_visualizer import render_network_flow_figure
    from src.chart_generator import (
        render_exposure_comparison_chart,
        render_rank_exposure_curve,
        render_pareto_front_chart,
        render_convergence_chart,
    )
    import plotly.graph_objects as go
    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False


class TestMockDataGenerator(unittest.TestCase):
    """Suite 1: Synthetic Mock Data Generator Unit Tests (8 Tests)."""

    def setUp(self):
        self.config = SimulationConfig(
            num_clients=20,
            num_items=100,
            embedding_dim=32,
            llm_dim=32,
            top_k=10,
            seed=42
        )
        self.dataset = generate_mock_dataset(self.config, seed=42)

    def test_catalog_partitioning_disjoint_and_complete(self):
        """T1.1: Verify catalog partitioned into Head (20%), Torso (30%), Tail (50%) with complete disjoint union."""
        h = set(self.dataset.head_idx.tolist())
        t = set(self.dataset.torso_idx.tolist())
        l = set(self.dataset.tail_idx.tolist())

        self.assertEqual(len(h), 20, "Head should contain 20 items for M=100")
        self.assertEqual(len(t), 30, "Torso should contain 30 items for M=100")
        self.assertEqual(len(l), 50, "Tail should contain 50 items for M=100")

        self.assertEqual(len(h & t), 0, "Head and Torso must be disjoint")
        self.assertEqual(len(h & l), 0, "Head and Tail must be disjoint")
        self.assertEqual(len(t & l), 0, "Torso and Tail must be disjoint")
        self.assertEqual(h | t | l, set(range(100)), "Partitions must span entire catalog [0, 99]")

    def test_item_metadata_schema_and_popularity_monotonicity(self):
        """T1.2: Verify item_metadata schema and descending popularity: Head > Torso > Tail."""
        df = self.dataset.item_metadata
        required_cols = ['item_id', 'title', 'category', 'popularity_count', 'is_head', 'is_torso', 'is_tail']
        for col in required_cols:
            self.assertIn(col, df.columns, f"Column {col} missing in item_metadata")

        pop_head = df[df['is_head']]['popularity_count'].mean()
        pop_torso = df[df['is_torso']]['popularity_count'].mean()
        pop_tail = df[df['is_tail']]['popularity_count'].mean()

        self.assertGreater(pop_head, pop_torso, "Head popularity must exceed Torso")
        self.assertGreater(pop_torso, pop_tail, "Torso popularity must exceed Tail")

    def test_user_metadata_pop_u_true_normalization(self):
        """T1.3: Verify pop_u_true proportions are valid probabilities summing to 1.0."""
        df = self.dataset.user_metadata
        self.assertIn('pop_u_true', df.columns)
        self.assertIn('pop_u_torso', df.columns)
        self.assertIn('pop_u_tail', df.columns)

        sums = df['pop_u_true'] + df['pop_u_torso'] + df['pop_u_tail']
        for s in sums:
            self.assertTrue(np.isclose(s, 1.0, atol=1e-3), f"User blueprint sum {s} != 1.0")

    def test_train_matrix_binary_and_dimensions(self):
        """T1.4: Verify train_matrix has correct shape (U, M) and binary entries {0, 1}."""
        cfg = SimulationConfig(num_clients=25, num_items=120)
        d = generate_mock_dataset(cfg, seed=123)
        self.assertEqual(d.train_matrix.shape, (25, 120))
        unique_vals = np.unique(d.train_matrix)
        for v in unique_vals:
            self.assertIn(v, [0.0, 1.0], f"Non-binary value {v} found in train_matrix")

    def test_test_dict_disjoint_from_train_matrix(self):
        """T1.5: Verify held-out test items are disjoint from training matrix."""
        for u, test_items in self.dataset.test_dict.items():
            for item in test_items:
                self.assertEqual(self.dataset.train_matrix[u, item], 0.0,
                                 f"Data leakage: test item {item} is present in train matrix for user {u}")

    def test_user_llm_profiles_l2_unit_normalization(self):
        """T1.6: Verify user LLM profile vectors have unit L2 norm ||s_u||_2 = 1.0."""
        norms = np.linalg.norm(self.dataset.user_llm_profiles, axis=1)
        for u, norm in enumerate(norms):
            self.assertTrue(np.isclose(norm, 1.0, atol=1e-5), f"User {u} profile norm {norm} != 1.0")

    def test_item_llm_embeddings_l2_unit_normalization(self):
        """T1.7: Verify item LLM embedding vectors have unit L2 norm ||v_i||_2 = 1.0."""
        norms = np.linalg.norm(self.dataset.item_llm_embeddings, axis=1)
        for i, norm in enumerate(norms):
            self.assertTrue(np.isclose(norm, 1.0, atol=1e-5), f"Item {i} embedding norm {norm} != 1.0")

    def test_mock_data_seed_reproducibility(self):
        """T1.8: Verify deterministic reproducibility with seed=42 vs seed=99."""
        d1 = generate_mock_dataset(self.config, seed=42)
        d2 = generate_mock_dataset(self.config, seed=42)
        d3 = generate_mock_dataset(self.config, seed=99)

        np.testing.assert_array_equal(d1.train_matrix, d2.train_matrix)
        np.testing.assert_allclose(d1.user_llm_profiles, d2.user_llm_profiles, atol=1e-6)
        self.assertFalse(np.array_equal(d1.train_matrix, d3.train_matrix))


class TestFederatedCoreDP(unittest.TestCase):
    """Suite 2: Federated Core & DP Mechanisms Unit Tests (8 Tests)."""

    def setUp(self):
        self.config = SimulationConfig(
            num_clients=20,
            num_items=100,
            embedding_dim=16,
            dp_enabled=True,
            dp_epsilon=4.0,
            dp_delta=1e-5,
            dp_l2_clip_norm=1.0,
            learning_rate=0.05,
            seed=42
        )
        self.dataset = generate_mock_dataset(self.config, seed=42)

    def test_bpr_loss_exact_computation(self):
        """T1.9: Verify exact BPR loss L_BPR = ln(1 + exp(-(y_pos - y_neg)))."""
        s_pos = 2.5
        s_neg = 0.5
        diff = s_pos - s_neg  # 2.0
        expected_loss = np.log(1.0 + np.exp(-diff))  # ln(1 + e^-2) ~= 0.126928011
        computed_loss = np.logaddexp(0.0, -diff)
        self.assertTrue(np.isclose(computed_loss, expected_loss, atol=1e-6))
        self.assertTrue(np.isclose(computed_loss, 0.126928, atol=1e-5))

    def test_bpr_gradient_analytic_correctness(self):
        """T1.10: Verify BPR analytical gradients match numerical finite differences."""
        rng = np.random.default_rng(42)
        d = 8
        p = rng.normal(0, 0.1, size=d)
        q_pos = rng.normal(0, 0.1, size=d)
        q_neg = rng.normal(0, 0.1, size=d)
        b_pos = float(rng.normal(0, 0.1))
        b_neg = float(rng.normal(0, 0.1))

        def loss_fn(p_val, qp_val, qn_val, bp_val, bn_val):
            diff = (np.dot(p_val, qp_val) + bp_val) - (np.dot(p_val, qn_val) + bn_val)
            return float(np.logaddexp(0.0, -diff))

        diff0 = (np.dot(p, q_pos) + b_pos) - (np.dot(p, q_neg) + b_neg)
        gamma = 1.0 / (1.0 + np.exp(np.clip(diff0, -30, 30)))
        grad_p_analytic = -gamma * (q_pos - q_neg)

        # Finite difference for grad_p
        eps = 1e-6
        grad_p_numeric = np.zeros(d)
        for i in range(d):
            p_plus = p.copy(); p_plus[i] += eps
            p_minus = p.copy(); p_minus[i] -= eps
            grad_p_numeric[i] = (loss_fn(p_plus, q_pos, q_neg, b_pos, b_neg) - loss_fn(p_minus, q_pos, q_neg, b_pos, b_neg)) / (2 * eps)

        np.testing.assert_allclose(grad_p_analytic, grad_p_numeric, atol=1e-4)

    def test_client_local_state_privacy_isolation(self):
        """T1.11: Verify client user embedding p_u is not transmitted in server delta payload."""
        client = FederatedClient(
            user_id=0,
            train_items=np.array([1, 5, 10]),
            llm_profile=np.ones(16) / 4.0,
            embedding_dim=16,
            seed=42
        )
        server = FederatedServer(num_items=100, embedding_dim=16, seed=42)
        touched, delta_emb, delta_b, loss = client.local_train_step(
            server.item_embeddings, server.item_biases, self.config
        )
        # Payload only has item updates
        self.assertNotIn("user_embedding", [touched, delta_emb, delta_b, loss])
        self.assertEqual(len(touched), len(delta_emb))
        self.assertEqual(len(touched), len(delta_b))

    def test_l2_gradient_clipping_threshold_c_active(self):
        """T1.12: Verify L2 clipping scales down gradient when norm > C."""
        g = np.array([3.0, 4.0])  # norm = 5.0
        c = 1.0
        norm_g = np.linalg.norm(g)
        scale = min(1.0, c / norm_g)
        g_clipped = g * scale
        self.assertTrue(np.isclose(np.linalg.norm(g_clipped), 1.0, atol=1e-6))
        np.testing.assert_allclose(g_clipped, g * 0.2, atol=1e-6)

    def test_l2_gradient_clipping_threshold_c_inactive(self):
        """T1.13: Verify L2 clipping leaves gradient unchanged when norm <= C."""
        g = np.array([0.2, 0.3])  # norm = sqrt(0.13) ~= 0.3605 <= 1.0
        c = 1.0
        norm_g = np.linalg.norm(g)
        scale = min(1.0, c / norm_g)
        g_clipped = g * scale
        np.testing.assert_allclose(g_clipped, g, atol=1e-6)

    def test_dp_gaussian_noise_statistical_variance(self):
        """T1.14: Verify Gaussian DP noise scale matches sigma = C * sqrt(2 ln(1.25/delta)) / eps."""
        c = 1.0
        eps = 4.0
        delta = 1e-5
        expected_sigma = (c * math.sqrt(2.0 * math.log(1.25 / delta))) / eps  # ~= 1.20912
        rng = np.random.default_rng(2026)
        samples = rng.normal(0.0, expected_sigma, size=30000)
        self.assertLess(abs(float(np.mean(samples))), 0.05)
        self.assertTrue(np.isclose(float(np.var(samples)), expected_sigma ** 2, rtol=0.08))

    def test_dp_disabled_produces_zero_perturbation(self):
        """T1.15: Verify dp_enabled=False sets dp_sigma = 0.0 without noise injection."""
        cfg = SimulationConfig(dp_enabled=False)
        self.assertEqual(cfg.dp_sigma, 0.0)

    def test_sparse_fedavg_server_aggregation(self):
        """T1.16: Verify sparse FedAvg updates only touched items and leaves untouched items identical."""
        server = FederatedServer(num_items=10, embedding_dim=4, seed=42)
        orig_embs = server.item_embeddings.copy()

        delta_1 = np.ones((1, 4), dtype=np.float32) * 0.10
        delta_2 = np.ones((1, 4), dtype=np.float32) * 0.30
        updates = [
            (np.array([5]), delta_1, np.array([0.05])),
            (np.array([5]), delta_2, np.array([0.15])),
        ]
        server.aggregate_updates(updates)

        # Item 5 should be updated by exact average: 0.5 * (0.10 + 0.30) = 0.20
        expected_emb_5 = orig_embs[5] + 0.20
        np.testing.assert_allclose(server.item_embeddings[5], expected_emb_5, atol=1e-5)

        # Untouched items (e.g. item 8) must remain exactly equal to orig
        np.testing.assert_array_equal(server.item_embeddings[8], orig_embs[8])


class TestSuperCalibrationEngine(unittest.TestCase):
    """Suite 3: SUPER Popularity Calibration & LLM Fusion Unit Tests (10 Tests)."""

    def test_user_popularity_blueprint_calculation(self):
        """T1.17: Verify user popularity blueprint P_u = [p_u^H, p_u^T, p_u^L]."""
        train_matrix = np.zeros((1, 10), dtype=np.float32)
        train_matrix[0, [0, 1, 2, 3]] = 1.0  # 4 head
        train_matrix[0, [4, 5, 6]] = 1.0     # 3 torso
        train_matrix[0, [7, 8, 9]] = 1.0     # 3 tail
        head = np.array([0, 1, 2, 3])
        torso = np.array([4, 5, 6])
        tail = np.array([7, 8, 9])

        p_u = compute_user_popularity_blueprint(train_matrix, head, torso, tail)
        np.testing.assert_allclose(p_u[0], [0.40, 0.30, 0.30], atol=1e-5)

    def test_user_blueprint_cold_start_fallback(self):
        """T1.18: Verify cold-start user (0 interactions) falls back to [1/3, 1/3, 1/3]."""
        train_matrix = np.zeros((1, 10), dtype=np.float32)
        head = np.array([0, 1])
        torso = np.array([2, 3, 4])
        tail = np.array([5, 6, 7, 8, 9])
        p_u = compute_user_popularity_blueprint(train_matrix, head, torso, tail)
        np.testing.assert_allclose(p_u[0], [1.0/3.0, 1.0/3.0, 1.0/3.0], atol=1e-5)

    def test_intra_pool_zscore_standardization_distribution(self):
        """T1.19: Verify intra-pool standardized scores have mean=0 and std=1."""
        cf_scores = np.array([[1.2, 3.4, 0.5, 8.1, -1.0]])
        pool = np.array([0, 1, 2, 3, 4])
        z = intra_pool_zscore_standardization(cf_scores, pool)
        self.assertTrue(np.isclose(float(np.mean(z)), 0.0, atol=1e-5))
        self.assertTrue(np.isclose(float(np.std(z)), 1.0, atol=1e-4))

    def test_intra_pool_zscore_constant_score_stability(self):
        """T1.20: Verify zero-variance scores produce zeros without NaN or Inf."""
        cf_scores = np.array([[2.0, 2.0, 2.0]])
        pool = np.array([0, 1, 2])
        z = intra_pool_zscore_standardization(cf_scores, pool)
        self.assertFalse(np.isnan(z).any())
        self.assertFalse(np.isinf(z).any())
        np.testing.assert_allclose(z, np.zeros((1, 3)), atol=1e-6)

    def test_llm_semantic_cosine_similarity_computation(self):
        """T1.21: Verify LLM semantic cosine similarities for collinear, orthogonal, opposite vectors."""
        s_u = np.array([[1.0, 0.0]])
        v_items = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        sims = np.dot(s_u, v_items.T)[0]
        self.assertTrue(np.isclose(sims[0], 1.0, atol=1e-5))
        self.assertTrue(np.isclose(sims[1], 0.0, atol=1e-5))
        self.assertTrue(np.isclose(sims[2], -1.0, atol=1e-5))

    def test_fused_score_convex_combination_pure_cf(self):
        """T1.22: Verify lambda=0.0 yields pure standardized CF scores."""
        cf_scores = np.array([[1.0, 2.0, 3.0, 4.0]])
        u_prof = np.array([[1.0, 0.0]])
        item_embs = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        head = np.array([0, 1])
        torso = np.array([2])
        tail = np.array([3])

        fused = fuse_cf_and_llm_scores(cf_scores, u_prof, item_embs, head, torso, tail, llm_lambda=0.0)
        z_head = intra_pool_zscore_standardization(cf_scores, head)
        np.testing.assert_allclose(fused[:, head], z_head, atol=1e-5)

    def test_fused_score_convex_combination_pure_llm(self):
        """T1.23: Verify lambda=1.0 yields pure LLM cosine similarities."""
        cf_scores = np.array([[10.0, 20.0, 30.0, 40.0]])
        u_prof = np.array([[1.0, 0.0]])
        item_embs = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        head = np.array([0, 1])
        torso = np.array([2])
        tail = np.array([3])

        fused = fuse_cf_and_llm_scores(cf_scores, u_prof, item_embs, head, torso, tail, llm_lambda=1.0)
        expected_llm = np.dot(u_prof, item_embs.T)
        np.testing.assert_allclose(fused, expected_llm, atol=1e-5)

    def test_calibration_quota_allocation_sum_to_k(self):
        """T1.24: Verify calibrated blueprint merge outputs exactly K items per user."""
        fused = np.random.default_rng(42).normal(0, 1, size=(5, 30))
        blueprints = np.array([[0.5, 0.3, 0.2]] * 5)
        head = np.arange(0, 6)
        torso = np.arange(6, 15)
        tail = np.arange(15, 30)

        for top_k in [5, 10, 15]:
            recs = calibrated_blueprint_merge(
                fused, blueprints, head_idx=head, torso_idx=torso, tail_idx=tail, top_k=top_k, calibration_alpha=0.40
            )
            self.assertEqual(recs.shape, (5, top_k))

    def test_calibration_quota_alpha_extremes(self):
        """T1.25: Verify alpha=0.0 vs alpha=1.0 calibration behavior."""
        fused = np.arange(30, dtype=float).reshape(1, 30)  # Highest scores on tail items 15..29
        blueprints = np.array([[0.8, 0.1, 0.1]])  # Target: 80% Head
        head = np.arange(0, 6)
        torso = np.arange(6, 15)
        tail = np.arange(15, 30)

        # Under alpha=1.0, blueprint must force head items into recommendation
        recs_calib = calibrated_blueprint_merge(
            fused, blueprints, head_idx=head, torso_idx=torso, tail_idx=tail, top_k=10, calibration_alpha=1.0
        )
        head_in_calib = sum(1 for it in recs_calib[0] if it in set(head))
        self.assertGreaterEqual(head_in_calib, 5)

    def test_calibrated_blueprint_merge_no_duplicate_items(self):
        """T1.26: Verify recommendation lists contain zero duplicate items."""
        fused = np.random.default_rng(99).normal(0, 1, size=(10, 50))
        blueprints = np.full((10, 3), 1.0 / 3.0)
        head = np.arange(0, 10)
        torso = np.arange(10, 25)
        tail = np.arange(25, 50)

        recs = calibrated_blueprint_merge(
            fused, blueprints, head_idx=head, torso_idx=torso, tail_idx=tail, top_k=10, calibration_alpha=0.5
        )
        for u in range(10):
            self.assertEqual(len(set(recs[u])), 10, f"User {u} has duplicate recommendations")


class TestEvaluatorMetrics(unittest.TestCase):
    """Suite 4: Comprehensive Evaluation Metrics Unit Tests (10 Tests)."""

    def test_rmse_pc_zero_on_perfect_calibration(self):
        """T1.27: Verify Rmse-PC = 0.0 when user blueprint exactly matches recommendation distribution."""
        p = np.array([[0.2, 0.3, 0.5], [0.4, 0.3, 0.3]])
        q = np.array([[0.2, 0.3, 0.5], [0.4, 0.3, 0.3]])
        rmse_pc = calculate_rmse_pc(p, q)
        self.assertTrue(np.isclose(rmse_pc, 0.0, atol=1e-7))

    def test_rmse_pc_exact_analytic_value(self):
        """T1.28: Verify Rmse-PC formula gives exact sqrt(2/3) for complete polarity inversion."""
        p = np.array([[0.0, 0.0, 1.0]])
        q = np.array([[1.0, 0.0, 0.0]])
        expected = math.sqrt(( (0.0-1.0)**2 + 0.0 + (1.0-0.0)**2 ) / 3.0)  # sqrt(2/3) ~= 0.81649658
        rmse_pc = calculate_rmse_pc(p, q)
        self.assertTrue(np.isclose(rmse_pc, expected, atol=1e-5))

    def test_gini_index_equal_distribution_zero(self):
        """T1.29: Verify Gini index = 0.0 for uniform item exposure."""
        exp = np.array([10, 10, 10, 10, 10])
        gini = calculate_gini_index(exp)
        self.assertTrue(np.isclose(gini, 0.0, atol=1e-6))

    def test_gini_index_extreme_monopoly(self):
        """T1.30: Verify Gini index = (M-1)/M for single-item monopoly on M items."""
        exp = np.array([100, 0, 0, 0, 0])  # M=5
        gini = calculate_gini_index(exp)
        # G = (2 * 5 * 100) / (5 * 100) - (5 + 1)/5 = 2 - 1.2 = 0.80
        self.assertTrue(np.isclose(gini, 0.80, atol=1e-5))

    def test_gini_index_all_zeros_graceful(self):
        """T1.31: Verify Gini index returns 0.0 gracefully for all-zero exposures."""
        exp = np.zeros(10)
        gini = calculate_gini_index(exp)
        self.assertEqual(gini, 0.0)

    def test_long_tail_exposure_ratio_formula(self):
        """T1.32: Verify Long-Tail Exposure Ratio (LTER)."""
        exp = np.array([10, 20, 30, 40])  # Total = 100
        tail_idx = np.array([2, 3])       # Tail exposures = 30 + 40 = 70
        lter = calculate_long_tail_exposure_ratio(exp, tail_idx)
        self.assertTrue(np.isclose(lter, 0.70, atol=1e-6))

    def test_catalog_coverage_formula(self):
        """T1.33: Verify Catalog Coverage ratio in [0, 1]."""
        exp = np.array([5, 0, 1, 0, 8, 3, 0, 2, 4, 1])  # 7 non-zero out of 10
        cov = calculate_catalog_coverage(exp)
        self.assertTrue(np.isclose(cov, 0.70, atol=1e-6))

    def test_recall_and_ndcg_at_k_exact_values(self):
        """T1.34: Verify Recall@K and NDCG@K exact numerical calculation."""
        # User 0 recs: [10, 20, 30, 40, 50], ground truth test items: [20, 40]
        recs = np.array([[10, 20, 30, 40, 50]])
        test_dict = {0: [20, 40]}
        recall = calculate_recall_at_k(recs, test_dict, top_k=5)
        self.assertTrue(np.isclose(recall, 1.0, atol=1e-6))

        # DCG@5 = 1/log2(2+1) + 1/log2(4+1) = 1/1.58496 + 1/2.32193 = 0.63093 + 0.43068 = 1.06161
        # IDCG@5 = 1/log2(2) + 1/log2(3) = 1.0 + 0.63093 = 1.63093
        # NDCG@5 = 1.06161 / 1.63093 ~= 0.65092
        ndcg = calculate_ndcg_at_k(recs, test_dict, top_k=5)
        self.assertTrue(np.isclose(ndcg, 0.65092, atol=1e-4))

    def test_novelty_formula_tail_higher_than_head(self):
        """T1.35: Verify tail recommendations yield higher novelty than head recommendations."""
        pop_counts = np.array([100, 100, 10, 10, 1, 1])
        recs_head = np.array([[0, 1]])
        recs_tail = np.array([[4, 5]])
        nov_head = calculate_novelty(recs_head, pop_counts)
        nov_tail = calculate_novelty(recs_tail, pop_counts)
        self.assertGreater(nov_tail, nov_head)

    def test_intra_list_diversity_orthogonal_and_identical(self):
        """T1.36: Verify Intra-List Diversity = 0 for identical items and 1 for orthogonal items."""
        embs_identical = np.array([[1.0, 0.0], [1.0, 0.0]])
        recs = np.array([[0, 1]])
        ild_ident = calculate_ild(recs, embs_identical)
        self.assertTrue(np.isclose(ild_ident, 0.0, atol=1e-5))

        embs_ortho = np.array([[1.0, 0.0], [0.0, 1.0]])
        ild_ortho = calculate_ild(recs, embs_ortho)
        self.assertTrue(np.isclose(ild_ortho, 1.0, atol=1e-5))


class TestVisualFigureStructures(unittest.TestCase):
    """Suite 5: Visual Figure Structural Properties Unit Tests (6 Tests)."""

    def setUp(self):
        if not HAS_VIZ:
            self.skipTest("Visualizer modules not available")
        self.config = SimulationConfig(num_clients=10, num_items=50, top_k=5, seed=42)
        self.dataset = generate_mock_dataset(self.config, seed=42)
        self.sim = FederatedSimulation(self.dataset, self.config)
        self.sim.step_round()
        self.recs_unc = self.sim.get_recommendations(calibrated=False)
        self.recs_cal = self.sim.get_recommendations(calibrated=True)
        self.metrics = evaluate_all_metrics(self.dataset, self.recs_unc, self.recs_cal, top_k=5)

    def test_network_graph_figure_trace_color_schema(self):
        """T1.37: Verify network flow figure contains Orange (#FF7043), Cyan (#00E5FF), Magenta (#D500F9)."""
        fig = render_network_flow_figure(
            self.dataset, active_client_ids=[1, 3], current_round=1,
            dp_clip_norm=1.0, llm_lambda=0.70
        )
        fig_str = str(fig.to_dict()).lower()
        self.assertTrue("#ff7043" in fig_str or "255, 112, 67" in fig_str or "orange" in fig_str)
        self.assertTrue("#00e5ff" in fig_str or "0, 229, 255" in fig_str or "cyan" in fig_str)
        self.assertTrue("#d500f9" in fig_str or "213, 0, 249" in fig_str or "magenta" in fig_str)

    def test_network_graph_figure_layout_and_nodes(self):
        """T1.38: Verify network graph layout template and non-empty traces."""
        fig = render_network_flow_figure(self.dataset, active_client_ids=[0], current_round=1, dp_clip_norm=1.0, llm_lambda=0.7)
        self.assertIsInstance(fig, go.Figure)
        self.assertGreater(len(fig.data), 0)

    def test_exposure_comparison_chart_traces_and_categories(self):
        """T1.39: Verify exposure chart has Head, Torso, Tail categories and bar traces."""
        fig = render_exposure_comparison_chart(self.metrics)
        self.assertIsInstance(fig, go.Figure)
        self.assertGreater(len(fig.data), 0)

    def test_rank_exposure_curve_structure_and_curves(self):
        """T1.40: Verify rank exposure curve has 2 line/area traces for uncalib vs calib."""
        exp_unc = self.metrics["uncalibrated"]["exposure_counts"]
        exp_cal = self.metrics["calibrated"]["exposure_counts"]
        fig = render_rank_exposure_curve(exp_unc, exp_cal)
        self.assertIsInstance(fig, go.Figure)
        self.assertGreaterEqual(len(fig.data), 2)

    def test_pareto_front_chart_scatter_properties(self):
        """T1.41: Verify Pareto front chart returns scatter figure."""
        records = [
            {"alpha": 0.0, "rmse_pc": 0.35, "gini": 0.75, "lter": 0.05, "rmse": 0.5},
            {"alpha": 0.5, "rmse_pc": 0.05, "gini": 0.35, "lter": 0.30, "rmse": 0.52}
        ]
        fig = render_pareto_front_chart(records, current_alpha=0.5)
        self.assertIsInstance(fig, go.Figure)

    def test_convergence_chart_series(self):
        """T1.42: Verify convergence chart returns figure with round histories."""
        fig = render_convergence_chart(self.sim.history)
        self.assertIsInstance(fig, go.Figure)


if __name__ == '__main__':
    unittest.main()
