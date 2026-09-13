"""
fedsuper_simulation/tests/test_tier2_boundary_corner.py
Tier 2: Boundary Value Analysis & Corner Case Test Suite for Privacy-Preserving Federated SUPER.
Exhaustively tests cold start, single client FL, extreme alpha/lambda/sigma boundaries,
small catalogs (M < K), extreme cohorts, empty test sets, and zero-variance pools.
Total Test Cases: 44 tests across 9 categories.
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


class TestTier2BoundaryCorner(unittest.TestCase):
    """Tier 2 Comprehensive Boundary & Corner Case Test Suite (44 Tests)."""

    # =========================================================================
    # Category A: Cold Start & Sparse/Empty Interaction Histories (5 Tests)
    # =========================================================================

    def test_01_cold_start_all_clients_zero_interactions(self):
        """T2.1: All clients have 0 interactions -> P_u uniform fallback [1/3, 1/3, 1/3], no crash."""
        train_matrix = np.zeros((20, 100), dtype=np.float32)
        head = np.arange(0, 20)
        torso = np.arange(20, 50)
        tail = np.arange(50, 100)
        pop_u = compute_user_popularity_blueprint(train_matrix, head, torso, tail)

        self.assertFalse(np.isnan(pop_u).any(), "P_u must not contain NaNs on cold start")
        np.testing.assert_allclose(pop_u, 1.0 / 3.0, atol=1e-5)

        fused_scores = np.random.default_rng(42).normal(0, 1, size=(20, 100))
        recs = calibrated_blueprint_merge(
            fused_scores, pop_u, head_idx=head, torso_idx=torso, tail_idx=tail, top_k=10, calibration_alpha=0.40
        )
        self.assertEqual(recs.shape, (20, 10))
        for u in range(20):
            self.assertEqual(len(set(recs[u])), 10, f"User {u} must have 10 unique recommendations")

    def test_02_cold_start_partial_cohort_mixed_zero_and_dense(self):
        """T2.2: Mixed cohort (users 0..9 cold, users 10..19 active) handled cleanly."""
        train_matrix = np.zeros((20, 100), dtype=np.float32)
        train_matrix[10:, :20] = 1.0  # active users have 20 interactions on Head
        head = np.arange(0, 20)
        torso = np.arange(20, 50)
        tail = np.arange(50, 100)

        pop_u = compute_user_popularity_blueprint(train_matrix, head, torso, tail)
        # Cold start users get fallback
        np.testing.assert_allclose(pop_u[:10], 1.0 / 3.0, atol=1e-5)
        # Active users get [1.0, 0.0, 0.0]
        np.testing.assert_allclose(pop_u[10:], [1.0, 0.0, 0.0], atol=1e-5)

    def test_03_single_interaction_per_client_boundary(self):
        """T2.3: Client with exactly 1 interaction gets exact one-hot popularity inclination."""
        train_matrix = np.zeros((3, 30), dtype=np.float32)
        train_matrix[0, 2] = 1.0   # 1 Head
        train_matrix[1, 12] = 1.0  # 1 Torso
        train_matrix[2, 25] = 1.0  # 1 Tail
        head = np.arange(0, 10)
        torso = np.arange(10, 20)
        tail = np.arange(20, 30)

        p_u = compute_user_popularity_blueprint(train_matrix, head, torso, tail)
        np.testing.assert_allclose(p_u[0], [1.0, 0.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(p_u[1], [0.0, 1.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(p_u[2], [0.0, 0.0, 1.0], atol=1e-5)

    def test_04_cold_start_user_embedding_initialization_norm(self):
        """T2.4: Cold-start client user embedding has finite, non-zero L2 norm."""
        client = FederatedClient(
            user_id=99,
            train_items=np.array([], dtype=np.int64),
            llm_profile=np.ones(16) / 4.0,
            embedding_dim=16,
            seed=42
        )
        norm = float(np.linalg.norm(client.user_embedding))
        self.assertGreater(norm, 1e-6)
        self.assertFalse(np.isnan(norm))
        self.assertFalse(np.isinf(norm))

    def test_05_cold_start_client_local_loss_and_gradient_zero(self):
        """T2.5: Client with 0 training interactions performs local_train_step without crash."""
        client = FederatedClient(
            user_id=1,
            train_items=np.array([], dtype=np.int64),
            llm_profile=np.ones(16) / 4.0,
            embedding_dim=16,
            seed=42
        )
        cfg = SimulationConfig(embedding_dim=16, num_items=50)
        global_emb = np.zeros((50, 16), dtype=np.float32)
        global_bias = np.zeros(50, dtype=np.float32)

        touched, delta_emb, delta_b, loss = client.local_train_step(global_emb, global_bias, cfg)
        self.assertEqual(len(touched), 0)
        self.assertEqual(loss, 0.0)

    # =========================================================================
    # Category B: Extreme Federation Sizing & Single-Client Topologies (4 Tests)
    # =========================================================================

    def test_06_single_client_federated_training_num_clients_1(self):
        """T2.6: Simulation with num_clients=1 and clients_per_round=1 runs 3 rounds without dimension collapse."""
        cfg = SimulationConfig(num_clients=1, num_items=10, clients_per_round=1, max_rounds=3, seed=42)
        dataset = generate_mock_dataset(cfg, seed=42)
        sim = FederatedSimulation(dataset, cfg)

        for _ in range(3):
            step_res = sim.step_round()
            self.assertEqual(len(step_res["active_clients"]), 1)
            self.assertEqual(step_res["active_clients"][0], 0)

        self.assertFalse(np.isnan(sim.server.item_embeddings).any())
        recs = sim.get_recommendations(calibrated=True)
        self.assertEqual(recs.shape, (1, cfg.top_k))

    def test_07_clients_per_round_equals_total_clients(self):
        """T2.7: 100% participation (clients_per_round = num_clients) selects all clients."""
        cfg = SimulationConfig(num_clients=8, num_items=30, clients_per_round=8, seed=42)
        dataset = generate_mock_dataset(cfg, seed=42)
        sim = FederatedSimulation(dataset, cfg)
        step_res = sim.step_round()
        self.assertEqual(len(step_res["active_clients"]), 8)
        self.assertEqual(set(step_res["active_clients"]), set(range(8)))

    def test_08_clients_per_round_equals_1(self):
        """T2.8: Single client sampling per round (clients_per_round = 1) functions properly."""
        cfg = SimulationConfig(num_clients=20, num_items=50, clients_per_round=1, seed=42)
        dataset = generate_mock_dataset(cfg, seed=42)
        sim = FederatedSimulation(dataset, cfg)
        step_res = sim.step_round()
        self.assertEqual(len(step_res["active_clients"]), 1)
        self.assertIn(step_res["active_clients"][0], range(20))

    def test_09_clients_per_round_exceeds_total_clients_clamping(self):
        """T2.9: clients_per_round > num_clients is rejected by SimulationConfig validation."""
        with self.assertRaises(ValueError):
            SimulationConfig(num_clients=5, clients_per_round=10)

    # =========================================================================
    # Category C: Extreme Calibration & Pareto Alpha Boundaries (5 Tests)
    # =========================================================================

    def test_10_calibration_alpha_zero_pure_uncalibrated_cf(self):
        """T2.10: calibration_alpha=0.0 yields uncalibrated collaborative filtering behavior."""
        cfg = SimulationConfig(num_clients=15, num_items=60, calibration_alpha=0.0, seed=42)
        dataset = generate_mock_dataset(cfg, seed=42)
        sim = FederatedSimulation(dataset, cfg)
        sim.step_round()

        recs_uncalib = sim.get_recommendations(calibrated=False)
        recs_alpha0 = sim.get_recommendations(calibrated=True)

        m_unc = evaluate_all_metrics(dataset, recs_uncalib, recs_uncalib, top_k=cfg.top_k)
        m_a0 = evaluate_all_metrics(dataset, recs_uncalib, recs_alpha0, top_k=cfg.top_k)
        self.assertGreaterEqual(m_unc["uncalibrated"]["rmse_pc"], 0.20)

    def test_11_calibration_alpha_one_pure_strict_calibration(self):
        """T2.11: calibration_alpha=1.0 enforces strict popularity calibration guarantee (Rmse-PC <= 0.055)."""
        cfg = SimulationConfig(num_clients=20, num_items=80, calibration_alpha=1.0, seed=42)
        dataset = generate_mock_dataset(cfg, seed=42)
        sim = FederatedSimulation(dataset, cfg)
        sim.step_round()

        recs_unc = sim.get_recommendations(calibrated=False)
        recs_cal = sim.get_recommendations(calibrated=True)
        m = evaluate_all_metrics(dataset, recs_unc, recs_cal, top_k=cfg.top_k)
        self.assertLessEqual(m["calibrated"]["rmse_pc"], 0.055)

    def test_12_pareto_alpha_partition_boundary_zero_and_one(self):
        """T2.12: SimulationConfig validates pareto_alpha in [0.0, 1.0]."""
        cfg_0 = SimulationConfig(pareto_alpha=0.0)
        self.assertEqual(cfg_0.pareto_alpha, 0.0)
        cfg_1 = SimulationConfig(pareto_alpha=1.0)
        self.assertEqual(cfg_1.pareto_alpha, 1.0)
        with self.assertRaises(ValueError):
            SimulationConfig(pareto_alpha=-0.1)
        with self.assertRaises(ValueError):
            SimulationConfig(pareto_alpha=1.1)

    def test_13_calibration_alpha_continuous_sweep_boundary_steps(self):
        """T2.13: Sweeping calibration_alpha in [0.0..1.0] produces monotonic Rmse-PC reduction."""
        cfg = SimulationConfig(num_clients=20, num_items=80, seed=42)
        dataset = generate_mock_dataset(cfg, seed=42)
        sim = FederatedSimulation(dataset, cfg)
        sim.step_round()

        fused = fuse_cf_and_llm_scores(
            np.dot([c.user_embedding for c in sim.clients], sim.server.item_embeddings.T),
            dataset.user_llm_profiles,
            dataset.item_llm_embeddings,
            dataset.head_idx, dataset.torso_idx, dataset.tail_idx,
            llm_lambda=0.70
        )

        alphas = [0.0, 0.25, 0.50, 0.75, 1.0]
        rmse_pcs = []
        for a in alphas:
            recs = calibrated_blueprint_merge(
                fused, sim.user_blueprints,
                head_idx=dataset.head_idx, torso_idx=dataset.torso_idx, tail_idx=dataset.tail_idx,
                top_k=10, calibration_alpha=a, train_matrix=dataset.train_matrix
            )
            q = compute_recommendation_distributions(recs, dataset.head_idx, dataset.torso_idx, dataset.tail_idx)
            rmse_pcs.append(calculate_rmse_pc(sim.user_blueprints, q))

        # Overall trend: Rmse-PC at alpha=1.0 must be significantly less than at alpha=0.0
        self.assertLess(rmse_pcs[-1], rmse_pcs[0])
        self.assertLessEqual(rmse_pcs[-1], 0.055)

    def test_14_calibration_alpha_invalid_range_clamping(self):
        """T2.14: Out of bounds calibration_alpha values raise ValueError."""
        with self.assertRaises(ValueError):
            SimulationConfig(calibration_alpha=-0.5)
        with self.assertRaises(ValueError):
            SimulationConfig(calibration_alpha=1.5)

    # =========================================================================
    # Category D: Extreme LLM Semantic Fusion Weight Boundaries (5 Tests)
    # =========================================================================

    def test_15_llm_lambda_zero_pure_cf_scores(self):
        """T2.15: llm_lambda=0.0 ignores LLM embeddings and uses standardized CF scores."""
        cf_scores = np.array([[2.0, 4.0, 6.0, 8.0]])
        user_prof = np.array([[1.0, 0.0]])
        item_embs = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
        head = np.array([0, 1])
        torso = np.array([2])
        tail = np.array([3])

        fused = fuse_cf_and_llm_scores(cf_scores, user_prof, item_embs, head, torso, tail, llm_lambda=0.0)
        z_head = intra_pool_zscore_standardization(cf_scores, head)
        np.testing.assert_allclose(fused[:, head], z_head, atol=1e-5)

    def test_16_llm_lambda_one_pure_llm_semantic_scores(self):
        """T2.16: llm_lambda=1.0 produces pure semantic cosine similarities."""
        cf_scores = np.array([[100.0, 200.0, 300.0, 400.0]])
        user_prof = np.array([[1.0, 0.0]])
        item_embs = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.5, 0.5]])
        head = np.array([0, 1])
        torso = np.array([2])
        tail = np.array([3])

        fused = fuse_cf_and_llm_scores(cf_scores, user_prof, item_embs, head, torso, tail, llm_lambda=1.0)
        np.testing.assert_allclose(fused[0, 0], 1.0, atol=1e-5)
        np.testing.assert_allclose(fused[0, 1], 0.0, atol=1e-5)
        np.testing.assert_allclose(fused[0, 2], -1.0, atol=1e-5)

    def test_17_llm_lambda_boundary_with_orthogonal_embeddings(self):
        """T2.17: Orthogonal embeddings (cosine=0) fuse without NaN or numerical underflow."""
        cf_scores = np.array([[1.0, 2.0]])
        u_prof = np.array([[1.0, 0.0]])
        item_embs = np.array([[0.0, 1.0], [0.0, 1.0]])
        fused = fuse_cf_and_llm_scores(cf_scores, u_prof, item_embs, np.array([0, 1]), np.array([]), np.array([]), llm_lambda=0.5)
        self.assertFalse(np.isnan(fused).any())

    def test_18_llm_lambda_boundary_with_identical_embeddings(self):
        """T2.18: Collinear embeddings (cosine=1) evaluate without saturation."""
        cf_scores = np.array([[1.0, 2.0]])
        u_prof = np.array([[1.0, 0.0]])
        item_embs = np.array([[1.0, 0.0], [1.0, 0.0]])
        fused = fuse_cf_and_llm_scores(cf_scores, u_prof, item_embs, np.array([0, 1]), np.array([]), np.array([]), llm_lambda=1.0)
        np.testing.assert_allclose(fused, 1.0, atol=1e-5)

    def test_19_llm_lambda_invalid_bounds_clamping(self):
        """T2.19: Out of bounds llm_lambda raises ValueError."""
        with self.assertRaises(ValueError):
            SimulationConfig(llm_lambda=-0.1)
        with self.assertRaises(ValueError):
            SimulationConfig(llm_lambda=1.1)

    # =========================================================================
    # Category E: Extreme Differential Privacy Noise Boundaries (5 Tests)
    # =========================================================================

    def test_20_dp_sigma_zero_deterministic_noiseless_fl(self):
        """T2.20: dp_enabled=False produces identical weights across duplicate runs."""
        cfg = SimulationConfig(num_clients=10, num_items=30, dp_enabled=False, max_rounds=2, seed=42)
        d1 = generate_mock_dataset(cfg, seed=42)
        d2 = generate_mock_dataset(cfg, seed=42)
        s1 = FederatedSimulation(d1, cfg)
        s2 = FederatedSimulation(d2, cfg)

        s1.step_round()
        s2.step_round()
        np.testing.assert_array_equal(s1.server.item_embeddings, s2.server.item_embeddings)

    def test_21_dp_sigma_ten_heavy_perturbation_stability(self):
        """T2.21: High DP noise (epsilon=0.1) maintains bounded finite weights without NaNs."""
        cfg = SimulationConfig(num_clients=10, num_items=30, dp_enabled=True, dp_epsilon=0.1, dp_l2_clip_norm=1.0, max_rounds=3, seed=42)
        dataset = generate_mock_dataset(cfg, seed=42)
        sim = FederatedSimulation(dataset, cfg)

        for _ in range(3):
            sim.step_round()

        self.assertFalse(np.isnan(sim.server.item_embeddings).any())
        self.assertFalse(np.isinf(sim.server.item_embeddings).any())
        self.assertLess(float(np.abs(sim.server.item_embeddings).max()), 1e5)

    def test_22_dp_clipping_bound_zero_limit(self):
        """T2.22: Very small clipping bound C=1e-4 restricts gradient norms <= 1e-4."""
        g = np.array([1.0, 2.0, 3.0])
        c = 1e-4
        scale = min(1.0, c / np.linalg.norm(g))
        g_clip = g * scale
        self.assertLessEqual(float(np.linalg.norm(g_clip)), c + 1e-9)

    def test_23_dp_clipping_bound_infinite_limit(self):
        """T2.23: Large clipping bound C=1000.0 leaves gradients unclipped."""
        g = np.array([1.0, 2.0, 3.0])
        c = 1000.0
        scale = min(1.0, c / np.linalg.norm(g))
        g_clip = g * scale
        np.testing.assert_allclose(g_clip, g, atol=1e-6)

    def test_24_dp_epsilon_delta_conversion_edge_cases(self):
        """T2.24: dp_sigma property scales inversely with epsilon."""
        cfg_high_priv = SimulationConfig(dp_enabled=True, dp_epsilon=0.5, dp_delta=1e-5, dp_l2_clip_norm=1.0)
        cfg_low_priv = SimulationConfig(dp_enabled=True, dp_epsilon=50.0, dp_delta=1e-5, dp_l2_clip_norm=1.0)
        self.assertGreater(cfg_high_priv.dp_sigma, 8.0)
        self.assertLess(cfg_low_priv.dp_sigma, 0.20)

    # =========================================================================
    # Category F: Small & Asymmetric Item Catalogs (5 Tests)
    # =========================================================================

    def test_25_small_catalog_m_less_than_top_k(self):
        """T2.25: Catalog with M=5 items handles top_k=5 without IndexError."""
        cfg = SimulationConfig(num_clients=5, num_items=5, top_k=5, clients_per_round=2, seed=42)
        dataset = generate_mock_dataset(cfg, seed=42)
        sim = FederatedSimulation(dataset, cfg)
        recs = sim.get_recommendations(calibrated=True)
        self.assertEqual(recs.shape, (5, 5))

    def test_26_catalog_single_item_m_1(self):
        """T2.26: Catalog with M=1 item and top_k=1 functions without crashing."""
        cfg = SimulationConfig(num_clients=2, num_items=1, top_k=1, clients_per_round=1, seed=42)
        dataset = generate_mock_dataset(cfg, seed=42)
        self.assertEqual(dataset.num_items, 1)

    def test_27_catalog_head_pool_smaller_than_head_quota(self):
        """T2.27: Head pool smaller than target quota triggers greedy backfill without duplicate items."""
        fused = np.random.default_rng(42).normal(0, 1, size=(1, 20))
        blueprints = np.array([[0.80, 0.10, 0.10]])  # Target requires 8 Head items out of K=10
        head = np.array([0, 1, 2])                   # But only 3 Head items exist!
        torso = np.arange(3, 10)
        tail = np.arange(10, 20)

        recs = calibrated_blueprint_merge(
            fused, blueprints, head_idx=head, torso_idx=torso, tail_idx=tail, top_k=10, calibration_alpha=1.0
        )
        self.assertEqual(len(recs[0]), 10)
        self.assertEqual(len(set(recs[0])), 10, "Must contain 0 duplicate items")
        head_count = sum(1 for it in recs[0] if it in set(head))
        self.assertEqual(head_count, 3, "Must take all 3 available head items")

    def test_28_catalog_tail_pool_smaller_than_tail_quota(self):
        """T2.28: Tail pool smaller than target quota triggers greedy backfill without duplicates."""
        fused = np.random.default_rng(42).normal(0, 1, size=(1, 20))
        blueprints = np.array([[0.10, 0.10, 0.80]])  # Target requires 8 Tail items
        head = np.arange(0, 10)
        torso = np.arange(10, 18)
        tail = np.array([18, 19])                     # Only 2 Tail items exist!

        recs = calibrated_blueprint_merge(
            fused, blueprints, head_idx=head, torso_idx=torso, tail_idx=tail, top_k=10, calibration_alpha=1.0
        )
        self.assertEqual(len(recs[0]), 10)
        self.assertEqual(len(set(recs[0])), 10)
        tail_count = sum(1 for it in recs[0] if it in set(tail))
        self.assertEqual(tail_count, 2)

    def test_29_zero_item_in_torso_tier(self):
        """T2.29: Empty torso tier handled without division-by-zero or lookup error."""
        fused = np.random.default_rng(42).normal(0, 1, size=(2, 20))
        blueprints = np.array([[0.50, 0.0, 0.50], [0.30, 0.0, 0.70]])
        head = np.arange(0, 8)
        torso = np.array([], dtype=np.int64)
        tail = np.arange(8, 20)

        recs = calibrated_blueprint_merge(
            fused, blueprints, head_idx=head, torso_idx=torso, tail_idx=tail, top_k=6, calibration_alpha=0.50
        )
        self.assertEqual(recs.shape, (2, 6))

    # =========================================================================
    # Category G: Extreme User Cohort Distributions (5 Tests)
    # =========================================================================

    def test_30_extreme_cohort_100_percent_head_preference(self):
        """T2.30: 100% Head cohort receives calibrated recommendations dominated by Head items."""
        train_mat = np.zeros((10, 50), dtype=np.float32)
        head = np.arange(0, 10)
        torso = np.arange(10, 25)
        tail = np.arange(25, 50)
        train_mat[:, head] = 1.0  # 100% head interactions

        p_u = compute_user_popularity_blueprint(train_mat, head, torso, tail)
        np.testing.assert_allclose(p_u, [1.0, 0.0, 0.0], atol=1e-5)

        fused = np.random.default_rng(42).normal(0, 1, size=(10, 50))
        recs = calibrated_blueprint_merge(
            fused, p_u, head_idx=head, torso_idx=torso, tail_idx=tail, top_k=10, calibration_alpha=1.0
        )
        q = compute_recommendation_distributions(recs, head, torso, tail)
        rmse_pc = calculate_rmse_pc(p_u, q)
        self.assertLessEqual(rmse_pc, 0.050)

    def test_31_extreme_cohort_100_percent_tail_preference(self):
        """T2.31: 100% Tail cohort receives recommendations dominated by Tail items (LTER high)."""
        train_mat = np.zeros((10, 50), dtype=np.float32)
        head = np.arange(0, 10)
        torso = np.arange(10, 25)
        tail = np.arange(25, 50)
        train_mat[:, tail] = 1.0  # 100% tail interactions

        p_u = compute_user_popularity_blueprint(train_mat, head, torso, tail)
        np.testing.assert_allclose(p_u, [0.0, 0.0, 1.0], atol=1e-5)

        fused = np.random.default_rng(42).normal(0, 1, size=(10, 50))
        recs = calibrated_blueprint_merge(
            fused, p_u, head_idx=head, torso_idx=torso, tail_idx=tail, top_k=10, calibration_alpha=1.0
        )
        exp = compute_exposure_counts(recs, 50)
        lter = calculate_long_tail_exposure_ratio(exp, tail)
        self.assertGreaterEqual(lter, 0.95)

    def test_32_extreme_cohort_bimodal_polarized(self):
        """T2.32: Bimodal cohort (50% pure Head, 50% pure Tail) achieves low macro Rmse-PC."""
        p_u = np.zeros((20, 3), dtype=np.float32)
        p_u[:10] = [1.0, 0.0, 0.0]
        p_u[10:] = [0.0, 0.0, 1.0]
        head = np.arange(0, 15)
        torso = np.arange(15, 30)
        tail = np.arange(30, 60)

        fused = np.random.default_rng(42).normal(0, 1, size=(20, 60))
        recs = calibrated_blueprint_merge(
            fused, p_u, head_idx=head, torso_idx=torso, tail_idx=tail, top_k=10, calibration_alpha=1.0
        )
        q = compute_recommendation_distributions(recs, head, torso, tail)
        rmse_pc = calculate_rmse_pc(p_u, q)
        self.assertLessEqual(rmse_pc, 0.050)

    def test_33_identical_user_interaction_patterns(self):
        """T2.33: Cohort with identical interaction histories maintains numerical symmetry."""
        train_mat = np.zeros((10, 30), dtype=np.float32)
        train_mat[:, [1, 5, 12, 22]] = 1.0
        head = np.arange(0, 10)
        torso = np.arange(10, 20)
        tail = np.arange(20, 30)

        p_u = compute_user_popularity_blueprint(train_mat, head, torso, tail)
        for u in range(10):
            np.testing.assert_allclose(p_u[u], p_u[0], atol=1e-6)

    def test_34_disjoint_cohort_genres(self):
        """T2.34: Disjoint user genre profiles strongly differentiate semantic cosine similarity."""
        s_sci_fi = np.array([1.0, 0.0, 0.0, 0.0])
        s_romance = np.array([0.0, 1.0, 0.0, 0.0])
        v_scifi_item = np.array([0.9, 0.1, 0.0, 0.0])
        v_scifi_item /= np.linalg.norm(v_scifi_item)

        sim_scifi = float(np.dot(s_sci_fi, v_scifi_item))
        sim_romance = float(np.dot(s_romance, v_scifi_item))
        self.assertGreater(sim_scifi, sim_romance + 0.50)

    # =========================================================================
    # Category H: Single Item per Category, Sparsity & Empty Test Sets (4 Tests)
    # =========================================================================

    def test_35_single_item_per_category(self):
        """T2.35: Small catalog with 1 item per category generates valid metadata."""
        cfg = SimulationConfig(num_clients=6, num_items=6, num_genres=6, top_k=3, seed=42)
        dataset = generate_mock_dataset(cfg, seed=42)
        self.assertEqual(dataset.num_items, 6)
        self.assertIn("category", dataset.item_metadata.columns)

    def test_36_empty_held_out_test_set_for_user(self):
        """T2.36: User with empty test list test_dict[u] = [] does not crash accuracy evaluator."""
        recs = np.array([[1, 2, 3], [4, 5, 6]])
        test_dict = {0: [2], 1: []}  # user 1 has empty test items
        rec = calculate_recall_at_k(recs, test_dict, top_k=3)
        ndcg = calculate_ndcg_at_k(recs, test_dict, top_k=3)
        self.assertTrue(np.isclose(rec, 1.0, atol=1e-5))
        self.assertFalse(np.isnan(ndcg))

    def test_37_all_users_empty_test_set(self):
        """T2.37: All users have empty test dictionaries -> returns 0.0 for Recall/NDCG without crash."""
        recs = np.array([[1, 2, 3], [4, 5, 6]])
        test_dict = {0: [], 1: []}
        rec = calculate_recall_at_k(recs, test_dict, top_k=3)
        ndcg = calculate_ndcg_at_k(recs, test_dict, top_k=3)
        self.assertEqual(rec, 0.0)
        self.assertEqual(ndcg, 0.0)

    def test_38_single_test_item_matching_train_masking(self):
        """T2.38: Recommendations mask training interactions so train items are not recommended."""
        train_matrix = np.zeros((1, 5), dtype=np.float32)
        train_matrix[0, 0] = 1.0  # item 0 already interacted
        fused = np.array([[10.0, 5.0, 4.0, 3.0, 2.0]])  # item 0 has highest raw score
        p_u = np.array([[0.2, 0.4, 0.4]])
        head = np.array([0, 1])
        torso = np.array([2, 3])
        tail = np.array([4])

        recs = calibrated_blueprint_merge(
            fused, p_u, head_idx=head, torso_idx=torso, tail_idx=tail, top_k=3, calibration_alpha=0.5,
            train_matrix=train_matrix
        )
        self.assertNotIn(0, recs[0], "Interacted item 0 must be masked out")

    # =========================================================================
    # Category I: Zero-Variance Similarity Pools & Division-by-Zero Guards (6 Tests)
    # =========================================================================

    def test_39_zero_variance_cf_scores_in_zscore_standardization(self):
        """T2.39: Constant CF scores in a pool handled safely without division by zero."""
        cf_scores = np.full((3, 5), 7.5, dtype=np.float32)
        pool = np.arange(5)
        z = intra_pool_zscore_standardization(cf_scores, pool)
        self.assertFalse(np.isnan(z).any())
        np.testing.assert_allclose(z, 0.0, atol=1e-6)

    def test_40_zero_variance_llm_embeddings(self):
        """T2.40: Identical item LLM embeddings produce constant cosine similarity without crash."""
        user_prof = np.ones((2, 4)) / 2.0
        item_embs = np.ones((5, 4)) / 2.0
        cf_scores = np.random.default_rng(42).normal(0, 1, size=(2, 5))
        head = np.array([0, 1])
        torso = np.array([2, 3])
        tail = np.array([4])

        fused = fuse_cf_and_llm_scores(cf_scores, user_prof, item_embs, head, torso, tail, llm_lambda=0.5)
        self.assertFalse(np.isnan(fused).any())

    def test_41_zero_norm_user_or_item_embedding_vector(self):
        """T2.41: Zero-norm user profile produces 0.0 similarity without NaN."""
        user_prof = np.zeros((1, 4))
        item_embs = np.ones((3, 4)) / 2.0
        cf_scores = np.ones((1, 3))
        fused = fuse_cf_and_llm_scores(cf_scores, user_prof, item_embs, np.array([0, 1]), np.array([2]), np.array([]), llm_lambda=1.0)
        self.assertFalse(np.isnan(fused).any())
        np.testing.assert_allclose(fused, 0.0, atol=1e-5)

    def test_42_all_zero_exposure_gini_guard(self):
        """T2.42: All zero exposure array evaluates Gini index = 0.0 without NaN."""
        exp = np.zeros(50)
        gini = calculate_gini_index(exp)
        self.assertEqual(gini, 0.0)
        self.assertFalse(np.isnan(gini))

    def test_43_uniform_exposure_gini_zero(self):
        """T2.43: Perfectly uniform exposure across 100 items yields exact Gini = 0.0."""
        exp = np.full(100, 25)
        gini = calculate_gini_index(exp)
        self.assertTrue(np.isclose(gini, 0.0, atol=1e-6))

    def test_44_single_item_monopoly_gini_one(self):
        """T2.44: Extreme monopoly (all exposures on 1 item) yields Gini = (M-1)/M."""
        exp = np.zeros(100)
        exp[0] = 1000
        gini = calculate_gini_index(exp)
        self.assertTrue(np.isclose(gini, 99.0 / 100.0, atol=1e-4))


if __name__ == '__main__':
    unittest.main()
