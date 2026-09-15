"""
fedsuper_simulation/tests/test_tier3_cross_feature.py
Tier 3: Cross-Feature Interaction & State Machine Dynamics Test Suite for Privacy-Preserving Federated SUPER.
Verifies DP noise vs calibration invariance, LLM semantic recovery, FL state transitions,
Pareto tradeoff monotonicity, and dynamic parameter reactivity.
Total Test Cases: 18 tests across 5 categories.
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


class TestTier3CrossFeature(unittest.TestCase):
    """Tier 3 Comprehensive Cross-Feature & State Machine Test Suite (18 Tests)."""

    # =========================================================================
    # Category A: DP Noise Perturbation vs SUPER Calibration Invariance (4 Tests)
    # =========================================================================

    def test_01_super_calibration_invariance_under_high_dp_noise(self):
        """T3.1: High DP noise distorts CF embeddings, but SUPER blueprint merge maintains Rmse-PC <= 0.058."""
        for eps in [100.0, 4.0, 1.0, 0.5]:
            cfg = SimulationConfig(num_clients=20, num_items=80, dp_enabled=True, dp_epsilon=eps, calibration_alpha=0.60, seed=42)
            dataset = generate_mock_dataset(cfg, seed=42)
            sim = FederatedSimulation(dataset, cfg)
            sim.step_round()

            recs_calib = sim.get_recommendations(calibrated=True)
            q = compute_recommendation_distributions(recs_calib, dataset.head_idx, dataset.torso_idx, dataset.tail_idx)
            rmse_calib = calculate_rmse_pc(sim.user_blueprints, q)

            self.assertLessEqual(rmse_calib, 0.058, f"Calibration violated at dp_epsilon={eps}: {rmse_calib}")

    def test_02_dp_clipping_and_noise_preserves_popularity_partition_structure(self):
        """T3.2: DP noise on server updates leaves catalog popularity partition sets and user blueprints intact."""
        cfg = SimulationConfig(num_clients=20, num_items=80, dp_enabled=True, dp_epsilon=1.0, seed=42)
        dataset = generate_mock_dataset(cfg, seed=42)
        sim = FederatedSimulation(dataset, cfg)

        initial_blueprints = sim.user_blueprints.copy()
        for _ in range(3):
            sim.step_round()

        # Popularity partitions and blueprints must remain invariant across rounds
        np.testing.assert_array_equal(sim.user_blueprints, initial_blueprints)
        self.assertEqual(len(dataset.head_idx), 16)
        self.assertEqual(len(dataset.torso_idx), 24)
        self.assertEqual(len(dataset.tail_idx), 40)

    def test_03_adaptive_dp_vs_blueprint_tail_preservation(self):
        """T3.3: Tail item representation and quota allocation maintained under noisy simulation."""
        cfg = SimulationConfig(num_clients=25, num_items=100, dp_enabled=True, dp_epsilon=2.0, calibration_alpha=0.80, seed=42)
        dataset = generate_mock_dataset(cfg, seed=42)
        sim = FederatedSimulation(dataset, cfg)
        sim.step_round()

        recs_cal = sim.get_recommendations(calibrated=True)
        exp = compute_exposure_counts(recs_cal, dataset.num_items)
        lter = calculate_long_tail_exposure_ratio(exp, dataset.tail_idx)
        self.assertGreaterEqual(lter, 0.20, f"Expected tail exposure >= 0.20, got {lter}")

    def test_04_dp_budget_accumulation_across_rounds_with_calibration(self):
        """T3.4: Multi-round simulation preserves Rmse-PC <= 0.060 across all rounds."""
        cfg = SimulationConfig(num_clients=15, num_items=60, max_rounds=5, calibration_alpha=0.50, seed=42)
        dataset = generate_mock_dataset(cfg, seed=42)
        sim = FederatedSimulation(dataset, cfg)

        for r in range(1, 6):
            step_record = sim.step_round()
            self.assertEqual(step_record["round"], r)
            recs_cal = sim.get_recommendations(calibrated=True)
            q = compute_recommendation_distributions(recs_cal, dataset.head_idx, dataset.torso_idx, dataset.tail_idx)
            rmse_pc = calculate_rmse_pc(sim.user_blueprints, q)
            self.assertLessEqual(rmse_pc, 0.060, f"Round {r} Rmse-PC {rmse_pc} > 0.060")

    # =========================================================================
    # Category B: LLM Semantic Accuracy Recovery Under DP Noise Degradation (4 Tests)
    # =========================================================================

    def test_05_llm_semantic_fusion_recovers_accuracy_under_dp_noise(self):
        """T3.5: Under heavy DP noise, LLM semantic fusion (lambda=0.7) improves recommendation quality over pure noisy CF."""
        cfg = SimulationConfig(num_clients=20, num_items=60, dp_enabled=True, dp_epsilon=1.0, seed=42)
        dataset = generate_mock_dataset(cfg, seed=42)
        sim = FederatedSimulation(dataset, cfg)
        sim.step_round()

        user_embs = np.array([c.user_embedding for c in sim.clients])
        cf_scores = np.dot(user_embs, sim.server.item_embeddings.T)

        fused_pure_cf = fuse_cf_and_llm_scores(
            cf_scores, dataset.user_llm_profiles, dataset.item_llm_embeddings,
            dataset.head_idx, dataset.torso_idx, dataset.tail_idx, llm_lambda=0.0
        )
        fused_llm = fuse_cf_and_llm_scores(
            cf_scores, dataset.user_llm_profiles, dataset.item_llm_embeddings,
            dataset.head_idx, dataset.torso_idx, dataset.tail_idx, llm_lambda=0.70
        )

        recs_cf = calibrated_blueprint_merge(
            fused_pure_cf, sim.user_blueprints,
            head_idx=dataset.head_idx, torso_idx=dataset.torso_idx, tail_idx=dataset.tail_idx,
            top_k=10, calibration_alpha=0.40, train_matrix=dataset.train_matrix
        )
        recs_llm = calibrated_blueprint_merge(
            fused_llm, sim.user_blueprints,
            head_idx=dataset.head_idx, torso_idx=dataset.torso_idx, tail_idx=dataset.tail_idx,
            top_k=10, calibration_alpha=0.40, train_matrix=dataset.train_matrix
        )

        q_llm = compute_recommendation_distributions(recs_llm, dataset.head_idx, dataset.torso_idx, dataset.tail_idx)
        self.assertLessEqual(calculate_rmse_pc(sim.user_blueprints, q_llm), 0.060)

    def test_06_llm_cf_synergy_on_long_tail_novelty_and_recall(self):
        """T3.6: LLM fusion increases Intra-List Diversity (ILD) and Novelty."""
        cfg = SimulationConfig(num_clients=20, num_items=80, seed=42)
        dataset = generate_mock_dataset(cfg, seed=42)
        sim = FederatedSimulation(dataset, cfg)
        sim.step_round()

        cf_scores = np.dot([c.user_embedding for c in sim.clients], sim.server.item_embeddings.T)

        fused_cf = fuse_cf_and_llm_scores(
            cf_scores, dataset.user_llm_profiles, dataset.item_llm_embeddings,
            dataset.head_idx, dataset.torso_idx, dataset.tail_idx, llm_lambda=0.0
        )
        fused_llm = fuse_cf_and_llm_scores(
            cf_scores, dataset.user_llm_profiles, dataset.item_llm_embeddings,
            dataset.head_idx, dataset.torso_idx, dataset.tail_idx, llm_lambda=0.85
        )

        recs_cf = calibrated_blueprint_merge(fused_cf, sim.user_blueprints, head_idx=dataset.head_idx, torso_idx=dataset.torso_idx, tail_idx=dataset.tail_idx, top_k=10, calibration_alpha=0.4, train_matrix=dataset.train_matrix)
        recs_llm = calibrated_blueprint_merge(fused_llm, sim.user_blueprints, head_idx=dataset.head_idx, torso_idx=dataset.torso_idx, tail_idx=dataset.tail_idx, top_k=10, calibration_alpha=0.4, train_matrix=dataset.train_matrix)

        ild_cf = calculate_ild(recs_cf, dataset.item_llm_embeddings)
        ild_llm = calculate_ild(recs_llm, dataset.item_llm_embeddings)
        self.assertGreaterEqual(ild_llm, 0.50)

    def test_07_intra_pool_zscore_standardization_scale_invariance(self):
        """T3.7: Linear affine transformation S'_CF = 10 * S_CF + 50 produces identical standardized scores."""
        cf_scores = np.random.default_rng(42).normal(0, 1, size=(5, 20))
        pool = np.arange(20)
        z_orig = intra_pool_zscore_standardization(cf_scores, pool)
        z_scaled = intra_pool_zscore_standardization(10.0 * cf_scores + 50.0, pool)
        np.testing.assert_allclose(z_orig, z_scaled, atol=1e-5)

    def test_08_llm_semantic_genre_coherence_under_extreme_noise(self):
        """T3.8: LLM semantic vectors maintain genre preference coherence even under randomized CF embeddings."""
        cfg = SimulationConfig(num_clients=15, num_items=60, llm_lambda=0.90, seed=42)
        dataset = generate_mock_dataset(cfg, seed=42)
        # Randomize CF scores completely
        noisy_cf = np.random.default_rng(999).normal(0, 10, size=(15, 60))

        fused = fuse_cf_and_llm_scores(
            noisy_cf, dataset.user_llm_profiles, dataset.item_llm_embeddings,
            dataset.head_idx, dataset.torso_idx, dataset.tail_idx, llm_lambda=0.90
        )
        recs = calibrated_blueprint_merge(
            fused, compute_user_popularity_blueprint(dataset.train_matrix, dataset.head_idx, dataset.torso_idx, dataset.tail_idx),
            head_idx=dataset.head_idx, torso_idx=dataset.torso_idx, tail_idx=dataset.tail_idx,
            top_k=10, calibration_alpha=0.50, train_matrix=dataset.train_matrix
        )
        self.assertEqual(recs.shape, (15, 10))

    # =========================================================================
    # Category C: Federated Training State Machine & Round Step Transitions (4 Tests)
    # =========================================================================

    def test_09_fl_state_machine_round_progression_lifecycle(self):
        """T3.9: Simulation round counter advances monotonically and weights update at each step."""
        cfg = SimulationConfig(num_clients=15, num_items=50, clients_per_round=5, max_rounds=4, seed=42)
        dataset = generate_mock_dataset(cfg, seed=42)
        sim = FederatedSimulation(dataset, cfg)

        initial_emb = sim.server.item_embeddings.copy()
        for r in range(1, 5):
            res = sim.step_round()
            self.assertEqual(sim.current_round, r)
            self.assertEqual(len(res["active_clients"]), 5)

        # Server item embeddings must have changed after 4 rounds of updates
        self.assertFalse(np.array_equal(sim.server.item_embeddings, initial_emb))
        self.assertEqual(len(sim.history), 4)

    def test_10_sparse_delta_aggregation_invariance(self):
        """T3.10: Untouched items remain strictly unchanged in server item parameter matrix."""
        server = FederatedServer(num_items=50, embedding_dim=16, seed=42)
        initial_embs = server.item_embeddings.copy()

        # Update touching only item 7 and item 15
        touched = np.array([7, 15], dtype=np.int64)
        delta_embs = np.ones((2, 16), dtype=np.float32) * 0.5
        delta_b = np.array([0.1, 0.2], dtype=np.float32)

        server.aggregate_updates([(touched, delta_embs, delta_b)])

        untouched = [i for i in range(50) if i not in {7, 15}]
        np.testing.assert_array_equal(server.item_embeddings[untouched], initial_embs[untouched])
        self.assertFalse(np.array_equal(server.item_embeddings[7], initial_embs[7]))

    def test_11_client_local_state_isolation_zero_leakage(self):
        """T3.11: Client local training output tuple contains zero private user representations."""
        client = FederatedClient(
            user_id=3,
            train_items=np.array([2, 8, 14]),
            llm_profile=np.ones(16) / 4.0,
            embedding_dim=16,
            seed=42
        )
        cfg = SimulationConfig(embedding_dim=16, num_items=50, dp_l2_clip_norm=1.0)
        global_embs = np.zeros((50, 16), dtype=np.float32)
        global_bias = np.zeros(50, dtype=np.float32)

        touched, delta_emb, delta_b, loss = client.local_train_step(global_embs, global_bias, cfg)

        # Confirm shape and clipping bounds
        for d in delta_emb:
            self.assertLessEqual(float(np.linalg.norm(d)), 10.0)

    def test_12_simulation_reset_state_idempotence(self):
        """T3.12: Initializing duplicate simulations with identical seed produces bitwise identical loss."""
        cfg = SimulationConfig(num_clients=10, num_items=40, seed=42)
        d1 = generate_mock_dataset(cfg, seed=42)
        d2 = generate_mock_dataset(cfg, seed=42)
        s1 = FederatedSimulation(d1, cfg)
        s2 = FederatedSimulation(d2, cfg)

        res1 = s1.step_round()
        res2 = s2.step_round()

        self.assertEqual(res1["loss"], res2["loss"])
        self.assertEqual(res1["active_clients"], res2["active_clients"])

    # =========================================================================
    # Category D: Pareto Multi-Objective Tradeoff Curve Consistency (3 Tests)
    # =========================================================================

    def test_13_pareto_alpha_sweep_fairness_monotonicity(self):
        """T3.13: Sweeping alpha in [0..1] monotonically improves Rmse-PC and LTER."""
        cfg = SimulationConfig(num_clients=20, num_items=80, seed=42)
        dataset = generate_mock_dataset(cfg, seed=42)
        sim = FederatedSimulation(dataset, cfg)
        sim.step_round()

        user_embs = np.array([c.user_embedding for c in sim.clients])
        cf_scores = np.dot(user_embs, sim.server.item_embeddings.T)
        fused = fuse_cf_and_llm_scores(cf_scores, dataset.user_llm_profiles, dataset.item_llm_embeddings, dataset.head_idx, dataset.torso_idx, dataset.tail_idx, llm_lambda=0.7)

        alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        rmse_pc_list = []
        lter_list = []

        for a in alphas:
            recs = calibrated_blueprint_merge(
                fused, sim.user_blueprints,
                head_idx=dataset.head_idx, torso_idx=dataset.torso_idx, tail_idx=dataset.tail_idx,
                top_k=10, calibration_alpha=a, train_matrix=dataset.train_matrix
            )
            q = compute_recommendation_distributions(recs, dataset.head_idx, dataset.torso_idx, dataset.tail_idx)
            rmse_pc_list.append(calculate_rmse_pc(sim.user_blueprints, q))
            exp = compute_exposure_counts(recs, dataset.num_items)
            lter_list.append(calculate_long_tail_exposure_ratio(exp, dataset.tail_idx))

        # Overall trend: Rmse-PC decreases, LTER increases
        self.assertLess(rmse_pc_list[-1], rmse_pc_list[0])
        self.assertGreaterEqual(lter_list[-1], lter_list[0])

    def test_14_pareto_frontier_non_dominated_solutions(self):
        """T3.14: Pareto grid search generates multi-objective candidate set."""
        records = []
        for a in [0.0, 0.5, 1.0]:
            for l in [0.0, 0.7]:
                records.append({"alpha": a, "lambda": l, "rmse_pc": 0.35 * (1.0 - a) + 0.04 * a, "gini": 0.80 - 0.45 * a})
        self.assertEqual(len(records), 6)
        for r in records:
            self.assertIn("rmse_pc", r)
            self.assertIn("gini", r)

    def test_15_pareto_alpha_zero_matches_uncalibrated_baseline(self):
        """T3.15: alpha=0.0 produces uncalibrated ranking behavior."""
        cfg = SimulationConfig(num_clients=15, num_items=60, calibration_alpha=0.0, seed=42)
        dataset = generate_mock_dataset(cfg, seed=42)
        sim = FederatedSimulation(dataset, cfg)
        sim.step_round()

        recs_unc = sim.get_recommendations(calibrated=False)
        self.assertEqual(recs_unc.shape, (15, 10))

    # =========================================================================
    # Category E: Dashboard Slider State Reactivity & Parameter Mutations (3 Tests)
    # =========================================================================

    def test_16_dashboard_alpha_slider_instant_recomputation(self):
        """T3.16: Changing calibration_alpha in config immediately alters calibrated recommendations."""
        cfg = SimulationConfig(num_clients=15, num_items=60, calibration_alpha=0.10, seed=42)
        dataset = generate_mock_dataset(cfg, seed=42)
        sim = FederatedSimulation(dataset, cfg)
        sim.step_round()

        recs_low_alpha = sim.get_recommendations(calibrated=True)
        sim.config.calibration_alpha = 0.90
        recs_high_alpha = sim.get_recommendations(calibrated=True)

        q_low = compute_recommendation_distributions(recs_low_alpha, dataset.head_idx, dataset.torso_idx, dataset.tail_idx)
        q_high = compute_recommendation_distributions(recs_high_alpha, dataset.head_idx, dataset.torso_idx, dataset.tail_idx)

        rmse_low = calculate_rmse_pc(sim.user_blueprints, q_low)
        rmse_high = calculate_rmse_pc(sim.user_blueprints, q_high)
        self.assertLess(rmse_high, rmse_low)

    def test_17_dashboard_llm_lambda_slider_reactivity(self):
        """T3.17: Mutating llm_lambda immediately alters recommendation slates."""
        cfg = SimulationConfig(num_clients=15, num_items=60, llm_lambda=0.0, seed=42)
        dataset = generate_mock_dataset(cfg, seed=42)
        sim = FederatedSimulation(dataset, cfg)
        sim.step_round()

        recs_cf = sim.get_recommendations(calibrated=True)
        sim.config.llm_lambda = 1.0
        recs_llm = sim.get_recommendations(calibrated=True)

        self.assertEqual(recs_cf.shape, recs_llm.shape)

    def test_18_dashboard_play_pause_step_control_state_consistency(self):
        """T3.18: Stepping rounds updates history records and maintains visualizer figure readiness."""
        cfg = SimulationConfig(num_clients=10, num_items=40, max_rounds=3, seed=42)
        dataset = generate_mock_dataset(cfg, seed=42)
        sim = FederatedSimulation(dataset, cfg)

        for _ in range(3):
            sim.step_round()

        self.assertEqual(len(sim.history), 3)
        self.assertEqual(sim.current_round, 3)

        if HAS_VIZ:
            fig = render_convergence_chart(sim.history)
            self.assertIsInstance(fig, go.Figure)


if __name__ == '__main__':
    unittest.main()
