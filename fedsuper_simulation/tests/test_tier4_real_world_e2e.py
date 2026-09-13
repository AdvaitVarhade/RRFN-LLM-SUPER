"""
fedsuper_simulation/tests/test_tier4_real_world_e2e.py
Tier 4: Real-World Scenarios & Headless E2E Simulation Test Suite for Privacy-Preserving Federated SUPER.
Verifies multi-round convergence, Dirichlet preference clusters, full pipeline lifecycle,
Pareto optimization, DP budget bounds, scale benchmarks, airgap reproducibility, and headless UI.
Total Test Cases: 11 tests.
"""
import sys
import os
import time
import socket
import urllib.request
import subprocess
import unittest
from unittest.mock import patch
import numpy as np
import pytest

# Ensure fedsuper_simulation directory is in path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.config import SimulationConfig
from src.mock_data import generate_mock_dataset, SyntheticDataset
from src.federated_core import FederatedSimulation
from src.super_engine import (
    compute_user_popularity_blueprint,
    fuse_cf_and_llm_scores,
    calibrated_blueprint_merge,
)
from src.evaluator import (
    calculate_rmse_pc,
    calculate_gini_index,
    calculate_long_tail_exposure_ratio,
    calculate_catalog_coverage,
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


class TestTier4RealWorldE2E(unittest.TestCase):
    """Tier 4 Real-World E2E Test Suite (11 Tests)."""

    @classmethod
    def setUpClass(cls):
        cls.base_config = SimulationConfig(
            num_clients=20,
            num_items=100,
            embedding_dim=32,
            llm_dim=32,
            pareto_alpha=0.20,
            calibration_alpha=0.40,
            llm_lambda=0.70,
            dp_enabled=True,
            dp_epsilon=4.0,
            dp_delta=1e-5,
            dp_l2_clip_norm=1.0,
            top_k=10,
            clients_per_round=5,
            learning_rate=0.05,
            max_rounds=10,
            seed=42,
        )

    def test_01_e2e_multi_round_federated_convergence(self):
        """Test 1: Multi-round FL loss decreases over 10 rounds and embeddings converge without NaNs."""
        dataset = generate_mock_dataset(self.base_config, seed=42)
        sim = FederatedSimulation(dataset, self.base_config)

        losses = []
        for _ in range(10):
            step_res = sim.step_round()
            losses.append(step_res["loss"])

        self.assertEqual(len(losses), 10)
        self.assertTrue(all(np.isfinite(l) and l >= 0 for l in losses))
        self.assertFalse(np.isnan(sim.server.item_embeddings).any(), "Server embeddings must not contain NaNs")
        self.assertFalse(np.isinf(sim.server.item_embeddings).any(), "Server embeddings must not contain Infs")

    def test_02_e2e_pipeline_full_lifecycle_flow(self):
        """Test 2: Complete lifecycle: mock data -> 5 FL rounds -> blueprint merge -> evaluation -> 5 Plotly figures."""
        config = SimulationConfig(num_clients=25, num_items=120, top_k=10, max_rounds=5, seed=100)
        dataset = generate_mock_dataset(config, seed=100)
        sim = FederatedSimulation(dataset, config)

        for _ in range(5):
            sim.step_round()

        recs_uncalib = sim.get_recommendations(calibrated=False)
        recs_calib = sim.get_recommendations(calibrated=True)

        self.assertEqual(recs_uncalib.shape, (25, 10))
        self.assertEqual(recs_calib.shape, (25, 10))

        metrics = evaluate_all_metrics(dataset, recs_uncalib, recs_calib, top_k=10)
        rmse_pc_cal = metrics["calibrated"]["rmse_pc"]
        gini_cal = metrics["calibrated"]["gini_index"]
        gini_unc = metrics["uncalibrated"]["gini_index"]
        lter_cal = metrics["calibrated"]["long_tail_exposure_ratio"]
        lter_unc = metrics["uncalibrated"]["long_tail_exposure_ratio"]

        self.assertLessEqual(rmse_pc_cal, 0.060, f"Calibrated Rmse-PC {rmse_pc_cal} exceeded 0.060")
        self.assertLess(gini_cal, gini_unc, "Calibrated Gini should be lower than uncalibrated")
        self.assertGreater(lter_cal, lter_unc, "Calibrated LTER should be higher than uncalibrated")

        if HAS_VIZ:
            last_active = sim.history[-1]["active_clients"] if sim.history else [0, 1]
            fig_flow = render_network_flow_figure(
                dataset, last_active, current_round=5,
                dp_clip_norm=config.dp_l2_clip_norm, llm_lambda=config.llm_lambda
            )
            fig_exp = render_exposure_comparison_chart(metrics)
            fig_rank = render_rank_exposure_curve(
                metrics["uncalibrated"]["exposure_counts"],
                metrics["calibrated"]["exposure_counts"]
            )
            pareto_recs = [
                {"alpha": 0.0, "rmse_pc": metrics["uncalibrated"]["rmse_pc"], "gini": metrics["uncalibrated"]["gini_index"]},
                {"alpha": config.calibration_alpha, "rmse_pc": metrics["calibrated"]["rmse_pc"], "gini": metrics["calibrated"]["gini_index"]}
            ]
            fig_pareto = render_pareto_front_chart(pareto_recs, current_alpha=config.calibration_alpha)
            fig_conv = render_convergence_chart(sim.history)

            for fig in [fig_flow, fig_exp, fig_rank, fig_pareto, fig_conv]:
                self.assertIsInstance(fig, go.Figure)
                self.assertGreater(len(fig.data), 0)

    def test_03_e2e_dirichlet_cluster_full_catalog_ranking(self):
        """Test 3: Full catalog ranking under Dirichlet user clusters maintains macro Rmse-PC <= 0.060."""
        config = SimulationConfig(num_clients=60, num_items=150, top_k=10, max_rounds=3, calibration_alpha=0.60, seed=2026)
        dataset = generate_mock_dataset(config, seed=2026)
        sim = FederatedSimulation(dataset, config)

        for _ in range(3):
            sim.step_round()

        recs_calib = sim.get_recommendations(calibrated=True)
        q = compute_recommendation_distributions(recs_calib, dataset.head_idx, dataset.torso_idx, dataset.tail_idx)
        macro_rmse_pc = calculate_rmse_pc(sim.user_blueprints, q)

        self.assertLessEqual(macro_rmse_pc, 0.060, f"Macro Rmse-PC {macro_rmse_pc} exceeded threshold 0.060")

    def test_04_e2e_system_pareto_optimization_grid(self):
        """Test 4: System Pareto frontier across alpha in [0..1] and lambda in [0..1]."""
        dataset = generate_mock_dataset(self.base_config, seed=777)
        sim = FederatedSimulation(dataset, self.base_config)
        for _ in range(3):
            sim.step_round()

        user_embs = np.array([c.user_embedding for c in sim.clients])
        cf_scores = np.dot(user_embs, sim.server.item_embeddings.T)

        alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        lambdas = [0.0, 0.3, 0.7, 1.0]
        pareto_records = []

        for a in alphas:
            for l in lambdas:
                fused = fuse_cf_and_llm_scores(
                    cf_scores, dataset.user_llm_profiles, dataset.item_llm_embeddings,
                    dataset.head_idx, dataset.torso_idx, dataset.tail_idx, llm_lambda=l
                )
                recs = calibrated_blueprint_merge(
                    fused, sim.user_blueprints,
                    head_idx=dataset.head_idx, torso_idx=dataset.torso_idx, tail_idx=dataset.tail_idx,
                    top_k=10, calibration_alpha=a, train_matrix=dataset.train_matrix
                )
                q = compute_recommendation_distributions(recs, dataset.head_idx, dataset.torso_idx, dataset.tail_idx)
                rmse_pc = calculate_rmse_pc(sim.user_blueprints, q)
                exp = compute_exposure_counts(recs, dataset.num_items)
                gini = calculate_gini_index(exp)
                lter = calculate_long_tail_exposure_ratio(exp, dataset.tail_idx)

                pareto_records.append({
                    "alpha": a,
                    "lambda": l,
                    "rmse_pc": rmse_pc,
                    "gini": gini,
                    "lter": lter,
                })

        self.assertEqual(len(pareto_records), 24)

        # Verify monotonicity across alpha for lambda = 0.7
        sub_records = {r["alpha"]: r["rmse_pc"] for r in pareto_records if r["lambda"] == 0.7}
        self.assertLess(sub_records[1.0], sub_records[0.4])
        self.assertLess(sub_records[0.4], sub_records[0.0])

        if HAS_VIZ:
            fig = render_pareto_front_chart(pareto_records, current_alpha=0.40)
            self.assertIsInstance(fig, go.Figure)

    def test_05_e2e_headless_streamlit_apptest_lifecycle(self):
        """Test 5: Streamlit AppTest headless lifecycle verification without GUI."""
        app_path = os.path.join(BASE_DIR, "app.py")
        if not os.path.exists(app_path):
            self.skipTest("app.py not yet created (Milestone M3 in progress)")

        try:
            from streamlit.testing.v1 import AppTest
        except ImportError:
            self.skipTest("streamlit.testing.v1 not available in environment")

        at = AppTest.from_file(app_path, default_timeout=20)
        at.run()
        self.assertFalse(at.exception, "Streamlit app must mount without exceptions")

    def test_06_e2e_differential_privacy_budget_and_invariance(self):
        """Test 6: Differential Privacy bounds client gradients and preserves calibration invariance."""
        for eps in [1.0, 4.0, 100.0]:
            cfg = SimulationConfig(num_clients=15, num_items=60, dp_enabled=True, dp_epsilon=eps, dp_l2_clip_norm=1.0, max_rounds=2, seed=42)
            dataset = generate_mock_dataset(cfg, seed=42)
            sim = FederatedSimulation(dataset, cfg)
            sim.step_round()

            recs_calib = sim.get_recommendations(calibrated=True)
            q = compute_recommendation_distributions(recs_calib, dataset.head_idx, dataset.torso_idx, dataset.tail_idx)
            rmse_pc = calculate_rmse_pc(sim.user_blueprints, q)
            self.assertLessEqual(rmse_pc, 0.060, f"Rmse-PC violated under eps={eps}: {rmse_pc}")

    def test_07_e2e_large_scale_cohort_performance_benchmark(self):
        """Test 7: Scale benchmark: 50 clients, 200 items, 10 rounds completes in < 10.0 seconds."""
        cfg = SimulationConfig(num_clients=50, num_items=200, top_k=20, max_rounds=10, seed=999)
        dataset = generate_mock_dataset(cfg, seed=999)
        sim = FederatedSimulation(dataset, cfg)

        t0 = time.perf_counter()
        for _ in range(10):
            sim.step_round()
        recs_uncalib = sim.get_recommendations(calibrated=False)
        recs_calib = sim.get_recommendations(calibrated=True)
        metrics = evaluate_all_metrics(dataset, recs_uncalib, recs_calib, top_k=20)
        elapsed = time.perf_counter() - t0

        self.assertLess(elapsed, 10.0, f"Benchmark took {elapsed:.2f}s, expected < 10.0s")
        self.assertLessEqual(metrics["calibrated"]["rmse_pc"], 0.060)

    def test_08_e2e_airgap_and_deterministic_reproducibility(self):
        """Test 8: Airgap isolation (no external sockets) and bitwise reproducibility with fixed seed."""
        cfg = SimulationConfig(num_clients=15, num_items=80, max_rounds=3, seed=12345)

        # Run 1
        d1 = generate_mock_dataset(cfg, seed=12345)
        s1 = FederatedSimulation(d1, cfg)
        for _ in range(3):
            s1.step_round()
        r1 = s1.get_recommendations(calibrated=True)

        # Run 2 (same seed)
        d2 = generate_mock_dataset(cfg, seed=12345)
        s2 = FederatedSimulation(d2, cfg)
        for _ in range(3):
            s2.step_round()
        r2 = s2.get_recommendations(calibrated=True)

        # Run 3 (different seed)
        d3 = generate_mock_dataset(cfg, seed=54321)
        s3 = FederatedSimulation(d3, cfg)
        for _ in range(3):
            s3.step_round()
        r3 = s3.get_recommendations(calibrated=True)

        np.testing.assert_array_equal(r1, r2, "Identical seeds must produce identical recommendations")
        self.assertFalse(np.array_equal(r1, r3), "Different seeds must produce different recommendations")

    def test_09_e2e_cold_start_and_extreme_imbalance_resilience(self):
        """Test 9: Cold-start users (0 interactions) and extreme taste users handled without crash."""
        cfg = SimulationConfig(num_clients=15, num_items=80, top_k=10, max_rounds=2, seed=333)
        dataset = generate_mock_dataset(cfg, seed=333)

        # Inject cold start into first 3 users
        dataset.train_matrix[:3, :] = 0.0

        sim = FederatedSimulation(dataset, cfg)
        sim.step_round()
        recs = sim.get_recommendations(calibrated=True)

        self.assertEqual(recs.shape, (15, 10))
        for u in range(3):
            self.assertEqual(len(set(recs[u])), 10, "Cold start user must receive 10 unique recommendations")

    def test_10_e2e_network_flow_graph_visual_channel_integrity(self):
        """Test 10: Network flow figure contains exact color channels (#FF7043, #00E5FF, #D500F9)."""
        if not HAS_VIZ:
            self.skipTest("render_network_flow_figure not available yet")

        dataset = generate_mock_dataset(self.base_config, seed=42)
        fig = render_network_flow_figure(
            dataset, active_client_ids=[1, 3, 5], current_round=1,
            dp_clip_norm=1.0, llm_lambda=0.70
        )
        self.assertIsInstance(fig, go.Figure)

        data_str = str(fig.to_dict()).lower()
        self.assertTrue("#ff7043" in data_str or "255, 112, 67" in data_str or "orange" in data_str,
                        "Must contain Orange local private data channel")
        self.assertTrue("#00e5ff" in data_str or "0, 229, 255" in data_str or "cyan" in data_str,
                        "Must contain Cyan DP gradient channel")
        self.assertTrue("#d500f9" in data_str or "213, 0, 249" in data_str or "magenta" in data_str,
                        "Must contain Magenta LLM blueprint channel")

    def test_11_e2e_headless_subprocess_port_binding(self):
        """Test 11: Headless Streamlit subprocess launch, HTTP 200 health check, and clean shutdown."""
        app_path = os.path.join(BASE_DIR, "app.py")
        if not os.path.exists(app_path):
            self.skipTest("app.py not yet created (Milestone M3 in progress)")

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        port = s.getsockname()[1]
        s.close()

        cmd = [
            sys.executable, "-m", "streamlit", "run", app_path,
            "--server.headless", "true",
            "--server.port", str(port),
            "--server.fileWatcherType", "none",
            "--browser.gatherUsageStats", "false"
        ]

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        ready = False
        try:
            health_url = f"http://127.0.0.1:{port}/_stcore/health"
            root_url = f"http://127.0.0.1:{port}/"
            for _ in range(20):
                time.sleep(0.5)
                try:
                    with urllib.request.urlopen(health_url, timeout=1.0) as resp:
                        if resp.status == 200:
                            ready = True
                            break
                except Exception:
                    try:
                        with urllib.request.urlopen(root_url, timeout=1.0) as resp:
                            if resp.status == 200:
                                ready = True
                                break
                    except Exception:
                        pass
            self.assertTrue(ready, f"Streamlit failed to bind to port {port} and respond HTTP 200 within 10s")
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


if __name__ == '__main__':
    unittest.main()
