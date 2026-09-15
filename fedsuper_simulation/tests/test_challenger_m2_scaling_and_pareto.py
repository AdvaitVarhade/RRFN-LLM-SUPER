"""
fedsuper_simulation/tests/test_challenger_m2_scaling_and_pareto.py
Milestone 2 Empirical Challenger Test Suite:
1. End-to-End M1 Simulation Integration across all 7+ Figure Types with Real Trace Data
2. Scaling Stress Testing (K=100/500 Clients, M=10,000 Items Catalog, 500 Pareto Points)
3. Pareto Frontier Geometric Correctness (Non-Dominated Sorting, Dominated Rejection, Invariants)
4. Headless Performance and Serialization Benchmarking
"""

import os
import sys
import time
import unittest
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Ensure fedsuper_simulation package root is on sys.path
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
from src.graph_visualizer import (
    render_network_flow_figure,
    get_network_topology_layout,
    compute_radial_layout,
    create_client_sandbox_halos_trace,
    create_client_nodes_trace,
    create_gradient_flow_traces,
    create_llm_stream_trace,
    create_server_node_trace,
    create_llm_node_trace,
    create_background_mesh_trace,
)
from src.chart_generator import (
    render_exposure_comparison_chart,
    render_exposure_distribution_chart,
    render_rank_exposure_curve,
    render_long_tail_flattening_chart,
    render_calibration_error_chart,
    render_user_calibration_distribution,
    render_pareto_front_chart,
    render_pareto_frontier_chart,
    render_convergence_chart,
    render_training_convergence_chart,
    apply_dark_theme,
)


class TestE2ESimulationIntegration(unittest.TestCase):
    """
    Objective 1: End-to-End Simulation Integration.
    Runs a live 5-round FL simulation and feeds real telemetry into all 7+ figure types.
    """

    @classmethod
    def setUpClass(cls):
        cls.config = SimulationConfig(
            num_clients=20,
            num_items=100,
            embedding_dim=16,
            llm_dim=16,
            top_k=10,
            clients_per_round=6,
            calibration_alpha=0.40,
            llm_lambda=0.70,
            dp_enabled=True,
            dp_epsilon=4.0,
            dp_delta=1e-5,
            dp_l2_clip_norm=1.0,
            max_rounds=5,
            seed=42,
        )
        cls.dataset = generate_mock_dataset(cls.config, seed=42)
        cls.sim = FederatedSimulation(cls.dataset, cls.config)

        # Run 5 federated rounds to produce rich real history
        cls.step_results = []
        for _ in range(5):
            res = cls.sim.step_round()
            cls.step_results.append(res)

        # Compute recommendations and metrics
        cls.recs_uncalib = cls.sim.get_recommendations(calibrated=False)
        cls.recs_calib = cls.sim.get_recommendations(calibrated=True)
        cls.metrics = evaluate_all_metrics(cls.dataset, cls.recs_uncalib, cls.recs_calib, top_k=10)

    def test_01_real_simulation_history_populated(self):
        """Verify real simulation produced non-empty history and valid decreasing loss."""
        self.assertEqual(len(self.sim.history), 5)
        losses = [h["loss"] for h in self.sim.history]
        self.assertTrue(all(np.isfinite(l) and l >= 0 for l in losses))
        self.assertLess(losses[-1], losses[0], "Loss should decrease over 5 rounds")

    def test_02_network_flow_figure_with_real_active_clients(self):
        """Verify render_network_flow_figure populates all 3 streams with real round data."""
        last_active = self.step_results[-1]["active_clients"]
        fig = render_network_flow_figure(
            dataset=self.dataset,
            active_client_ids=last_active,
            current_round=5,
            dp_clip_norm=self.config.dp_l2_clip_norm,
            dp_epsilon=self.config.dp_epsilon,
            dp_sigma=self.config.dp_sigma,
            llm_lambda=self.config.llm_lambda,
            calibration_alpha=self.config.calibration_alpha,
        )
        self.assertIsInstance(fig, go.Figure)
        self.assertGreaterEqual(len(fig.data), 6, "Must contain all layers: mesh, llm beam, pulses, gradients, halos, nodes, server, llm")

        # Verify active gradient traces are non-empty
        cyan_traces = [t for t in fig.data if hasattr(t, "line") and t.line and (t.line.color == "#00E5FF" or "0, 229, 255" in str(t.line.color))]
        self.assertGreaterEqual(len(cyan_traces), 1, "Must contain Cyan active gradient trace")

        # Verify halos contain real user interaction counts
        halo_trace = [t for t in fig.data if "Stream 1" in (t.name or "")][0]
        self.assertEqual(len(halo_trace.x), 20)
        self.assertIn("ZERO EGRESS", halo_trace.hovertext[0])

    def test_03_exposure_distribution_chart_with_real_metrics(self):
        """Verify render_exposure_distribution_chart populates 3 bar traces with real percentages."""
        fig = render_exposure_distribution_chart(self.metrics)
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 3)  # Target, Baseline, FedSUPER
        for trace in fig.data:
            self.assertEqual(trace.type, "bar")
            self.assertEqual(len(trace.x), 3)  # Head, Torso, Tail
            self.assertEqual(len(trace.y), 3)
            # Ensure percentages are valid positive floats
            for y_val in trace.y:
                self.assertGreaterEqual(y_val, 0.0)
                self.assertLessEqual(y_val, 100.0)

    def test_04_long_tail_flattening_chart_with_real_exposures(self):
        """Verify render_long_tail_flattening_chart contains real sorted rank exposures."""
        exp_unc = self.metrics["uncalibrated"]["exposure_counts"]
        exp_cal = self.metrics["calibrated"]["exposure_counts"]
        fig = render_long_tail_flattening_chart(exp_unc, exp_cal)
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 2)  # Baseline, FedSUPER
        for trace in fig.data:
            self.assertEqual(trace.type, "scatter")
            self.assertEqual(len(trace.x), 100)  # M=100 items
            self.assertEqual(len(trace.y), 100)
            # Verify rank ordering monotonicity
            y_arr = np.asarray(trace.y)
            self.assertTrue(np.all(np.diff(y_arr) <= 1e-6), "Rank curve must be monotonically non-increasing")

    def test_05_calibration_error_chart_with_real_per_user_ce(self):
        """Verify render_calibration_error_chart renders box & violin with real user errors."""
        # Box plot
        fig_box = render_calibration_error_chart(self.metrics, plot_type="box")
        self.assertIsInstance(fig_box, go.Figure)
        self.assertEqual(len(fig_box.data), 2)
        self.assertEqual(fig_box.data[0].type, "box")
        self.assertEqual(fig_box.data[1].type, "box")
        self.assertEqual(len(fig_box.data[0].y), 20)  # U=20 users
        self.assertEqual(len(fig_box.data[1].y), 20)

        # Violin plot
        fig_violin = render_calibration_error_chart(self.metrics, plot_type="violin")
        self.assertIsInstance(fig_violin, go.Figure)
        self.assertEqual(len(fig_violin.data), 2)
        self.assertEqual(fig_violin.data[0].type, "violin")
        self.assertEqual(fig_violin.data[1].type, "violin")

        # Verify FedSUPER CE is significantly lower than baseline
        unc_ce = self.metrics["uncalibrated"]["per_user_ce"]
        cal_ce = self.metrics["calibrated"]["per_user_ce"]
        self.assertLess(np.mean(cal_ce), np.mean(unc_ce))

    def test_06_pareto_frontier_chart_with_real_sweep(self):
        """Verify render_pareto_frontier_chart generates 2D and 3D with real sweep data."""
        # Generate real parameter sweep
        user_embs = np.array([c.user_embedding for c in self.sim.clients])
        cf_scores = np.dot(user_embs, self.sim.server.item_embeddings.T)

        pareto_records = []
        for a in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            fused = fuse_cf_and_llm_scores(
                cf_scores, self.dataset.user_llm_profiles, self.dataset.item_llm_embeddings,
                self.dataset.head_idx, self.dataset.torso_idx, self.dataset.tail_idx, llm_lambda=0.70
            )
            recs = calibrated_blueprint_merge(
                fused, self.sim.user_blueprints,
                head_idx=self.dataset.head_idx, torso_idx=self.dataset.torso_idx, tail_idx=self.dataset.tail_idx,
                top_k=10, calibration_alpha=a, train_matrix=self.dataset.train_matrix
            )
            q = compute_recommendation_distributions(recs, self.dataset.head_idx, self.dataset.torso_idx, self.dataset.tail_idx)
            rmse_pc = calculate_rmse_pc(self.sim.user_blueprints, q)
            exp = compute_exposure_counts(recs, self.dataset.num_items)
            gini = calculate_gini_index(exp)
            lter = calculate_long_tail_exposure_ratio(exp, self.dataset.tail_idx)
            pareto_records.append({
                "alpha": a,
                "lambda": 0.70,
                "epsilon": 4.0,
                "rmse": 0.45 + 0.04 * a,
                "rmse_pc": rmse_pc,
                "gini": gini,
                "lter": lter,
            })

        # 2D projection
        fig_2d = render_pareto_frontier_chart(pareto_records, current_alpha=0.40, is_3d=False)
        self.assertIsInstance(fig_2d, go.Figure)
        self.assertGreaterEqual(len(fig_2d.data), 2)  # Points + Pareto Line + Active Marker

        # 3D scatter
        fig_3d = render_pareto_frontier_chart(pareto_records, current_alpha=0.40, is_3d=True)
        self.assertIsInstance(fig_3d, go.Figure)
        self.assertTrue(any(t.type == "scatter3d" for t in fig_3d.data))

    def test_07_training_convergence_chart_with_real_history(self):
        """Verify render_training_convergence_chart renders real dual-axis convergence."""
        fig = render_training_convergence_chart(self.sim.history)
        self.assertIsInstance(fig, go.Figure)
        self.assertGreaterEqual(len(fig.data), 2)  # Loss + Touched %
        self.assertEqual(len(fig.data[0].x), 5)    # 5 rounds
        self.assertEqual(len(fig.data[0].y), 5)


class TestScalingStressHarness(unittest.TestCase):
    """
    Objective 2: Scaling Stress Tests.
    Tests extreme workloads: K=100/500 client networks, M=10,000 items catalog, 500 Pareto points.
    """

    def test_08_scale_k100_clients_network_flow_figure(self):
        """Stress Test 2.1: Large client network K=100 clients in render_network_flow_figure()."""
        cfg_100 = SimulationConfig(num_clients=100, num_items=500, top_k=10, seed=123)
        dataset_100 = generate_mock_dataset(cfg_100, seed=123)
        active_ids = list(range(0, 100, 5))  # 20 active clients

        t0 = time.perf_counter()
        fig = render_network_flow_figure(
            dataset=dataset_100,
            active_client_ids=active_ids,
            current_round=10,
            dp_clip_norm=1.0,
            dp_epsilon=4.0,
            llm_lambda=0.70,
            calibration_alpha=0.40,
        )
        elapsed = time.perf_counter() - t0

        self.assertIsInstance(fig, go.Figure)
        self.assertLess(elapsed, 0.50, f"K=100 figure construction took {elapsed:.4f}s, expected < 0.5s")

        # Verify coordinate geometry: exactly 100 client nodes placed on unit circle
        halos_trace = [t for t in fig.data if "Stream 1" in (t.name or "")][0]
        self.assertEqual(len(halos_trace.x), 100)
        radii = np.sqrt(np.array(halos_trace.x)**2 + np.array(halos_trace.y)**2)
        np.testing.assert_allclose(radii, 1.0, atol=1e-4, err_msg="All 100 client nodes must lie on orbit R=1.0")

        # Headless serialization benchmark
        t_json0 = time.perf_counter()
        json_str = fig.to_json()
        t_json_elapsed = time.perf_counter() - t_json0
        self.assertLess(t_json_elapsed, 0.30, f"K=100 JSON serialization took {t_json_elapsed:.4f}s")
        self.assertGreater(len(json_str), 1000)

    def test_09_scale_k500_clients_extreme_topology(self):
        """Stress Test 2.2: Extreme client network K=500 clients."""
        topology = get_network_topology_layout(num_clients=500, radius=1.0)
        self.assertEqual(len(topology["clients"]), 500)
        xs = np.array([c["x"] for c in topology["clients"]])
        ys = np.array([c["y"] for c in topology["clients"]])
        radii = np.sqrt(xs**2 + ys**2)
        np.testing.assert_allclose(radii, 1.0, atol=1e-4)

        # Test full figure construction with K=500
        t0 = time.perf_counter()
        fig = render_network_flow_figure(dataset=500, active_client_ids=list(range(50)))
        elapsed = time.perf_counter() - t0
        self.assertLess(elapsed, 0.80, f"K=500 figure construction took {elapsed:.4f}s")
        self.assertIsInstance(fig, go.Figure)

    def test_10_scale_m10000_items_catalog_rank_curve(self):
        """Stress Test 2.3: Large catalog rank curves M=10,000 items in render_long_tail_flattening_chart()."""
        m = 10000
        ranks = np.arange(1, m + 1)
        # Power law distribution with steep tail
        exp_unc = 50000.0 / (ranks ** 1.1)
        # FedSUPER flattened distribution
        exp_cal = 5000.0 / (ranks ** 0.4)

        t0 = time.perf_counter()
        fig = render_long_tail_flattening_chart(exp_unc, exp_cal, log_y=True)
        elapsed = time.perf_counter() - t0

        self.assertIsInstance(fig, go.Figure)
        self.assertLess(elapsed, 0.20, f"M=10,000 chart construction took {elapsed:.4f}s, expected < 0.2s")
        self.assertEqual(len(fig.data), 2)
        self.assertEqual(len(fig.data[0].x), 10000)
        self.assertEqual(len(fig.data[1].x), 10000)

        # Verify JSON serialization speed on 10,000 points
        t_json0 = time.perf_counter()
        json_str = fig.to_json()
        t_json_elapsed = time.perf_counter() - t_json0
        self.assertLess(t_json_elapsed, 0.40, f"M=10,000 JSON serialization took {t_json_elapsed:.4f}s")
        self.assertGreater(len(json_str), 10000)

    def test_11_scale_500_points_pareto_frontier_3d_and_2d(self):
        """Stress Test 2.4: Large Pareto frontier (500 points in 3D/2D space) in render_pareto_frontier_chart()."""
        rng = np.random.default_rng(2026)
        n_points = 500
        alphas = rng.uniform(0.0, 1.0, size=n_points)
        epsilons = rng.choice([1.0, 2.0, 4.0, 8.0, 16.0], size=n_points)
        rmses = 0.45 + 0.10 * alphas + 0.20 / np.sqrt(epsilons) + rng.normal(0, 0.02, size=n_points)
        rmse_pcs = 0.40 * (1.0 - alphas)**1.5 + 0.03 + rng.normal(0, 0.01, size=n_points)
        ginis = 0.80 - 0.45 * alphas + rng.normal(0, 0.02, size=n_points)
        lters = 0.05 + 0.35 * alphas + rng.normal(0, 0.02, size=n_points)

        pareto_records = [
            {
                "alpha": float(alphas[i]),
                "lambda": 0.70,
                "epsilon": float(epsilons[i]),
                "rmse": float(rmses[i]),
                "rmse_pc": float(rmse_pcs[i]),
                "gini": float(ginis[i]),
                "lter": float(lters[i]),
            }
            for i in range(n_points)
        ]

        # 3D Stress Test
        t0_3d = time.perf_counter()
        fig_3d = render_pareto_frontier_chart(pareto_records, current_alpha=0.40, current_epsilon=4.0, is_3d=True)
        elapsed_3d = time.perf_counter() - t0_3d
        self.assertLess(elapsed_3d, 0.20, f"500-pt 3D Pareto render took {elapsed_3d:.4f}s")
        self.assertEqual(len(fig_3d.data[0].x), 500)

        # 2D Stress Test
        t0_2d = time.perf_counter()
        fig_2d = render_pareto_frontier_chart(pareto_records, current_alpha=0.40, current_epsilon=4.0, is_3d=False)
        elapsed_2d = time.perf_counter() - t0_2d
        self.assertLess(elapsed_2d, 0.20, f"500-pt 2D Pareto render took {elapsed_2d:.4f}s")
        self.assertGreaterEqual(len(fig_2d.data), 2)


class TestParetoFrontierGeometricCorrectness(unittest.TestCase):
    """
    Objective 3: Pareto Frontier Geometric Correctness.
    Verifies non-dominated sorting logic, monotonicity, rejection of dominated points, and edge cases.
    """

    def test_12_non_dominated_frontier_formal_invariance(self):
        """Verify that every point on the 2D Pareto frontier is strictly non-dominated by any other input point."""
        rng = np.random.default_rng(42)
        n_points = 200
        # Generate random cloud in [0.2, 0.8] x [0.01, 0.50]
        rmses = rng.uniform(0.2, 0.8, size=n_points)
        rmse_pcs = rng.uniform(0.01, 0.50, size=n_points)
        records = [
            {"rmse": float(rmses[i]), "rmse_pc": float(rmse_pcs[i]), "alpha": float(i / n_points), "epsilon": 4.0, "lter": 0.3}
            for i in range(n_points)
        ]

        fig = render_pareto_frontier_chart(records, is_3d=False)

        # Extract the Pareto Frontier line trace
        line_traces = [t for t in fig.data if "Pareto Frontier" in (t.name or "")]
        self.assertGreaterEqual(len(line_traces), 1, "Must contain Pareto Frontier line trace")
        frontier_x = list(line_traces[0].x)
        frontier_y = list(line_traces[0].y)

        self.assertGreater(len(frontier_x), 0)
        self.assertEqual(len(frontier_x), len(frontier_y))

        # Check 1: Monotonicity: as RMSE increases, Rmse-PC must strictly decrease
        for i in range(len(frontier_x) - 1):
            self.assertLess(frontier_x[i], frontier_x[i + 1], "Frontier X (RMSE) must be strictly ascending")
            self.assertGreater(frontier_y[i], frontier_y[i + 1], "Frontier Y (Rmse-PC) must be strictly descending")

        # Check 2: Global Non-Domination: No input point can dominate any point on the frontier
        all_pts = list(zip(rmses, rmse_pcs))
        for fx, fy in zip(frontier_x, frontier_y):
            for px, py in all_pts:
                # px <= fx and py <= fy with at least one strict inequality means (px, py) strictly dominates (fx, fy)
                if px <= fx and py <= fy and (px < fx or py < fy):
                    self.fail(f"Point ({px:.4f}, {py:.4f}) strictly dominates Pareto frontier point ({fx:.4f}, {fy:.4f})")

    def test_13_pareto_adversarial_strictly_decreasing_curve(self):
        """Adversarial Test: When all points are naturally non-dominated, ALL must be on the frontier."""
        records = [
            {"rmse": 0.10, "rmse_pc": 0.50, "alpha": 0.0, "epsilon": 4.0, "lter": 0.1},
            {"rmse": 0.20, "rmse_pc": 0.40, "alpha": 0.2, "epsilon": 4.0, "lter": 0.2},
            {"rmse": 0.30, "rmse_pc": 0.30, "alpha": 0.4, "epsilon": 4.0, "lter": 0.3},
            {"rmse": 0.40, "rmse_pc": 0.20, "alpha": 0.6, "epsilon": 4.0, "lter": 0.4},
            {"rmse": 0.50, "rmse_pc": 0.10, "alpha": 0.8, "epsilon": 4.0, "lter": 0.5},
        ]
        fig = render_pareto_frontier_chart(records, is_3d=False)
        line_trace = [t for t in fig.data if "Pareto Frontier" in (t.name or "")][0]
        self.assertEqual(len(line_trace.x), 5, "All 5 points must be included on the frontier")
        self.assertEqual(list(line_trace.x), [0.10, 0.20, 0.30, 0.40, 0.50])
        self.assertEqual(list(line_trace.y), [0.50, 0.40, 0.30, 0.20, 0.10])

    def test_14_pareto_adversarial_strictly_increasing_curve(self):
        """Adversarial Test: When points are strictly increasing (worse in both), only 1 point is non-dominated."""
        records = [
            {"rmse": 0.10, "rmse_pc": 0.10, "alpha": 0.0, "epsilon": 4.0, "lter": 0.1},
            {"rmse": 0.20, "rmse_pc": 0.20, "alpha": 0.2, "epsilon": 4.0, "lter": 0.2},
            {"rmse": 0.30, "rmse_pc": 0.30, "alpha": 0.4, "epsilon": 4.0, "lter": 0.3},
            {"rmse": 0.40, "rmse_pc": 0.40, "alpha": 0.6, "epsilon": 4.0, "lter": 0.4},
        ]
        fig = render_pareto_frontier_chart(records, is_3d=False)
        line_traces = [t for t in fig.data if "Pareto Frontier" in (t.name or "")]
        # With only 1 non-dominated point, a line trace connecting >= 2 points is omitted
        self.assertEqual(len(line_traces), 0)


class TestPerformanceAndSerialization(unittest.TestCase):
    """
    Objective 4: Headless Performance and Serialization Benchmarking.
    """

    def test_15_full_suite_serialization_benchmark(self):
        """Benchmark: Ensure all figure types serialize to JSON in < 150ms each."""
        config = SimulationConfig(num_clients=20, num_items=100, seed=42)
        dataset = generate_mock_dataset(config, seed=42)
        sim = FederatedSimulation(dataset, config)
        sim.step_round()
        recs_unc = sim.get_recommendations(calibrated=False)
        recs_cal = sim.get_recommendations(calibrated=True)
        metrics = evaluate_all_metrics(dataset, recs_unc, recs_cal, top_k=10)

        figs = [
            ("NetworkFlow", render_network_flow_figure(dataset, [0, 1, 2], current_round=1)),
            ("ExposureBar", render_exposure_distribution_chart(metrics)),
            ("RankCurve", render_long_tail_flattening_chart(metrics["uncalibrated"]["exposure_counts"], metrics["calibrated"]["exposure_counts"])),
            ("CalibBox", render_calibration_error_chart(metrics, plot_type="box")),
            ("CalibViolin", render_calibration_error_chart(metrics, plot_type="violin")),
            ("Pareto2D", render_pareto_frontier_chart([{"rmse": 0.5, "rmse_pc": 0.04, "alpha": 0.4, "epsilon": 4.0, "lter": 0.35}], is_3d=False)),
            ("Pareto3D", render_pareto_frontier_chart([{"rmse": 0.5, "rmse_pc": 0.04, "alpha": 0.4, "epsilon": 4.0, "lter": 0.35}], is_3d=True)),
            ("Convergence", render_training_convergence_chart(sim.history)),
        ]

        for name, fig in figs:
            t0 = time.perf_counter()
            j = fig.to_json()
            elapsed = (time.perf_counter() - t0) * 1000.0  # ms
            self.assertGreater(len(j), 50)
            self.assertLess(elapsed, 150.0, f"Figure '{name}' JSON serialization took {elapsed:.2f}ms, expected < 150ms")


if __name__ == "__main__":
    unittest.main()
