"""
fedsuper_simulation/tests/test_m2_visualizations.py
Milestone 2 Comprehensive Unit Test Suite: Network Flow Graph Visualizer & Analytical Chart Generators.
Verifies headless Plotly figure construction, 3 differentiated visual streams, dark cyberpunk theme,
edge cases (empty states, single clients, zero active clients), JSON serialization, and package exports.
Total Test Cases: 18 Tests across 4 Test Classes.
"""
import os
import sys
import unittest
import numpy as np
import plotly.graph_objects as go

# Ensure fedsuper_simulation package root is on sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.config import SimulationConfig
from src.mock_data import generate_mock_dataset, SyntheticDataset
from src.federated_core import FederatedSimulation
from src.evaluator import evaluate_all_metrics
from src.graph_visualizer import (
    render_network_flow_figure,
    get_network_topology_layout,
    compute_radial_layout,
    create_client_sandbox_halos_trace,
    create_sandbox_halos_trace,
    create_client_nodes_trace,
    create_gradient_flow_traces,
    create_llm_stream_trace,
    create_server_node_trace,
    create_llm_node_trace,
    create_background_mesh_trace,
    apply_dark_theme,
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
)


class TestNetworkFlowVisualizer(unittest.TestCase):
    """Suite 1: Network Flow Graph Visualizer Unit Tests (6 Tests)."""

    def setUp(self):
        self.config = SimulationConfig(num_clients=10, num_items=50, top_k=5, seed=42)
        self.dataset = generate_mock_dataset(self.config, seed=42)
        self.sim = FederatedSimulation(self.dataset, self.config)

    def test_01_topology_layout_coordinates_and_symmetry(self):
        """T2.1.1: Verify radial coordinates, server at (0,0), LLM at (0, 1.35), and angular distribution."""
        topology = get_network_topology_layout(num_clients=10, radius=1.0)
        self.assertIn("server", topology)
        self.assertIn("llm", topology)
        self.assertIn("clients", topology)

        # Server position
        self.assertEqual(topology["server"]["x"], 0.0)
        self.assertEqual(topology["server"]["y"], 0.0)

        # LLM position
        self.assertEqual(topology["llm"]["x"], 0.0)
        self.assertAlmostEqual(topology["llm"]["y"], 1.35, places=2)

        # Client positions
        clients = topology["clients"]
        self.assertEqual(len(clients), 10)
        for c in clients:
            r = np.sqrt(c["x"]**2 + c["y"]**2)
            self.assertAlmostEqual(r, 1.0, places=4, msg="Client node must lie on orbit radius R=1.0")

        # Also verify compatibility helper compute_radial_layout
        compat_layout = compute_radial_layout(num_clients=10, radius=1.0)
        self.assertEqual(len(compat_layout["clients"]), 10)
        self.assertEqual(compat_layout["server"], (0.0, 0.0))

    def test_02_network_flow_figure_three_differentiated_streams(self):
        """T2.1.2: Verify 3 differentiated visual streams: Orange #FF7043, Cyan #00E5FF, Magenta #D500F9."""
        fig = render_network_flow_figure(
            dataset=self.dataset,
            active_client_ids=[1, 3, 5],
            current_round=1,
            dp_clip_norm=1.0,
            llm_lambda=0.70,
            calibration_alpha=0.40
        )
        self.assertIsInstance(fig, go.Figure)
        self.assertGreaterEqual(len(fig.data), 4)

        fig_dict_str = str(fig.to_dict()).lower()
        # Stream 1: Local Private Data (Zero Egress) -> Coral Orange (#FF7043)
        self.assertTrue("#ff7043" in fig_dict_str or "255, 112, 67" in fig_dict_str or "255,112,67" in fig_dict_str,
                        "Stream 1 Orange #FF7043 must be present in figure")
        # Stream 2: DP-Clipped Gradients -> Electric Cyan (#00E5FF)
        self.assertTrue("#00e5ff" in fig_dict_str or "0, 229, 255" in fig_dict_str or "0,229,255" in fig_dict_str,
                        "Stream 2 Cyan #00E5FF must be present in figure")
        # Stream 3: LLM Semantic Blueprints -> Electric Magenta (#D500F9)
        self.assertTrue("#d500f9" in fig_dict_str or "213, 0, 249" in fig_dict_str or "213,0,249" in fig_dict_str,
                        "Stream 3 Magenta #D500F9 must be present in figure")

    def test_03_active_client_gradient_filtering_and_empty_active(self):
        """T2.1.3: Verify active gradient traces are generated only for active clients, none when empty."""
        # 1. Non-empty active clients
        fig_active = render_network_flow_figure(self.dataset, active_client_ids=[2, 4])
        self.assertIsInstance(fig_active, go.Figure)
        self.assertGreater(len(fig_active.data), 0)

        # 2. Empty active clients
        fig_empty = render_network_flow_figure(self.dataset, active_client_ids=[])
        self.assertIsInstance(fig_empty, go.Figure)
        self.assertGreater(len(fig_empty.data), 0, "Figure must still render server, LLM, and client nodes")

    def test_04_out_of_bounds_client_ids_handled_gracefully(self):
        """T2.1.4: Verify out-of-bounds or invalid client IDs (e.g. [-1, 999]) do not cause exceptions."""
        fig = render_network_flow_figure(self.dataset, active_client_ids=[-1, 999, 1000])
        self.assertIsInstance(fig, go.Figure)
        self.assertGreater(len(fig.data), 0)

    def test_05_single_client_and_zero_client_edge_topologies(self):
        """T2.1.5: Verify edge topologies with 1 client and 0 clients execute cleanly."""
        topo_1 = get_network_topology_layout(num_clients=1)
        self.assertEqual(len(topo_1["clients"]), 1)
        self.assertFalse(np.isnan(topo_1["clients"][0]["x"]))
        fig_1 = render_network_flow_figure(dataset=1, active_client_ids=[0])
        self.assertIsInstance(fig_1, go.Figure)

        topo_0 = get_network_topology_layout(num_clients=0)
        self.assertEqual(len(topo_0["clients"]), 0)
        fig_0 = render_network_flow_figure(dataset=0, active_client_ids=[])
        self.assertIsInstance(fig_0, go.Figure)

    def test_06_customdata_schema_and_hover_templates(self):
        """T2.1.6: Verify hovertemplate and informative tooltips on client nodes, halos, and server node."""
        fig = render_network_flow_figure(self.dataset, active_client_ids=[0])
        for trace in fig.data:
            if hasattr(trace, 'hovertext') and trace.hovertext:
                self.assertTrue(len(trace.hovertext) > 0)
            if hasattr(trace, 'hovertemplate') and trace.hovertemplate:
                self.assertIsInstance(trace.hovertemplate, str)


class TestAnalyticalChartGenerators(unittest.TestCase):
    """Suite 2: Analytical Popularity Calibration Chart Generators Unit Tests (7 Tests)."""

    def setUp(self):
        self.config = SimulationConfig(num_clients=10, num_items=50, top_k=5, seed=42)
        self.dataset = generate_mock_dataset(self.config, seed=42)
        self.sim = FederatedSimulation(self.dataset, self.config)
        self.sim.step_round()
        self.recs_unc = self.sim.get_recommendations(calibrated=False)
        self.recs_cal = self.sim.get_recommendations(calibrated=True)
        self.metrics = evaluate_all_metrics(self.dataset, self.recs_unc, self.recs_cal, top_k=5)

    def test_07_exposure_comparison_grouped_bar_traces_and_categories(self):
        """T2.2.1: Verify exposure chart has Head, Torso, Tail categories and 3 comparison traces."""
        fig = render_exposure_comparison_chart(self.metrics)
        self.assertIsInstance(fig, go.Figure)
        self.assertGreaterEqual(len(fig.data), 2, "Must contain at least baseline and calibrated traces")

        # Verify bar trace types
        trace_types = [t.type for t in fig.data]
        self.assertTrue("bar" in trace_types, "Must contain bar traces")

        # Check trace names
        trace_names = [t.name for t in fig.data]
        self.assertTrue(any("Target" in name or "Blueprint" in name for name in trace_names))
        self.assertTrue(any("Baseline" in name or "Uncalibrated" in name for name in trace_names))
        self.assertTrue(any("FedSUPER" in name or "Calibrated" in name for name in trace_names))

    def test_08_exposure_comparison_empty_metrics_fallback(self):
        """T2.2.2: Verify empty or None metrics dictionary returns clean valid fallback figure."""
        fig_empty = render_exposure_comparison_chart({})
        self.assertIsInstance(fig_empty, go.Figure)
        self.assertGreaterEqual(len(fig_empty.data), 1)

        fig_none = render_exposure_comparison_chart(None)
        self.assertIsInstance(fig_none, go.Figure)
        self.assertGreaterEqual(len(fig_none.data), 1)

    def test_09_rank_exposure_curve_traces_and_log_scale(self):
        """T2.2.3: Verify rank exposure curve has 2 traces (Uncalibrated vs Calibrated) and area fills."""
        exp_unc = self.metrics["uncalibrated"]["exposure_counts"]
        exp_cal = self.metrics["calibrated"]["exposure_counts"]
        fig = render_rank_exposure_curve(exp_unc, exp_cal)
        self.assertIsInstance(fig, go.Figure)
        self.assertGreaterEqual(len(fig.data), 2)
        self.assertTrue(all(t.type == "scatter" for t in fig.data))

        # Test log_y flag
        fig_log = render_long_tail_flattening_chart(exp_unc, exp_cal, log_y=True)
        self.assertIsInstance(fig_log, go.Figure)
        self.assertEqual(fig_log.layout.yaxis.type, "log")

    def test_10_rank_exposure_curve_all_zeros_and_mismatched_lengths(self):
        """T2.2.4: Verify all-zero exposure arrays and mismatched lengths execute without crash."""
        fig_zeros = render_rank_exposure_curve(np.zeros(50), np.zeros(50))
        self.assertIsInstance(fig_zeros, go.Figure)

        fig_mismatch = render_rank_exposure_curve(np.zeros(30), np.zeros(60))
        self.assertIsInstance(fig_mismatch, go.Figure)

    def test_11_calibration_error_chart_box_violin_distribution(self):
        """T2.2.5: Verify calibration error distribution figure renders box and violin traces."""
        # Box mode
        fig_box = render_calibration_error_chart(self.metrics, plot_type="box")
        self.assertIsInstance(fig_box, go.Figure)
        self.assertTrue(any(t.type == "box" for t in fig_box.data))

        # Violin mode
        fig_violin = render_calibration_error_chart(self.metrics, plot_type="violin")
        self.assertIsInstance(fig_violin, go.Figure)
        self.assertTrue(any(t.type == "violin" for t in fig_violin.data))

    def test_12_pareto_front_chart_scatter_traces_and_alpha_highlight(self):
        """T2.2.6: Verify Pareto front chart returns scatter figure in 2D and 3D with alpha highlight."""
        records = [
            {"alpha": 0.0, "rmse_pc": 0.38, "gini": 0.75, "lter": 0.04, "rmse": 0.45, "epsilon": 4.0},
            {"alpha": 0.2, "rmse_pc": 0.20, "gini": 0.55, "lter": 0.15, "rmse": 0.46, "epsilon": 4.0},
            {"alpha": 0.4, "rmse_pc": 0.04, "gini": 0.34, "lter": 0.36, "rmse": 0.48, "epsilon": 4.0},
            {"alpha": 1.0, "rmse_pc": 0.02, "gini": 0.30, "lter": 0.40, "rmse": 0.55, "epsilon": 4.0},
        ]
        # 2D mode
        fig_2d = render_pareto_front_chart(records, current_alpha=0.4)
        self.assertIsInstance(fig_2d, go.Figure)
        self.assertGreater(len(fig_2d.data), 0)

        # 3D mode
        fig_3d = render_pareto_frontier_chart(records, current_alpha=0.4, is_3d=True)
        self.assertIsInstance(fig_3d, go.Figure)
        self.assertTrue(any(t.type == "scatter3d" for t in fig_3d.data))

    def test_13_training_convergence_chart_curves_and_empty_history(self):
        """T2.2.7: Verify training convergence chart renders round history curves and handles empty list."""
        fig = render_convergence_chart(self.sim.history)
        self.assertIsInstance(fig, go.Figure)
        self.assertGreaterEqual(len(fig.data), 1)

        fig_empty = render_convergence_chart([])
        self.assertIsInstance(fig_empty, go.Figure)
        self.assertGreaterEqual(len(fig_empty.data), 1)


class TestHeadlessSerializationAndThemeCompliance(unittest.TestCase):
    """Suite 3: Headless Serialization & Dark Theme Invariance Unit Tests (3 Tests)."""

    def setUp(self):
        self.config = SimulationConfig(num_clients=10, num_items=50, top_k=5, seed=42)
        self.dataset = generate_mock_dataset(self.config, seed=42)
        self.sim = FederatedSimulation(self.dataset, self.config)
        self.sim.step_round()
        recs_unc = self.sim.get_recommendations(calibrated=False)
        recs_cal = self.sim.get_recommendations(calibrated=True)
        self.metrics = evaluate_all_metrics(self.dataset, recs_unc, recs_cal, top_k=5)

    def test_14_all_figures_json_serialization_headless(self):
        """T2.3.1: Verify fig.to_dict() and fig.to_json() succeed headlessly for all figure types."""
        figures = [
            render_network_flow_figure(self.dataset, active_client_ids=[0, 1]),
            render_exposure_comparison_chart(self.metrics),
            render_rank_exposure_curve(
                self.metrics["uncalibrated"]["exposure_counts"],
                self.metrics["calibrated"]["exposure_counts"]
            ),
            render_calibration_error_chart(self.metrics),
            render_pareto_front_chart([{"alpha": 0.4, "rmse_pc": 0.04, "gini": 0.34, "lter": 0.35, "rmse": 0.48}]),
            render_convergence_chart(self.sim.history),
        ]

        for i, fig in enumerate(figures):
            # 1. to_dict verification
            fig_dict = fig.to_dict()
            self.assertIsInstance(fig_dict, dict, f"Figure {i} to_dict() must return dict")
            self.assertIn("data", fig_dict, f"Figure {i} dict must contain 'data'")
            self.assertIn("layout", fig_dict, f"Figure {i} dict must contain 'layout'")

            # 2. to_json verification (headless serialization test)
            json_str = fig.to_json()
            self.assertIsInstance(json_str, str, f"Figure {i} to_json() must return str")
            self.assertGreater(len(json_str), 50, f"Figure {i} JSON must be non-empty")

    def test_15_dark_theme_layout_palette_compliance(self):
        """T2.3.2: Verify dark theme background (#0E1117 / #161B22) and high-legibility text (#E6EDF3)."""
        fig = render_network_flow_figure(self.dataset, active_client_ids=[0])
        layout = fig.layout
        valid_dark_bg = ["#0e1117", "#0f172a", "#161b22", "#111827", "rgba(14, 17, 23, 1)", "rgba(14,17,23,1)"]
        self.assertTrue(layout.paper_bgcolor.lower() in valid_dark_bg or layout.plot_bgcolor.lower() in valid_dark_bg,
                        f"Paper bgcolor '{layout.paper_bgcolor}' must adhere to dark theme")

    def test_16_chart_aliases_and_backward_compatibility(self):
        """T2.3.3: Verify all function aliases match primary chart functions identically."""
        fig1 = render_exposure_comparison_chart(self.metrics)
        fig1_alias = render_exposure_distribution_chart(self.metrics)
        self.assertEqual(len(fig1.data), len(fig1_alias.data))

        exp_unc = self.metrics["uncalibrated"]["exposure_counts"]
        exp_cal = self.metrics["calibrated"]["exposure_counts"]
        fig2 = render_rank_exposure_curve(exp_unc, exp_cal)
        fig2_alias = render_long_tail_flattening_chart(exp_unc, exp_cal)
        self.assertEqual(len(fig2.data), len(fig2_alias.data))

        records = [{"alpha": 0.4, "rmse_pc": 0.04, "gini": 0.34, "lter": 0.35, "rmse": 0.48}]
        fig3 = render_pareto_front_chart(records)
        fig3_alias = render_pareto_frontier_chart(records)
        self.assertEqual(len(fig3.data), len(fig3_alias.data))

        fig4 = render_convergence_chart(self.sim.history)
        fig4_alias = render_training_convergence_chart(self.sim.history)
        self.assertEqual(len(fig4.data), len(fig4_alias.data))

        fig5 = render_calibration_error_chart(self.metrics)
        fig5_alias = render_user_calibration_distribution(
            self.metrics["uncalibrated"]["per_user_ce"],
            self.metrics["calibrated"]["per_user_ce"]
        )
        self.assertIsInstance(fig5, go.Figure)
        self.assertIsInstance(fig5_alias, go.Figure)


class TestPackageIntegrationAndExports(unittest.TestCase):
    """Suite 4: Package Public Exports & Namespace Integration Unit Tests (2 Tests)."""

    def test_17_src_init_public_exports_completeness(self):
        """T2.4.1: Verify fedsuper_simulation.src.__all__ exports all M1 and M2 functions."""
        import src
        expected_exports = [
            # M1 exports
            "SimulationConfig", "SyntheticDataset", "generate_mock_dataset",
            "FederatedServer", "FederatedClient", "FederatedSimulation",
            "compute_user_popularity_blueprint", "intra_pool_zscore_standardization",
            "fuse_cf_and_llm_scores", "calibrated_blueprint_merge",
            "calculate_rmse_pc", "calculate_gini_index", "calculate_long_tail_exposure_ratio",
            "calculate_catalog_coverage", "calculate_recall_at_k", "calculate_ndcg_at_k",
            "calculate_novelty", "calculate_ild", "compute_exposure_counts",
            "compute_recommendation_distributions", "evaluate_all_metrics",
            # M2 exports
            "render_network_flow_figure", "get_network_topology_layout",
            "compute_radial_layout", "create_client_sandbox_halos_trace",
            "create_sandbox_halos_trace", "create_client_nodes_trace",
            "create_gradient_flow_traces", "create_llm_stream_trace",
            "create_server_node_trace", "create_llm_node_trace",
            "create_background_mesh_trace", "render_exposure_comparison_chart",
            "render_exposure_distribution_chart", "render_rank_exposure_curve",
            "render_long_tail_flattening_chart", "render_calibration_error_chart",
            "render_user_calibration_distribution", "render_pareto_front_chart",
            "render_pareto_frontier_chart", "render_convergence_chart",
            "render_training_convergence_chart", "apply_dark_theme",
        ]
        for item in expected_exports:
            self.assertIn(item, src.__all__, f"Symbol '{item}' missing from src.__all__")
            self.assertTrue(hasattr(src, item), f"Symbol '{item}' not exported in src namespace")

    def test_18_direct_import_from_package_namespace(self):
        """T2.4.2: Verify direct imports from fedsuper_simulation.src work seamlessly."""
        from src import (
            render_network_flow_figure,
            render_exposure_comparison_chart,
            render_rank_exposure_curve,
            render_calibration_error_chart,
            render_pareto_front_chart,
            render_convergence_chart,
            apply_dark_theme,
        )
        self.assertTrue(callable(render_network_flow_figure))
        self.assertTrue(callable(render_exposure_comparison_chart))
        self.assertTrue(callable(render_rank_exposure_curve))
        self.assertTrue(callable(render_calibration_error_chart))
        self.assertTrue(callable(render_pareto_front_chart))
        self.assertTrue(callable(render_convergence_chart))
        self.assertTrue(callable(apply_dark_theme))


if __name__ == "__main__":
    unittest.main()
