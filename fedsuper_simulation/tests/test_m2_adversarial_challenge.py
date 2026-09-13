"""
fedsuper_simulation/tests/test_m2_adversarial_challenge.py
Milestone 2 Adversarial Stress Test and Invariant Challenge Suite for Pytest.
"""

import os
import sys
import json
import unittest
import numpy as np
import plotly.graph_objects as go

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.config import SimulationConfig
from src.mock_data import generate_mock_dataset
from src.federated_core import FederatedSimulation
from src.evaluator import evaluate_all_metrics
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
    COLOR_STREAM_1_ORANGE,
    COLOR_STREAM_2_CYAN,
    COLOR_STREAM_3_MAGENTA,
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


class TestMilestone2AdversarialInputs(unittest.TestCase):
    """Adversarial Inputs and Edge Cases."""

    def setUp(self):
        self.config = SimulationConfig(num_clients=12, num_items=50, top_k=5, seed=42)
        self.dataset = generate_mock_dataset(self.config, seed=42)
        self.sim = FederatedSimulation(self.dataset, self.config)
        self.sim.step_round()
        recs_unc = self.sim.get_recommendations(calibrated=False)
        recs_cal = self.sim.get_recommendations(calibrated=True)
        self.metrics = evaluate_all_metrics(self.dataset, recs_unc, recs_cal, top_k=5)

    def test_adv_01_none_and_empty_inputs_across_all_generators(self):
        """Verify zero unhandled exceptions on None and empty inputs."""
        fig_net = render_network_flow_figure(None, active_client_ids=None)
        self.assertIsInstance(fig_net, go.Figure)

        fig_exp = render_exposure_distribution_chart(None)
        self.assertIsInstance(fig_exp, go.Figure)

        fig_exp_e = render_exposure_distribution_chart({})
        self.assertIsInstance(fig_exp_e, go.Figure)

        fig_rank = render_long_tail_flattening_chart(None, None)
        self.assertIsInstance(fig_rank, go.Figure)

        fig_ce = render_calibration_error_chart(None)
        self.assertIsInstance(fig_ce, go.Figure)

        fig_par = render_pareto_frontier_chart(None)
        self.assertIsInstance(fig_par, go.Figure)

        fig_conv = render_training_convergence_chart(None)
        self.assertIsInstance(fig_conv, go.Figure)

    def test_adv_02_negative_zero_and_extreme_hyperparameters(self):
        """Verify DP epsilon <= 0, lambda < 0 or > 1, clip norm = 0."""
        fig1 = render_network_flow_figure(self.dataset, dp_epsilon=0.0, dp_sigma=10.0, dp_clip_norm=0.0)
        self.assertIsInstance(fig1, go.Figure)

        fig2 = render_network_flow_figure(self.dataset, dp_epsilon=-5.0, llm_lambda=-1.0)
        self.assertIsInstance(fig2, go.Figure)

        fig3 = render_pareto_frontier_chart(None, current_alpha=-0.5, current_epsilon=-1.0)
        self.assertIsInstance(fig3, go.Figure)

    def test_adv_03_out_of_bounds_client_ids_and_mismatched_arrays(self):
        """Verify negative client IDs, huge client IDs, and length mismatches."""
        fig_oob = render_network_flow_figure(self.dataset, active_client_ids=[-99, 100, 500])
        self.assertIsInstance(fig_oob, go.Figure)

        fig_mismatch = render_long_tail_flattening_chart(np.array([1, 2]), np.zeros(200))
        self.assertIsInstance(fig_mismatch, go.Figure)

    def test_adv_04_single_client_and_zero_client_topologies(self):
        """Verify num_clients=0 and num_clients=1."""
        topo0 = get_network_topology_layout(0)
        self.assertEqual(len(topo0["clients"]), 0)
        fig0 = render_network_flow_figure(0, active_client_ids=[])
        self.assertIsInstance(fig0, go.Figure)

        topo1 = get_network_topology_layout(1)
        self.assertEqual(len(topo1["clients"]), 1)
        fig1 = render_network_flow_figure(1, active_client_ids=[0])
        self.assertIsInstance(fig1, go.Figure)


class TestMilestone2InvariantsAndColors(unittest.TestCase):
    """Invariants: go.Figure types, JSON serialization, color hex codes, dark theme."""

    def setUp(self):
        self.config = SimulationConfig(num_clients=10, num_items=40, top_k=5, seed=42)
        self.dataset = generate_mock_dataset(self.config, seed=42)
        self.sim = FederatedSimulation(self.dataset, self.config)
        self.sim.step_round()
        recs_unc = self.sim.get_recommendations(calibrated=False)
        recs_cal = self.sim.get_recommendations(calibrated=True)
        self.metrics = evaluate_all_metrics(self.dataset, recs_unc, recs_cal, top_k=5)

    def test_inv_01_all_returned_objects_are_plotly_figures_and_serialize_json(self):
        """Verify all returned objects are go.Figure and json.dumps(fig.to_dict()) works."""
        figs = [
            render_network_flow_figure(self.dataset, active_client_ids=[0, 1]),
            render_exposure_comparison_chart(self.metrics),
            render_exposure_distribution_chart(self.metrics),
            render_rank_exposure_curve(self.metrics["uncalibrated"]["exposure_counts"], self.metrics["calibrated"]["exposure_counts"]),
            render_long_tail_flattening_chart(self.metrics["uncalibrated"]["exposure_counts"], self.metrics["calibrated"]["exposure_counts"]),
            render_calibration_error_chart(self.metrics),
            render_user_calibration_distribution(self.metrics["uncalibrated"]["per_user_ce"], self.metrics["calibrated"]["per_user_ce"]),
            render_pareto_front_chart(None),
            render_pareto_frontier_chart(None, is_3d=True),
            render_convergence_chart(self.sim.history),
            render_training_convergence_chart(self.sim.history),
        ]
        for f in figs:
            self.assertIsInstance(f, go.Figure)
            d = f.to_dict()
            json_str = json.dumps(d)
            self.assertIsInstance(json_str, str)
            self.assertGreater(len(json_str), 50)

    def test_inv_02_three_stream_color_hex_codes(self):
        """Verify #FF7043, #00E5FF, #D500F9 are embedded in graph visualizer."""
        fig = render_network_flow_figure(self.dataset, active_client_ids=[0, 2])
        fig_str = str(fig.to_dict()).lower()
        self.assertTrue("#ff7043" in fig_str or "255, 112, 67" in fig_str)
        self.assertTrue("#00e5ff" in fig_str or "0, 229, 255" in fig_str)
        self.assertTrue("#d500f9" in fig_str or "213, 0, 249" in fig_str)


if __name__ == "__main__":
    unittest.main()
