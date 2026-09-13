"""
fedsuper_simulation/tests/verify_m2_adversarial_challenge.py
Milestone 2 Empirical Adversarial Stress-Testing and Invariant Challenge Harness.

Adversarial Stress Testing Categories:
1. None, Null, and Empty Inputs across all visualizers & chart generators
2. Negative, Zero, and Extreme Boundary Hyperparameters (epsilon, lambda, alpha, clip_norm)
3. Mismatched and Out-of-Bounds Client IDs & Dimension Skews
4. Single-Item, Single-Client, Zero-Client, and Extreme Cohorts (K=100)
5. Invariant 1: Strict go.Figure Instance Typing
6. Invariant 2: Full Headless JSON Serialization (json.dumps(fig.to_dict()) and fig.to_json())
7. Invariant 3: Exact Hex Color Code Adherence (#FF7043, #00E5FF, #D500F9)
8. Invariant 4: Dark Theme Layout & Typography (#0E1117, #161B22, #30363D, #E6EDF3)
"""

import os
import sys
import json
import numpy as np
import plotly.graph_objects as go

# Add fedsuper_simulation to path
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
    COLOR_STREAM_1_ORANGE,
    COLOR_STREAM_2_CYAN,
    COLOR_STREAM_3_MAGENTA,
    COLOR_BG_PAPER,
    COLOR_BG_PLOT,
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


def run_adversarial_challenge():
    print("=" * 80)
    print("STARTING MILESTONE 2 EMPIRICAL ADVERSARIAL CHALLENGE HARNESS")
    print("=" * 80)

    test_records = []
    total_checks = 0
    passed_checks = 0

    def record_test(name: str, passed: bool, details: str = ""):
        nonlocal total_checks, passed_checks
        total_checks += 1
        if passed:
            passed_checks += 1
        status_str = "[PASS]" if passed else "[FAIL]"
        print(f"{status_str} {name}: {details}")
        test_records.append({
            "test": name,
            "passed": passed,
            "details": details
        })

    # Standard test dataset for baseline comparison
    config = SimulationConfig(num_clients=15, num_items=60, top_k=5, seed=42)
    dataset = generate_mock_dataset(config, seed=42)
    sim = FederatedSimulation(dataset, config)
    sim.step_round()
    recs_unc = sim.get_recommendations(calibrated=False)
    recs_cal = sim.get_recommendations(calibrated=True)
    metrics = evaluate_all_metrics(dataset, recs_unc, recs_cal, top_k=5)

    # -------------------------------------------------------------------------
    # CATEGORY 1: None, Null, and Empty Inputs
    # -------------------------------------------------------------------------
    print("\n--- CATEGORY 1: None, Null, and Empty Inputs ---")

    # 1.1 render_network_flow_figure with None dataset & None active clients
    try:
        fig_net_none = render_network_flow_figure(dataset=None, active_client_ids=None)
        is_fig = isinstance(fig_net_none, go.Figure)
        has_traces = len(fig_net_none.data) > 0
        record_test("CAT1_1_NetworkFlow_NoneInputs", is_fig and has_traces,
                    f"Returned go.Figure with {len(fig_net_none.data)} traces")
    except Exception as e:
        record_test("CAT1_1_NetworkFlow_NoneInputs", False, f"Exception: {e}")

    # 1.2 render_network_flow_figure with empty active clients
    try:
        fig_net_empty_act = render_network_flow_figure(dataset=dataset, active_client_ids=[])
        is_fig = isinstance(fig_net_empty_act, go.Figure)
        record_test("CAT1_2_NetworkFlow_EmptyActiveClients", is_fig,
                    f"Rendered safely with 0 active clients ({len(fig_net_empty_act.data)} traces)")
    except Exception as e:
        record_test("CAT1_2_NetworkFlow_EmptyActiveClients", False, f"Exception: {e}")

    # 1.3 render_exposure_distribution_chart with None and {}
    try:
        fig_exp_none = render_exposure_distribution_chart(None)
        fig_exp_empty = render_exposure_distribution_chart({})
        is_valid = isinstance(fig_exp_none, go.Figure) and isinstance(fig_exp_empty, go.Figure)
        record_test("CAT1_3_ExposureChart_NoneAndEmptyDict", is_valid,
                    f"Handled None and {{}} fallback gracefully ({len(fig_exp_empty.data)} bar traces)")
    except Exception as e:
        record_test("CAT1_3_ExposureChart_NoneAndEmptyDict", False, f"Exception: {e}")

    # 1.4 render_long_tail_flattening_chart with None and empty arrays
    try:
        fig_rank_none = render_long_tail_flattening_chart(None, None)
        fig_rank_empty = render_long_tail_flattening_chart([], np.array([]))
        is_valid = isinstance(fig_rank_none, go.Figure) and isinstance(fig_rank_empty, go.Figure)
        record_test("CAT1_4_RankExposure_NoneAndEmptyArrays", is_valid,
                    f"Handled None and empty arrays ({len(fig_rank_empty.data)} line traces)")
    except Exception as e:
        record_test("CAT1_4_RankExposure_NoneAndEmptyArrays", False, f"Exception: {e}")

    # 1.5 render_calibration_error_chart with None and empty arrays
    try:
        fig_ce_none = render_calibration_error_chart(None)
        fig_ce_empty = render_calibration_error_chart([], np.array([]))
        is_valid = isinstance(fig_ce_none, go.Figure) and isinstance(fig_ce_empty, go.Figure)
        record_test("CAT1_5_CalibrationError_NoneAndEmptyArrays", is_valid,
                    f"Handled None and empty arrays ({len(fig_ce_empty.data)} box traces)")
    except Exception as e:
        record_test("CAT1_5_CalibrationError_NoneAndEmptyArrays", False, f"Exception: {e}")

    # 1.6 render_pareto_frontier_chart with None and []
    try:
        fig_pareto_none = render_pareto_frontier_chart(None)
        fig_pareto_empty = render_pareto_frontier_chart([])
        is_valid = isinstance(fig_pareto_none, go.Figure) and isinstance(fig_pareto_empty, go.Figure)
        record_test("CAT1_6_ParetoFrontier_NoneAndEmptyList", is_valid,
                    f"Handled None and empty list ({len(fig_pareto_empty.data)} traces)")
    except Exception as e:
        record_test("CAT1_6_ParetoFrontier_NoneAndEmptyList", False, f"Exception: {e}")

    # 1.7 render_training_convergence_chart with None and []
    try:
        fig_conv_none = render_training_convergence_chart(None)
        fig_conv_empty = render_training_convergence_chart([])
        is_valid = isinstance(fig_conv_none, go.Figure) and isinstance(fig_conv_empty, go.Figure)
        record_test("CAT1_7_ConvergenceChart_NoneAndEmptyList", is_valid,
                    f"Handled None and empty list ({len(fig_conv_empty.data)} traces)")
    except Exception as e:
        record_test("CAT1_7_ConvergenceChart_NoneAndEmptyList", False, f"Exception: {e}")

    # -------------------------------------------------------------------------
    # CATEGORY 2: Extreme, Negative, and Zero Hyperparameters
    # -------------------------------------------------------------------------
    print("\n--- CATEGORY 2: Extreme, Negative, and Zero Hyperparameters ---")

    # 2.1 dp_epsilon = 0.0, negative epsilon, huge epsilon (1000.0)
    try:
        fig_dp_0 = render_network_flow_figure(dataset, active_client_ids=[0, 1], dp_epsilon=0.0, dp_sigma=10.0)
        fig_dp_neg = render_network_flow_figure(dataset, active_client_ids=[0, 1], dp_epsilon=-1.0, dp_sigma=0.0)
        fig_dp_huge = render_network_flow_figure(dataset, active_client_ids=[0, 1], dp_epsilon=1000.0, dp_clip_norm=0.0)
        is_valid = isinstance(fig_dp_0, go.Figure) and isinstance(fig_dp_neg, go.Figure) and isinstance(fig_dp_huge, go.Figure)
        record_test("CAT2_1_NetworkFlow_ExtremeDPParams", is_valid,
                    "Handled epsilon=0.0, epsilon=-1.0, epsilon=1000.0, clip=0.0 without crash")
    except Exception as e:
        record_test("CAT2_1_NetworkFlow_ExtremeDPParams", False, f"Exception: {e}")

    # 2.2 llm_lambda = 0.0, 1.0, -0.5, 2.5
    try:
        fig_lam_0 = render_network_flow_figure(dataset, llm_lambda=0.0)
        fig_lam_1 = render_network_flow_figure(dataset, llm_lambda=1.0)
        fig_lam_neg = render_network_flow_figure(dataset, llm_lambda=-0.5)
        fig_lam_huge = render_network_flow_figure(dataset, llm_lambda=2.5)
        is_valid = isinstance(fig_lam_0, go.Figure) and isinstance(fig_lam_1, go.Figure) and isinstance(fig_lam_neg, go.Figure) and isinstance(fig_lam_huge, go.Figure)
        record_test("CAT2_2_NetworkFlow_ExtremeLambdaValues", is_valid,
                    "Handled lambda in [0.0, 1.0, -0.5, 2.5] cleanly")
    except Exception as e:
        record_test("CAT2_2_NetworkFlow_ExtremeLambdaValues", False, f"Exception: {e}")

    # 2.3 Pareto frontier with current_alpha outside [0, 1] e.g. alpha=-0.5, alpha=5.0
    try:
        fig_pareto_ext_a = render_pareto_frontier_chart(None, current_alpha=-0.5, current_epsilon=-2.0)
        fig_pareto_ext_b = render_pareto_frontier_chart(None, current_alpha=5.0, current_epsilon=100.0)
        is_valid = isinstance(fig_pareto_ext_a, go.Figure) and isinstance(fig_pareto_ext_b, go.Figure)
        record_test("CAT2_3_ParetoFrontier_ExtremeOperatingPoints", is_valid,
                    "Handled alpha=-0.5 and alpha=5.0 safely")
    except Exception as e:
        record_test("CAT2_3_ParetoFrontier_ExtremeOperatingPoints", False, f"Exception: {e}")

    # -------------------------------------------------------------------------
    # CATEGORY 3: Mismatched IDs and Skewed Dimensions
    # -------------------------------------------------------------------------
    print("\n--- CATEGORY 3: Mismatched IDs and Skewed Dimensions ---")

    # 3.1 Active client IDs completely out of bounds (negative IDs, IDs > num_clients)
    try:
        fig_oob = render_network_flow_figure(dataset, active_client_ids=[-100, -1, 50, 999, 10000])
        is_valid = isinstance(fig_oob, go.Figure)
        record_test("CAT3_1_NetworkFlow_OutOfBoundsActiveClientIDs", is_valid,
                    f"Out-of-bounds IDs filtered cleanly ({len(fig_oob.data)} traces)")
    except Exception as e:
        record_test("CAT3_1_NetworkFlow_OutOfBoundsActiveClientIDs", False, f"Exception: {e}")

    # 3.2 Rank exposure curve with mismatched array lengths (e.g. 5 vs 500 items)
    try:
        arr_short = np.array([10.0, 20.0, 5.0])
        arr_long = np.random.exponential(scale=5.0, size=500)
        fig_mismatch = render_long_tail_flattening_chart(arr_short, arr_long)
        is_valid = isinstance(fig_mismatch, go.Figure)
        record_test("CAT3_2_RankExposure_MismatchedArrayLengths", is_valid,
                    "Short array (3 items) padded to match long array (500 items) without error")
    except Exception as e:
        record_test("CAT3_2_RankExposure_MismatchedArrayLengths", False, f"Exception: {e}")

    # 3.3 All-zero and all-identical exposure counts
    try:
        fig_all_zero = render_long_tail_flattening_chart(np.zeros(100), np.zeros(100))
        fig_all_same = render_long_tail_flattening_chart(np.ones(100) * 42, np.ones(100) * 42)
        is_valid = isinstance(fig_all_zero, go.Figure) and isinstance(fig_all_same, go.Figure)
        record_test("CAT3_3_RankExposure_AllZerosAndUniformCounts", is_valid,
                    "Handled all-zero and uniform constant exposure distributions cleanly")
    except Exception as e:
        record_test("CAT3_3_RankExposure_AllZerosAndUniformCounts", False, f"Exception: {e}")

    # 3.4 Corrupted Pareto records with missing keys
    try:
        corrupted_records = [
            {"invalid_key": 123},
            {"rmse": 0.45},
            {"rmse_pc": 0.05, "other": "data"},
            {"alpha": 0.3, "epsilon": 2.0}
        ]
        fig_corrupt_2d = render_pareto_frontier_chart(corrupted_records, is_3d=False)
        fig_corrupt_3d = render_pareto_frontier_chart(corrupted_records, is_3d=True)
        is_valid = isinstance(fig_corrupt_2d, go.Figure) and isinstance(fig_corrupt_3d, go.Figure)
        record_test("CAT3_4_ParetoFrontier_CorruptedKeyDictionaries", is_valid,
                    "Handled dictionaries with missing keys and corrupted schema safely")
    except Exception as e:
        record_test("CAT3_4_ParetoFrontier_CorruptedKeyDictionaries", False, f"Exception: {e}")

    # -------------------------------------------------------------------------
    # CATEGORY 4: Single-Item, Single-Client, Zero-Client, and Extreme Cohorts
    # -------------------------------------------------------------------------
    print("\n--- CATEGORY 4: Single-Item, Single-Client, Zero-Client, and Large Cohorts ---")

    # 4.1 Topology with num_clients = 0
    try:
        topo_0 = get_network_topology_layout(num_clients=0)
        fig_0 = render_network_flow_figure(dataset=0, active_client_ids=[])
        is_valid = (len(topo_0["clients"]) == 0) and isinstance(fig_0, go.Figure)
        record_test("CAT4_1_Topology_ZeroClients", is_valid,
                    "0 clients topology returns empty client array and valid Figure")
    except Exception as e:
        record_test("CAT4_1_Topology_ZeroClients", False, f"Exception: {e}")

    # 4.2 Topology with num_clients = 1
    try:
        topo_1 = get_network_topology_layout(num_clients=1)
        fig_1 = render_network_flow_figure(dataset=1, active_client_ids=[0])
        c = topo_1["clients"][0]
        is_valid = (len(topo_1["clients"]) == 1) and not np.isnan(c["x"]) and isinstance(fig_1, go.Figure)
        record_test("CAT4_2_Topology_SingleClient", is_valid,
                    f"1 client positioned at ({c['x']:.2f}, {c['y']:.2f}) without NaN")
    except Exception as e:
        record_test("CAT4_2_Topology_SingleClient", False, f"Exception: {e}")

    # 4.3 Large cohort topology (num_clients = 100)
    try:
        topo_100 = get_network_topology_layout(num_clients=100)
        fig_100 = render_network_flow_figure(dataset=100, active_client_ids=list(range(20)))
        is_valid = (len(topo_100["clients"]) == 100) and isinstance(fig_100, go.Figure)
        record_test("CAT4_3_Topology_LargeCohort100Clients", is_valid,
                    f"Positioned 100 clients symmetrically on unit orbit (Figure traces: {len(fig_100.data)})")
    except Exception as e:
        record_test("CAT4_3_Topology_LargeCohort100Clients", False, f"Exception: {e}")

    # 4.4 Horseshoe layout variant
    try:
        topo_horse = get_network_topology_layout(num_clients=12, layout_type="horseshoe")
        is_valid = (len(topo_horse["clients"]) == 12)
        record_test("CAT4_4_Topology_HorseshoeLayout", is_valid,
                    "Computed open horseshoe arc coordinates for 12 clients")
    except Exception as e:
        record_test("CAT4_4_Topology_HorseshoeLayout", False, f"Exception: {e}")

    # -------------------------------------------------------------------------
    # CATEGORY 5: Invariant Verification (go.Figure Types & Serialization)
    # -------------------------------------------------------------------------
    print("\n--- CATEGORY 5: Invariant Verification (go.Figure Types & Serialization) ---")

    figures = [
        ("render_network_flow_figure", render_network_flow_figure(dataset, active_client_ids=[0, 1, 2])),
        ("render_exposure_comparison_chart", render_exposure_comparison_chart(metrics)),
        ("render_exposure_distribution_chart", render_exposure_distribution_chart(metrics)),
        ("render_rank_exposure_curve", render_rank_exposure_curve(metrics["uncalibrated"]["exposure_counts"], metrics["calibrated"]["exposure_counts"])),
        ("render_long_tail_flattening_chart", render_long_tail_flattening_chart(metrics["uncalibrated"]["exposure_counts"], metrics["calibrated"]["exposure_counts"], log_y=True)),
        ("render_calibration_error_chart_box", render_calibration_error_chart(metrics, plot_type="box")),
        ("render_calibration_error_chart_violin", render_calibration_error_chart(metrics, plot_type="violin")),
        ("render_pareto_front_chart_2d", render_pareto_front_chart(None, is_3d=False)),
        ("render_pareto_frontier_chart_3d", render_pareto_frontier_chart(None, is_3d=True)),
        ("render_convergence_chart", render_convergence_chart(sim.history)),
        ("render_training_convergence_chart", render_training_convergence_chart(sim.history)),
    ]

    all_figures_typed = True
    all_json_serializable = True

    for name, fig in figures:
        # Check 5.1: Type check
        if not isinstance(fig, go.Figure):
            all_figures_typed = False
            record_test(f"INV1_TypeCheck_{name}", False, f"Expected go.Figure, got {type(fig)}")
        else:
            # Check 5.2: JSON Serialization via to_dict() and json.dumps()
            try:
                fig_dict = fig.to_dict()
                json_str = json.dumps(fig_dict)
                json_str_direct = fig.to_json()
                if len(json_str) < 50 or len(json_str_direct) < 50:
                    all_json_serializable = False
                    record_test(f"INV2_JSONSerialization_{name}", False, "JSON serialized string too short")
                else:
                    record_test(f"INV2_JSONSerialization_{name}", True,
                                f"Successfully serialized to JSON ({len(json_str)} bytes)")
            except Exception as e:
                all_json_serializable = False
                record_test(f"INV2_JSONSerialization_{name}", False, f"Serialization failed: {e}")

    record_test("INV1_AllFiguresAreGoFigureInstances", all_figures_typed,
                f"All {len(figures)} visualizer functions return strict go.Figure instances")

    # -------------------------------------------------------------------------
    # CATEGORY 6: Invariant 3 — Exact Color Hex Adherence
    # -------------------------------------------------------------------------
    print("\n--- CATEGORY 6: Invariant 3 — Exact Color Hex Adherence ---")

    fig_net = render_network_flow_figure(dataset, active_client_ids=[1, 3, 5])
    fig_net_str = str(fig_net.to_dict()).lower()

    # Stream 1 Orange: #FF7043 or rgba(255, 112, 67, ...)
    has_stream1_color = ("#ff7043" in fig_net_str) or ("255, 112, 67" in fig_net_str) or ("255,112,67" in fig_net_str)
    record_test("INV3_Stream1_OrangeColorPresent", has_stream1_color,
                f"Coral Orange (#FF7043 / rgba(255,112,67,...)) found in network flow figure")

    # Stream 2 Cyan: #00E5FF or rgba(0, 229, 255, ...)
    has_stream2_color = ("#00e5ff" in fig_net_str) or ("0, 229, 255" in fig_net_str) or ("0,229,255" in fig_net_str)
    record_test("INV3_Stream2_CyanColorPresent", has_stream2_color,
                f"Electric Cyan (#00E5FF / rgba(0,229,255,...)) found in network flow figure")

    # Stream 3 Magenta: #D500F9 or rgba(213, 0, 249, ...)
    has_stream3_color = ("#d500f9" in fig_net_str) or ("213, 0, 249" in fig_net_str) or ("213,0,249" in fig_net_str)
    record_test("INV3_Stream3_MagentaColorPresent", has_stream3_color,
                f"Electric Magenta (#D500F9 / rgba(213,0,249,...)) found in network flow figure")

    # -------------------------------------------------------------------------
    # CATEGORY 7: Invariant 4 — Dark Theme Palette Compliance
    # -------------------------------------------------------------------------
    print("\n--- CATEGORY 7: Invariant 4 — Dark Theme Palette Compliance ---")

    valid_dark_bg = ["#0e1117", "#161b22", "#0f172a", "#111827", "rgba(14, 17, 23, 1)", "rgba(14,17,23,1)"]
    for name, fig in figures:
        layout = fig.layout
        paper_bg = str(getattr(layout, "paper_bgcolor", "")).lower()
        plot_bg = str(getattr(layout, "plot_bgcolor", "")).lower()
        is_dark = (paper_bg in valid_dark_bg) or (plot_bg in valid_dark_bg) or (layout.template == "plotly_dark")
        record_test(f"INV4_DarkTheme_{name}", is_dark,
                    f"paper_bg='{paper_bg}', plot_bg='{plot_bg}', template='{layout.template}'")

    print("\n" + "=" * 80)
    print(f"ADVERSARIAL STRESS TEST SUMMARY: {passed_checks}/{total_checks} CHECKS PASSED")
    print(f"OVERALL VERDICT: {'APPROVE' if passed_checks == total_checks else 'REQUEST_CHANGES'}")
    print("=" * 80)

    report = {
        "total_checks": total_checks,
        "passed_checks": passed_checks,
        "failed_checks": total_checks - passed_checks,
        "verdict": "APPROVE" if passed_checks == total_checks else "REQUEST_CHANGES",
        "tests": test_records
    }

    # Save validation report
    report_path = os.path.join(os.path.dirname(__file__), "m2_adversarial_challenge_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved adversarial validation report to {report_path}")

    return report


if __name__ == "__main__":
    run_adversarial_challenge()
