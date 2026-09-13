"""
Privacy-Preserving Federated SUPER Simulation & Recommendation Package.
Exports all core configuration, dataset generation, federated training,
SUPER popularity blueprint calibration, evaluation metrics, network topology
visualizers, and analytical chart generators.
"""

# 1. Configuration & Synthetic Mock Data (M1)
from .config import SimulationConfig
from .mock_data import SyntheticDataset, generate_mock_dataset

# 2. Federated Core & Differential Privacy (M1)
from .federated_core import FederatedServer, FederatedClient, FederatedSimulation

# 3. SUPER Popularity Blueprint Calibration & LLM Semantic Fusion (M1)
from .super_engine import (
    compute_user_popularity_blueprint,
    intra_pool_zscore_standardization,
    fuse_cf_and_llm_scores,
    calibrated_blueprint_merge,
)

# 4. Comprehensive Evaluation Metrics Engine (M1)
from .evaluator import (
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

# 5. Network Flow Graph Visualizer & Topology Engine (M2)
from .graph_visualizer import (
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
)

# 6. Analytical Popularity Calibration Chart Generators (M2)
from .chart_generator import (
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

__all__ = [
    # M1 Config & Data
    "SimulationConfig",
    "SyntheticDataset",
    "generate_mock_dataset",
    # M1 Federated Core
    "FederatedServer",
    "FederatedClient",
    "FederatedSimulation",
    # M1 SUPER Engine
    "compute_user_popularity_blueprint",
    "intra_pool_zscore_standardization",
    "fuse_cf_and_llm_scores",
    "calibrated_blueprint_merge",
    # M1 Evaluator
    "calculate_rmse_pc",
    "calculate_gini_index",
    "calculate_long_tail_exposure_ratio",
    "calculate_catalog_coverage",
    "calculate_recall_at_k",
    "calculate_ndcg_at_k",
    "calculate_novelty",
    "calculate_ild",
    "compute_exposure_counts",
    "compute_recommendation_distributions",
    "evaluate_all_metrics",
    # M2 Network Flow Visualizer
    "render_network_flow_figure",
    "get_network_topology_layout",
    "compute_radial_layout",
    "create_client_sandbox_halos_trace",
    "create_sandbox_halos_trace",
    "create_client_nodes_trace",
    "create_gradient_flow_traces",
    "create_llm_stream_trace",
    "create_server_node_trace",
    "create_llm_node_trace",
    "create_background_mesh_trace",
    # M2 Chart Generators & Aliases
    "render_exposure_comparison_chart",
    "render_exposure_distribution_chart",
    "render_rank_exposure_curve",
    "render_long_tail_flattening_chart",
    "render_calibration_error_chart",
    "render_user_calibration_distribution",
    "render_pareto_front_chart",
    "render_pareto_frontier_chart",
    "render_convergence_chart",
    "render_training_convergence_chart",
    "apply_dark_theme",
]
