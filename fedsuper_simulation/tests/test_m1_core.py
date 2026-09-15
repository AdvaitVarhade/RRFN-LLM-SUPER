"""
tests/test_m1_core.py - Comprehensive Unit & Invariant Test Suite for Milestone 1
Verifies config validation, synthetic data generation, federated core & DP mechanics,
SUPER popularity blueprint calibration & LLM fusion, evaluation metrics, and core invariants.
"""
import math
import numpy as np
import pandas as pd
import pytest

try:
    from fedsuper_simulation.src.config import SimulationConfig
    from fedsuper_simulation.src.mock_data import SyntheticDataset, generate_mock_dataset
    from fedsuper_simulation.src.federated_core import FederatedServer, FederatedClient, FederatedSimulation
    from fedsuper_simulation.src.super_engine import (
        compute_user_popularity_blueprint,
        intra_pool_zscore_standardization,
        fuse_cf_and_llm_scores,
        calibrated_blueprint_merge,
    )
    from fedsuper_simulation.src.evaluator import (
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
except ImportError:
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


# ============================================================================
# 1. Configuration Unit & Boundary Tests
# ============================================================================

def test_default_simulation_config():
    """Verifies default hyperparameter values in SimulationConfig."""
    cfg = SimulationConfig()
    assert cfg.num_clients == 20
    assert cfg.num_items == 100
    assert cfg.embedding_dim == 32
    assert cfg.llm_dim == 32
    assert cfg.pareto_alpha == 0.20
    assert cfg.calibration_alpha == 0.40
    assert cfg.llm_lambda == 0.70
    assert cfg.dp_enabled is True
    assert cfg.dp_epsilon == 4.0
    assert cfg.dp_delta == 1e-5
    assert cfg.dp_l2_clip_norm == 1.0
    assert cfg.top_k == 10
    assert cfg.clients_per_round == 5
    assert cfg.learning_rate == 0.05
    assert cfg.max_rounds == 20
    assert cfg.seed == 42
    assert cfg.merge_mode == "stochastic"


def test_config_dp_sigma_formula():
    """Verifies theoretical Gaussian DP sigma matches (C * sqrt(2 * ln(1.25 / delta))) / epsilon."""
    cfg = SimulationConfig(dp_enabled=True, dp_epsilon=4.0, dp_delta=1e-5, dp_l2_clip_norm=1.0)
    expected_sigma = (1.0 * math.sqrt(2.0 * math.log(1.25 / 1e-5))) / 4.0
    assert math.isclose(cfg.dp_sigma, expected_sigma, rel_tol=1e-6)

    # When DP is disabled, sigma should be 0.0
    cfg_no_dp = SimulationConfig(dp_enabled=False)
    assert cfg_no_dp.dp_sigma == 0.0


@pytest.mark.parametrize("invalid_param,value", [
    ("num_clients", 0),
    ("num_clients", -5),
    ("num_items", 0),
    ("embedding_dim", 0),
    ("llm_dim", 0),
    ("pareto_alpha", -0.1),
    ("pareto_alpha", 1.5),
    ("calibration_alpha", -0.01),
    ("calibration_alpha", 1.01),
    ("llm_lambda", -0.5),
    ("llm_lambda", 1.5),
    ("dp_epsilon", 0.0),
    ("dp_epsilon", -1.0),
    ("dp_delta", 0.0),
    ("dp_delta", 1.0),
    ("dp_l2_clip_norm", 0.0),
    ("top_k", 0),
    ("top_k", 200),  # top_k > num_items (100)
    ("clients_per_round", 0),
    ("clients_per_round", 50),  # > num_clients (20)
    ("learning_rate", 0.0),
    ("max_rounds", 0),
    ("merge_mode", "invalid_strategy"),
    ("num_genres", 0),
])
def test_config_validation_raises_error(invalid_param, value):
    """Verifies that invalid hyperparameters raise ValueError in __post_init__."""
    kwargs = {invalid_param: value}
    with pytest.raises(ValueError):
        SimulationConfig(**kwargs)


def test_config_serialization():
    """Verifies to_dict and from_dict roundtrip."""
    cfg = SimulationConfig(num_clients=15, calibration_alpha=0.6)
    d = cfg.to_dict()
    assert d["num_clients"] == 15
    assert d["calibration_alpha"] == 0.6
    cfg2 = SimulationConfig.from_dict(d)
    assert cfg2.num_clients == 15
    assert cfg2.calibration_alpha == 0.6


# ============================================================================
# 2. Synthetic Mock Data Generator Tests
# ============================================================================

def test_mock_dataset_generation_shapes():
    """Verifies SyntheticDataset output shapes and partition sizes."""
    cfg = SimulationConfig(num_clients=20, num_items=100, embedding_dim=32, llm_dim=32, seed=42)
    dataset = generate_mock_dataset(cfg)

    assert dataset.num_users == 20
    assert dataset.num_items == 100
    assert dataset.train_matrix.shape == (20, 100)
    assert dataset.train_matrix.dtype == np.float32
    assert len(dataset.test_dict) == 20
    assert dataset.user_llm_profiles.shape == (20, 32)
    assert dataset.item_llm_embeddings.shape == (100, 32)
    assert len(dataset.user_metadata) == 20
    assert len(dataset.item_metadata) == 100


def test_mock_dataset_l2_normalization():
    """Verifies that LLM semantic profile and item vectors are strictly L2 unit normalized."""
    cfg = SimulationConfig(num_clients=20, num_items=100, seed=42)
    dataset = generate_mock_dataset(cfg)

    user_norms = np.linalg.norm(dataset.user_llm_profiles, axis=1)
    item_norms = np.linalg.norm(dataset.item_llm_embeddings, axis=1)

    np.testing.assert_allclose(user_norms, 1.0, atol=1e-5)
    np.testing.assert_allclose(item_norms, 1.0, atol=1e-5)


def test_mock_dataset_partition_completeness_and_disjointness():
    """Verifies Head (20%), Torso (30%), Tail (50%) partitions form a complete disjoint partition."""
    cfg = SimulationConfig(num_clients=20, num_items=100, seed=42)
    dataset = generate_mock_dataset(cfg)

    head_set = set(dataset.head_idx.tolist())
    torso_set = set(dataset.torso_idx.tolist())
    tail_set = set(dataset.tail_idx.tolist())

    # Disjointness
    assert len(head_set & torso_set) == 0
    assert len(head_set & tail_set) == 0
    assert len(torso_set & tail_set) == 0

    # Completeness
    all_items = head_set | torso_set | tail_set
    assert all_items == set(range(100))
    assert len(dataset.head_idx) == 20
    assert len(dataset.torso_idx) == 30
    assert len(dataset.tail_idx) == 50


def test_mock_dataset_leave_one_out_split():
    """Verifies leave-one-out test split has no data leakage into train_matrix."""
    cfg = SimulationConfig(num_clients=20, num_items=100, seed=42)
    dataset = generate_mock_dataset(cfg)

    for u in range(dataset.num_users):
        held_out = dataset.test_dict[u]
        assert len(held_out) == 1
        test_item = held_out[0]
        # Test item must not be present in train_matrix for user u
        assert dataset.train_matrix[u, test_item] == 0.0
        # User must have at least 1 training interaction
        assert np.sum(dataset.train_matrix[u]) >= 1


def test_mock_dataset_small_catalog_boundary():
    """Verifies generation robustness on small catalog edge cases."""
    cfg_small = SimulationConfig(num_clients=5, num_items=10, top_k=3, clients_per_round=3, seed=123)
    dataset_small = generate_mock_dataset(cfg_small)
    assert dataset_small.num_users == 5
    assert dataset_small.num_items == 10
    assert len(dataset_small.head_idx) + len(dataset_small.torso_idx) + len(dataset_small.tail_idx) == 10


# ============================================================================
# 3. Federated Core & DP Engine Tests
# ============================================================================

def test_federated_server_sparse_fedavg():
    """Verifies FederatedServer initializes parameters and aggregates sparse updates correctly."""
    server = FederatedServer(num_items=10, embedding_dim=4, seed=42)
    assert server.item_embeddings.shape == (10, 4)
    assert server.item_biases.shape == (10,)

    orig_emb = server.item_embeddings.copy()
    orig_bias = server.item_biases.copy()

    # Update items 2 and 5
    touched_idx = np.array([2, 5], dtype=np.int64)
    d_emb = np.ones((2, 4), dtype=np.float32) * 0.1
    d_bias = np.ones(2, dtype=np.float32) * 0.05

    stats = server.aggregate_updates([(touched_idx, d_emb, d_bias)])
    assert stats["items_touched_count"] == 2
    assert stats["items_touched_pct"] == 20.0

    # Items 2 and 5 should have changed
    np.testing.assert_allclose(server.item_embeddings[2], orig_emb[2] + 0.1, atol=1e-5)
    np.testing.assert_allclose(server.item_embeddings[5], orig_emb[5] + 0.1, atol=1e-5)
    # Untouched items should remain identical
    np.testing.assert_allclose(server.item_embeddings[0], orig_emb[0], atol=1e-7)


def test_federated_client_local_step_and_dp_clipping():
    """Verifies FederatedClient local BPR step applies L2 clipping and produces valid deltas."""
    cfg = SimulationConfig(dp_enabled=True, dp_l2_clip_norm=1.0, dp_epsilon=4.0, dp_delta=1e-5, learning_rate=0.05)
    client = FederatedClient(
        user_id=0,
        train_items=np.array([1, 2, 3, 4], dtype=np.int64),
        llm_profile=np.ones(32, dtype=np.float32) / np.sqrt(32),
        embedding_dim=32,
        seed=42
    )

    global_emb = np.random.normal(0, 0.1, size=(50, 32)).astype(np.float32)
    global_bias = np.zeros(50, dtype=np.float32)

    touched_idx, delta_emb, delta_bias, loss = client.local_train_step(global_emb, global_bias, cfg)

    assert len(touched_idx) > 0
    assert delta_emb.shape == (len(touched_idx), 32)
    assert delta_bias.shape == (len(touched_idx),)
    assert loss >= 0.0

    # Client user embedding remains local
    assert client.user_embedding.shape == (32,)


def test_federated_simulation_stepping():
    """Verifies multi-round simulation stepping, history tracking, and recommendation synthesis."""
    cfg = SimulationConfig(num_clients=10, num_items=30, top_k=5, clients_per_round=4, seed=42)
    dataset = generate_mock_dataset(cfg)
    sim = FederatedSimulation(dataset, cfg)

    assert sim.current_round == 0
    record1 = sim.step_round()
    assert sim.current_round == 1
    assert len(record1["active_clients"]) == 4
    assert "loss" in record1
    assert "dp_sigma" in record1
    assert len(sim.history) == 1

    # Recommendations
    recs_uncalib = sim.get_recommendations(calibrated=False)
    recs_calib = sim.get_recommendations(calibrated=True)

    assert recs_uncalib.shape == (10, 5)
    assert recs_calib.shape == (10, 5)


# ============================================================================
# 4. SUPER Calibration & LLM Semantic Fusion Engine Tests
# ============================================================================

def test_user_popularity_blueprint_computation():
    """Verifies calculation of user popularity blueprint P_u."""
    # 2 users, 6 items (head: 0,1; torso: 2,3; tail: 4,5)
    train_mat = np.array([
        [1, 1, 1, 0, 0, 0],  # User 0: 2 head, 1 torso, 0 tail -> [2/3, 1/3, 0]
        [0, 0, 0, 1, 1, 1],  # User 1: 0 head, 1 torso, 2 tail -> [0, 1/3, 2/3]
        [0, 0, 0, 0, 0, 0],  # User 2: 0 items (cold start) -> uniform [1/3, 1/3, 1/3]
    ], dtype=np.float32)

    head = np.array([0, 1])
    torso = np.array([2, 3])
    tail = np.array([4, 5])

    blueprints = compute_user_popularity_blueprint(train_mat, head, torso, tail)

    np.testing.assert_allclose(blueprints[0], [2/3, 1/3, 0.0], atol=1e-5)
    np.testing.assert_allclose(blueprints[1], [0.0, 1/3, 2/3], atol=1e-5)
    np.testing.assert_allclose(blueprints[2], [1/3, 1/3, 1/3], atol=1e-5)
    np.testing.assert_allclose(blueprints.sum(axis=1), 1.0, atol=1e-6)


def test_intra_pool_zscore_standardization():
    """Verifies intra-pool standardization produces mean 0 and unit variance."""
    scores = np.array([
        [1.0, 3.0, 5.0, 10.0, 20.0, 30.0],
        [2.0, 4.0, 6.0, 0.0, 0.0, 0.0]
    ], dtype=np.float32)

    pool = np.array([0, 1, 2])
    std_scores = intra_pool_zscore_standardization(scores, pool)

    assert std_scores.shape == (2, 3)
    np.testing.assert_allclose(np.mean(std_scores, axis=1), 0.0, atol=1e-6)
    np.testing.assert_allclose(np.std(std_scores, axis=1), 1.0, atol=1e-5)


def test_calibrated_blueprint_merge_exact_quota_sum():
    """Verifies Largest Remainder method ensures exact sum-to-K quota balancing with no duplicates."""
    fused_scores = np.random.uniform(0, 1, size=(5, 30)).astype(np.float32)
    user_blueprints = np.array([
        [0.6, 0.3, 0.1],
        [0.1, 0.2, 0.7],
        [0.333, 0.333, 0.334],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ], dtype=np.float32)

    head_idx = np.arange(0, 6)
    torso_idx = np.arange(6, 15)
    tail_idx = np.arange(15, 30)

    recs = calibrated_blueprint_merge(
        fused_scores=fused_scores,
        user_blueprints=user_blueprints,
        head_idx=head_idx,
        torso_idx=torso_idx,
        tail_idx=tail_idx,
        top_k=10,
        calibration_alpha=0.5
    )

    assert recs.shape == (5, 10)
    for u in range(5):
        assert len(set(recs[u])) == 10, f"Duplicate recommendations found for user {u}"


# ============================================================================
# 5. Evaluation Metrics Engine Tests
# ============================================================================

def test_rmse_pc_hand_calculated():
    """Verifies calculate_rmse_pc on exact analytical cases."""
    # Perfect match: P == Q -> RMSE = 0.0
    p1 = np.array([[0.2, 0.3, 0.5]])
    q1 = np.array([[0.2, 0.3, 0.5]])
    assert calculate_rmse_pc(p1, q1) == 0.0

    # Inverted: P = [0, 0, 1], Q = [1, 0, 0] -> RMSE = sqrt((1^2 + 0 + (-1)^2) / 3) = sqrt(2/3)
    p2 = np.array([[0.0, 0.0, 1.0]])
    q2 = np.array([[1.0, 0.0, 0.0]])
    expected = math.sqrt(2.0 / 3.0)
    assert math.isclose(calculate_rmse_pc(p2, q2), expected, rel_tol=1e-5)


def test_gini_index_bounds():
    """Verifies calculate_gini_index boundary values."""
    # Uniform exposure -> Gini = 0.0
    uniform_exp = np.array([10, 10, 10, 10], dtype=np.float32)
    assert calculate_gini_index(uniform_exp) == 0.0

    # Extreme monopoly -> Gini -> 1.0
    monopoly_exp = np.array([1000, 0, 0, 0], dtype=np.float32)
    assert calculate_gini_index(monopoly_exp) >= 0.70


def test_long_tail_exposure_ratio():
    """Verifies calculate_long_tail_exposure_ratio fraction."""
    exp = np.array([10, 20, 30, 40], dtype=np.float32)
    tail_idx = np.array([2, 3])  # sum = 30 + 40 = 70; total = 100
    assert calculate_long_tail_exposure_ratio(exp, tail_idx) == 0.70


def test_catalog_coverage():
    """Verifies calculate_catalog_coverage fraction."""
    exp = np.array([5, 0, 10, 0, 2], dtype=np.float32)
    assert calculate_catalog_coverage(exp) == 3.0 / 5.0


def test_recall_and_ndcg_at_k():
    """Verifies calculate_recall_at_k and calculate_ndcg_at_k."""
    recs = np.array([
        [10, 20, 30, 40, 50],
        [5, 15, 25, 35, 45]
    ])
    test_dict = {
        0: [10, 30],  # 2 hits at rank 0, 2
        1: [99]       # 0 hits
    }
    # User 0 hit rate = 2/2 = 1.0; User 1 hit rate = 0/1 = 0.0 -> Recall = 0.5
    assert calculate_recall_at_k(recs, test_dict, top_k=5) == 0.5

    # NDCG for user 0: DCG = 1/log2(2) + 1/log2(4) = 1.0 + 0.5 = 1.5. IDCG = 1/log2(2) + 1/log2(3) = 1.0 + 0.6309 = 1.6309.
    ndcg = calculate_ndcg_at_k(recs, test_dict, top_k=5)
    assert 0.40 < ndcg < 0.60


def test_novelty_and_ild():
    """Verifies novelty and intra-list diversity computations."""
    recs = np.array([[0, 1, 2], [3, 4, 5]])
    item_pop = np.array([100, 50, 10, 5, 2, 1], dtype=np.float32)
    nov = calculate_novelty(recs, item_pop, total_interactions=168)
    assert nov > 0.0

    # ILD
    embs = np.eye(6, dtype=np.float32)  # orthogonal vectors -> distance = 1.0
    ild = calculate_ild(recs, embs)
    np.testing.assert_allclose(ild, 1.0, atol=1e-5)


def test_evaluate_all_metrics_structure():
    """Verifies evaluate_all_metrics outputs a fully populated dictionary with uncalibrated vs calibrated."""
    cfg = SimulationConfig(num_clients=10, num_items=30, top_k=5, seed=42)
    dataset = generate_mock_dataset(cfg)
    sim = FederatedSimulation(dataset, cfg)
    sim.step_round()

    recs_uncalib = sim.get_recommendations(calibrated=False)
    recs_calib = sim.get_recommendations(calibrated=True)

    metrics = evaluate_all_metrics(dataset, recs_uncalib, recs_calib, top_k=5)

    assert "uncalibrated" in metrics
    assert "calibrated" in metrics
    assert "blueprint" in metrics
    assert "kpis" in metrics

    assert "rmse_pc" in metrics["calibrated"]
    assert "gini_index" in metrics["calibrated"]
    assert "long_tail_exposure_ratio" in metrics["calibrated"]
    assert "catalog_coverage" in metrics["calibrated"]
    assert "recall_at_k" in metrics["calibrated"]
    assert "ndcg_at_k" in metrics["calibrated"]
    assert "novelty" in metrics["calibrated"]
    assert "ild" in metrics["calibrated"]


# ============================================================================
# 6. Core Milestone 1 Calibration Invariant Guarantee
# ============================================================================

def test_calibration_guarantee_uncalib_vs_calib_invariants():
    """
    MANDATORY ACCEPTANCE INVARIANT:
    Calibrated recommendations satisfy:
      - Calibrated Rmse-PC <= 0.060 (vs uncalibrated >= 0.300)
      - Calibrated Gini <= 0.400 (vs uncalibrated >= 0.700)
      - Calibrated LTER >= 0.250 (vs uncalibrated <= 0.080)
      - Calibrated Catalog Coverage >= 0.800 (vs uncalibrated <= 0.400)
    """
    cfg = SimulationConfig(
        num_clients=20,
        num_items=100,
        calibration_alpha=0.40,
        llm_lambda=0.70,
        top_k=10,
        seed=42
    )
    dataset = generate_mock_dataset(cfg, seed=42)
    sim = FederatedSimulation(dataset, cfg)

    # Train for 5 rounds
    for _ in range(5):
        sim.step_round()

    recs_uncalib = sim.get_recommendations(calibrated=False)
    recs_calib = sim.get_recommendations(calibrated=True)

    metrics = evaluate_all_metrics(dataset, recs_uncalib, recs_calib, top_k=10)

    calib_rmse = metrics["calibrated"]["rmse_pc"]
    uncalib_rmse = metrics["uncalibrated"]["rmse_pc"]
    calib_gini = metrics["calibrated"]["gini_index"]
    uncalib_gini = metrics["uncalibrated"]["gini_index"]
    calib_lter = metrics["calibrated"]["long_tail_exposure_ratio"]
    uncalib_lter = metrics["uncalibrated"]["long_tail_exposure_ratio"]
    calib_cov = metrics["calibrated"]["catalog_coverage"]

    # Assert invariant thresholds
    assert calib_rmse <= 0.060, f"Calibrated Rmse-PC {calib_rmse:.4f} > 0.060"
    assert uncalib_rmse >= 0.300, f"Uncalibrated Rmse-PC {uncalib_rmse:.4f} < 0.300"
    assert calib_gini <= 0.400, f"Calibrated Gini {calib_gini:.4f} > 0.400"
    assert uncalib_gini >= 0.700, f"Uncalibrated Gini {uncalib_gini:.4f} < 0.700"
    assert calib_lter >= 0.250, f"Calibrated LTER {calib_lter:.4f} < 0.250"
    assert uncalib_lter <= 0.080, f"Uncalibrated LTER {uncalib_lter:.4f} > 0.080"
    assert calib_cov >= 0.800, f"Calibrated Coverage {calib_cov:.4f} < 0.800"
