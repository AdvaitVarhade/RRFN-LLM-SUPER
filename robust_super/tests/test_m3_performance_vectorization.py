import time
import pytest
import torch
import numpy as np
from unittest.mock import patch

from robust_super.src.models.lightgcn import LightGCN
from robust_super.src.models.simgcl import SimGCL
from robust_super.src.models.sgl import SGL
from robust_super.src.models.neumf import NeuMF
from robust_super.src.models.vaecf import VaeCF

from robust_super.src.bombing_detector import (
    compute_temporal_acceleration,
    compute_polarity_skew,
    compute_bomb_scores,
    sweep_omega_sensitivity,
)
from robust_super.src.fusion.reliability_fusion import (
    compute_R_RRFN,
    fuse_reliability_scores,
)
from torch.utils.data import TensorDataset, DataLoader


def test_compute_temporal_acceleration_vectorization_and_speed():
    """
    Verifies that compute_temporal_acceleration is vectorized,
    runs in sub-second time on thousands of interactions,
    and produces mathematically correct acceleration scores.
    """
    num_users = 100
    num_items = 50
    rng = np.random.RandomState(42)

    # Generate synthetic train_dict with 5,000 interactions
    train_dict = {}
    base_ts = 1700000000
    for u in range(num_users):
        interactions = []
        # Each user interacts with 20 items
        items = rng.choice(num_items, size=20, replace=False)
        for i in items:
            rating = int(rng.choice([1, 2, 3, 4, 5]))
            # Burst pattern on item 0: clustered in same hour
            if i == 0:
                ts = base_ts + int(rng.randint(0, 3600))
            else:
                ts = base_ts + int(rng.randint(0, 3600 * 24 * 30))
            interactions.append((int(i), rating, ts))
        train_dict[u] = interactions

    # Time execution
    t0 = time.perf_counter()
    scores = compute_temporal_acceleration(train_dict, time_window_hours=24)
    elapsed = time.perf_counter() - t0

    # Sub-second execution requirement (typically < 30ms)
    assert elapsed < 0.50, f"compute_temporal_acceleration took {elapsed:.4f}s, expected < 0.50s"
    assert len(scores) > 0

    # Verify score bounds [0.0, 10.0]
    for key, val in scores.items():
        assert 0.0 <= val <= 10.0
        assert isinstance(val, float)

    # Item 0 had a massive burst within 1 hour: acceleration should be > 1.0 for item 0
    item_0_scores = [val for (u, i), val in scores.items() if i == 0]
    assert len(item_0_scores) > 0
    assert max(item_0_scores) > 1.0

    # Test sparse/empty items
    sparse_dict = {0: [(1, 5, 1000), (2, 5, 2000)]}
    sparse_scores = compute_temporal_acceleration(sparse_dict)
    assert sparse_scores[(0, 1)] == 1.0
    assert sparse_scores[(0, 2)] == 1.0

    # Test empty dict
    assert compute_temporal_acceleration({}) == {}


def test_compute_polarity_skew_vectorization_and_speed():
    """
    Verifies that compute_polarity_skew is vectorized with per-item prefix sums,
    runs in sub-second time, and accurately identifies polar ratings.
    """
    num_users = 100
    num_items = 50
    rng = np.random.RandomState(42)

    train_dict = {}
    base_ts = 1700000000
    for u in range(num_users):
        interactions = []
        items = rng.choice(num_items, size=20, replace=False)
        for i in items:
            # Item 0 has strictly extreme ratings (1 and 5)
            if i == 0:
                rating = int(rng.choice([1, 5]))
            # Item 1 has strictly neutral ratings (3)
            elif i == 1:
                rating = 3
            else:
                rating = int(rng.choice([1, 2, 3, 4, 5]))
            ts = base_ts + int(rng.randint(0, 3600 * 24 * 10))
            interactions.append((int(i), rating, ts))
        train_dict[u] = interactions

    t0 = time.perf_counter()
    skew_scores = compute_polarity_skew(train_dict, time_window_hours=24)
    elapsed = time.perf_counter() - t0

    assert elapsed < 0.50, f"compute_polarity_skew took {elapsed:.4f}s, expected < 0.50s"
    assert len(skew_scores) > 0

    # Verify score bounds [0.0, 1.0]
    for key, val in skew_scores.items():
        assert 0.0 <= val <= 1.0
        assert isinstance(val, float)

    # Item 0 should have high polarity skew (all ratings are 1 or 5)
    item_0_skew = [val for (u, i), val in skew_scores.items() if i == 0]
    assert len(item_0_skew) > 0
    assert np.mean(item_0_skew) == 1.0

    # Item 1 should have 0.0 polarity skew (all ratings are 3)
    item_1_skew = [val for (u, i), val in skew_scores.items() if i == 1]
    assert len(item_1_skew) > 0
    assert np.mean(item_1_skew) == 0.0

    # Test small items (<= 2 interactions) defaults to 0.5
    small_dict = {0: [(1, 5, 100)]}
    small_scores = compute_polarity_skew(small_dict)
    assert small_scores[(0, 1)] == 0.5

    # Test empty dict
    assert compute_polarity_skew({}) == {}


def test_sweep_omega_sensitivity_2d_matrix_speed_and_correctness():
    """
    Verifies that sweep_omega_sensitivity executes via 2D matrix broadcasting
    in < 150ms on 40k+ interactions, and matches expected F1/ROC-AUC metrics.
    """
    n_interactions = 42000
    rng = np.random.RandomState(42)

    temporal_accel = {}
    polarity_skew = {}
    semantic_sim = {}
    ground_truth = {}

    for idx in range(n_interactions):
        key = (idx // 100, idx % 100)
        is_attack = bool(rng.rand() < 0.15)
        ground_truth[key] = 0 if is_attack else 1

        if is_attack:
            temporal_accel[key] = float(rng.uniform(3.0, 9.0))
            polarity_skew[key] = float(rng.uniform(0.75, 1.0))
        else:
            temporal_accel[key] = float(rng.uniform(1.0, 2.0))
            polarity_skew[key] = float(rng.uniform(0.2, 0.6))
        semantic_sim[key] = 0.0

    t0 = time.perf_counter()
    results = sweep_omega_sensitivity(
        temporal_accel=temporal_accel,
        polarity_skew=polarity_skew,
        semantic_sim=semantic_sim,
        ground_truth_labels=ground_truth,
        steps=11
    )
    elapsed = time.perf_counter() - t0

    # Dispatch requirement: < 150ms on 40k+ interactions
    assert elapsed < 0.150, f"sweep_omega_sensitivity took {elapsed:.4f}s, expected < 0.150s (>50x speedup)"
    assert len(results) == 11

    # Verify result structure and metrics
    for step_res in results:
        assert "omega_1_temporal" in step_res
        assert "omega_2_polarity" in step_res
        assert "Denoising_F1" in step_res
        assert "ROC_AUC" in step_res
        w1 = step_res["omega_1_temporal"]
        w2 = step_res["omega_2_polarity"]
        assert abs(w1 + w2 - 1.0) < 1e-3
        assert 0.0 <= step_res["Denoising_F1"] <= 1.0
        assert 0.0 <= step_res["ROC_AUC"] <= 1.0

    # Best ROC-AUC should be high because signals clearly separate attack vs benign
    aucs = [r["ROC_AUC"] for r in results]
    assert max(aucs) > 0.85

    # Test empty ground truth
    assert sweep_omega_sensitivity({}, {}, {}, {}) == []


def test_lightgcn_forward_pass_caching():
    """
    Verifies that LightGCN reuses cached embeddings during eval mode in forward(),
    avoiding redundant full-graph 3-layer GCN convolutions.
    """
    num_users = 20
    num_items = 30
    model = LightGCN(num_users=num_users, num_items=num_items, embedding_dim=16, num_layers=2)

    # Build adjacency
    train_dict = {
        0: [(0, 5, 100), (1, 4, 101)],
        1: [(1, 3, 102), (2, 2, 103)],
        2: [(0, 1, 104), (3, 5, 105)]
    }
    model.set_adjacency(train_dict)
    assert model._cached_user_embs is None
    assert model._cached_item_embs is None

    # Preallocated rating weights buffer check
    assert hasattr(model, "rating_weights")
    assert "rating_weights" in dict(model.named_buffers())
    assert torch.equal(model.rating_weights, torch.arange(1, 6, dtype=torch.float32))

    # Switch to eval mode
    model.eval()

    user_batch = torch.tensor([0, 1, 2], dtype=torch.int64)
    item_batch = torch.tensor([0, 1, 2], dtype=torch.int64)

    # First forward call in eval mode computes and populates cache
    out1 = model(user_batch, item_batch)
    assert model._cached_user_embs is not None
    assert model._cached_item_embs is not None
    cached_u = model._cached_user_embs
    cached_i = model._cached_item_embs

    # Spy on compute_all_embeddings to confirm it is NOT called again
    with patch.object(model, "compute_all_embeddings", wraps=model.compute_all_embeddings) as mock_compute:
        out2 = model(user_batch, item_batch)
        mock_compute.assert_not_called()
        assert torch.allclose(out1, out2)

    # Verify score_items also uses cache without recomputing
    with patch.object(model, "compute_all_embeddings", wraps=model.compute_all_embeddings) as mock_compute:
        scores = model.score_items(0, [0, 1])
        mock_compute.assert_not_called()
        assert len(scores) == 2

    # Switch to training mode: cache must be invalidated
    model.train()
    assert model._cached_user_embs is None
    assert model._cached_item_embs is None


def test_score_candidates_batch_across_all_models():
    """
    Verifies score_candidates_batch across LightGCN, SimGCL, SGL, NeuMF, and VaeCF:
    - Proper [B, C] output dimensions
    - Equivalence with individual score_items calls
    - 1D candidate matrix broadcast handling
    - Empty candidate matrix handling
    - rating_weights buffer registered
    """
    num_users = 15
    num_items = 25
    num_classes = 5

    models = [
        LightGCN(num_users=num_users, num_items=num_items, embedding_dim=16, num_layers=2, num_classes=num_classes),
        SimGCL(num_users=num_users, num_items=num_items, embedding_dim=16, num_layers=2, num_classes=num_classes),
        SGL(num_users=num_users, num_items=num_items, embedding_dim=16, num_layers=2, num_classes=num_classes),
        NeuMF(num_users=num_users, num_items=num_items, embedding_dim=16, num_classes=num_classes),
        VaeCF(num_users=num_users, num_items=num_items, embedding_dim=16, latent_dim=8, num_classes=num_classes),
    ]

    train_dict = {
        u: [(i, 5, 1000) for i in range(min(5, num_items))]
        for u in range(num_users)
    }

    test_users = torch.tensor([0, 2, 4], dtype=torch.int64)
    # 2D candidate matrix: 3 users, 6 candidates each
    cand_matrix = torch.tensor([
        [1, 3, 5, 7, 9, 11],
        [2, 4, 6, 8, 10, 12],
        [0, 1, 2, 3, 4, 5],
    ], dtype=torch.int64)

    for model in models:
        # 1. Buffer preallocation check
        assert hasattr(model, "rating_weights"), f"{model.__class__.__name__} missing rating_weights attribute"
        assert "rating_weights" in dict(model.named_buffers()), f"{model.__class__.__name__} rating_weights not registered as buffer"
        assert torch.equal(model.rating_weights, torch.arange(1, num_classes + 1, dtype=torch.float32))

        # Adjacency for GNN models
        if hasattr(model, "set_adjacency"):
            model.set_adjacency(train_dict)

        model.eval()

        # 2. Batched scoring
        batched_scores = model.score_candidates_batch(test_users, cand_matrix)
        assert batched_scores.shape == (3, 6), f"{model.__class__.__name__} output shape {batched_scores.shape} != (3, 6)"
        assert not torch.isnan(batched_scores).any()

        # 3. Equivalence with score_items
        for row_idx, u_id in enumerate(test_users.tolist()):
            items_list = cand_matrix[row_idx].tolist()
            single_scores = model.score_items(u_id, items_list)
            assert torch.allclose(batched_scores[row_idx], single_scores, atol=1e-5), (
                f"{model.__class__.__name__} batch score mismatch for user {u_id}"
            )

        # 4. 1D candidate matrix broadcasting test: [C] -> [B, C]
        cands_1d = torch.tensor([2, 5, 8], dtype=torch.int64)
        broadcast_scores = model.score_candidates_batch(test_users, cands_1d)
        assert broadcast_scores.shape == (3, 3)

        # 5. Empty candidate matrix test: [B, 0]
        empty_matrix = torch.empty((3, 0), dtype=torch.int64)
        empty_scores = model.score_candidates_batch(test_users, empty_matrix)
        assert empty_scores.shape == (3, 0)


def test_reliability_fusion_vectorization_and_caching():
    """
    Verifies that compute_R_RRFN reuses cached embeddings on LightGCN
    and fuse_reliability_scores runs in vectorized mode with exact bounds.
    """
    num_users = 20
    num_items = 30
    model = LightGCN(num_users=num_users, num_items=num_items, embedding_dim=16, num_layers=2)
    train_dict = {
        0: [(1, 4, 100)],
        1: [(2, 5, 101)],
        2: [(3, 2, 102)],
    }
    model.set_adjacency(train_dict)

    # Build DataLoader with 10 batches
    u_data = torch.randint(0, num_users, (100,))
    i_data = torch.randint(0, num_items, (100,))
    y_data = torch.randint(1, 6, (100,))
    dataset = TensorDataset(u_data, i_data, y_data)
    loader = DataLoader(dataset, batch_size=10, shuffle=False)

    # compute_R_RRFN should populate model cache
    r_rrfn = compute_R_RRFN(model, loader)
    assert len(r_rrfn) > 0
    assert model._cached_user_embs is not None
    assert model._cached_item_embs is not None

    for k, v in r_rrfn.items():
        assert 0.05 <= v <= 1.0

    # fuse_reliability_scores with vectorized NumPy arrays
    r_dummy = {k: 0.8 for k in r_rrfn.keys()}
    fused = fuse_reliability_scores(r_rrfn, r_dummy, r_dummy, alpha=0.5, beta=0.3, gamma=0.2, min_weight=0.02)
    assert len(fused) == len(r_rrfn)
    for k, v in fused.items():
        assert 0.02 <= v <= 1.0
        assert isinstance(v, float)

    # Empty inputs
    assert fuse_reliability_scores({}, {}, {}) == {}
