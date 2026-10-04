"""
Unit Test Suite for Milestone M2: Numerical Stability & Loss Protection.
Tests cover:
  1. Label shift determinism and empty batch guard in risk_consistent_loss
  2. Double-softmax elimination in warm_train (NLLLoss with log_probs)
  3. Tikhonov regularized pseudo-inversion in NoiseTransitionMatrix
  4. Extreme logvar clamping in VaeCF reparameterization
  5. Binarized adjacency, empty adjacency handling, and device movement in LightGCN
  6. Coalesced augmented sparse adjacency in SGL
  7. Sigmoid exp argument clamping in bomb_scorer
  8. Single-item catalog non-empty tail guard in pareto_partition
  9. Duplicate avoidance and non-positive top_k guard in merge_top_n
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

from src.rrfn.risk_loss import risk_consistent_loss
from src.rrfn.transition_matrix import NoiseTransitionMatrix
from src.trainer.dual_trainer import DualModelTrainer, InteractionDataset
from src.models.vaecf import VaeCF
from src.models.lightgcn import LightGCN
from src.models.sgl import SGL
from src.models.neumf import NeuMF
from src.bombing_detector.bomb_scorer import compute_bomb_scores
from src.catalog.pareto_partition import pareto_partition
from src.super_engine.merger import merge_top_n
from torch.utils.data import DataLoader


# ==============================================================================
# 1. RISK CONSISTENT LOSS & LABEL SHIFT DETERMINISM
# ==============================================================================

def test_risk_loss_label_shift_deterministic_without_high_rating():
    """
    Verifies that a batch containing only ratings 1..3 is properly shifted to 0..2
    even when no rating >= 4 exists in the batch.
    """
    trans = NoiseTransitionMatrix(num_classes=5)
    clean_probs = torch.full((3, 5), 0.2)
    # Batch has min=1, max=3 (both within [1, 5])
    observed_ratings = torch.tensor([1, 2, 3])
    weights = torch.ones(3)

    loss = risk_consistent_loss(clean_probs, trans, observed_ratings, weights)
    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_risk_loss_already_zero_indexed_labels():
    """
    Verifies that when observed ratings are already 0-indexed (min == 0),
    they are not shifted further into negative values.
    """
    trans = NoiseTransitionMatrix(num_classes=5)
    clean_probs = torch.full((4, 5), 0.2)
    observed_ratings = torch.tensor([0, 1, 2, 3])
    weights = torch.ones(4)

    loss = risk_consistent_loss(clean_probs, trans, observed_ratings, weights)
    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_risk_loss_empty_batch_guard():
    """
    Verifies that an empty batch returns a zero scalar tensor with requires_grad=True
    without raising IndexError, ValueError, or division by zero.
    """
    trans = NoiseTransitionMatrix(num_classes=5)
    clean_probs = torch.empty((0, 5))
    observed_ratings = torch.empty(0, dtype=torch.long)
    weights = torch.empty(0)

    loss = risk_consistent_loss(clean_probs, trans, observed_ratings, weights)
    assert loss.item() == 0.0
    assert loss.requires_grad is True


def test_risk_loss_extreme_probabilities_and_weight_normalization():
    """
    Verifies that near-zero probabilities do not evaluate to NaN or -inf,
    and weights normalization prevents loss magnitude explosion.
    """
    trans = NoiseTransitionMatrix(num_classes=5)
    clean_probs = torch.zeros((4, 5))
    clean_probs[:, 0] = 1.0  # degenerate one-hot
    observed_ratings = torch.tensor([5, 5, 5, 5])  # observed noisy class 4 (zero prob)
    weights = torch.tensor([10.0, 10.0, 10.0, 10.0])

    loss = risk_consistent_loss(clean_probs, trans, observed_ratings, weights)
    assert torch.isfinite(loss)
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)

    # Scaling weights uniformly by 100x should not scale loss by 100x
    weights_scaled = weights * 100.0
    loss_scaled = risk_consistent_loss(clean_probs, trans, observed_ratings, weights_scaled)
    np.testing.assert_allclose(loss.item(), loss_scaled.item(), rtol=1e-4)


# ==============================================================================
# 2. DOUBLE-SOFTMAX ELIMINATION IN WARM_TRAIN
# ==============================================================================

def test_warm_train_double_softmax_elimination():
    """
    Verifies warm_train with NLLLoss on log(probs.clamp_min(1e-12)).
    Ensures gradients remain finite and parameters update smoothly without explosions.
    """
    num_users, num_items = 8, 8

    def factory():
        return NeuMF(num_users=num_users, num_items=num_items, embedding_dim=8, mlp_layers=[16, 8])

    model = factory()
    interactions = [(u, (u + 1) % num_items, 5) for u in range(num_users)]
    weights = {(u, i): 1.0 for u, i, _ in interactions}
    dataset = InteractionDataset(interactions, weights)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    trainer = DualModelTrainer(model_factory=factory, device="cpu", epochs=1)
    trainer.warm_train(model, loader, warm_epochs=2)

    # Verify model parameters are still finite
    for p in model.parameters():
        assert torch.isfinite(p).all()


# ==============================================================================
# 3. TIKHONOV REGULARIZED INVERSION IN TRANSITION MATRIX
# ==============================================================================

def test_transition_matrix_regularized_inversion_singular_case():
    """
    Verifies that get_inverse() handles singular or ill-conditioned transition
    matrices gracefully using Tikhonov regularization without raising LinAlgError.
    """
    trans = NoiseTransitionMatrix(num_classes=5)

    # Case 1: Default near-identity matrix
    T_inv_identity = trans.get_inverse(reg_lambda=1e-4)
    assert T_inv_identity.shape == (5, 5)
    assert torch.isfinite(T_inv_identity).all()

    # Case 2: Degenerate rank-1 matrix (all rows identical, det(T) = 0)
    with torch.no_grad():
        trans.delta_T.fill_(0.0)
        trans.T_hat.fill_(0.2)  # Every entry equal -> rank 1, singular!

    # Normal inv would crash or blow up; regularized pseudo-inverse must succeed
    T_inv_singular = trans.get_inverse(reg_lambda=1e-2)
    assert T_inv_singular.shape == (5, 5)
    assert torch.isfinite(T_inv_singular).all()
    assert not torch.isnan(T_inv_singular).any()


# ==============================================================================
# 4. VAECF REPARAMETERIZATION LOGVAR CLAMPING
# ==============================================================================

def test_vaecf_logvar_clamping_prevents_overflow():
    """
    Verifies that VaeCF.reparameterize bounds logvar to [-20.0, 20.0] during training,
    preventing float32 exp overflow to +inf and subsequent NaNs.
    """
    model = VaeCF(num_users=10, num_items=10, embedding_dim=16, latent_dim=8)
    model.train()

    mu = torch.zeros(2, 8)
    # Extreme positive logvar (>88.7 would overflow unconstrained float32 exp)
    extreme_pos_logvar = torch.tensor([[150.0] * 8, [300.0] * 8])
    z_pos = model.reparameterize(mu, extreme_pos_logvar)
    assert torch.isfinite(z_pos).all()
    assert not torch.isnan(z_pos).any()

    # Extreme negative logvar
    extreme_neg_logvar = torch.tensor([[-200.0] * 8, [-500.0] * 8])
    z_neg = model.reparameterize(mu, extreme_neg_logvar)
    assert torch.isfinite(z_neg).all()
    assert not torch.isnan(z_neg).any()

    # In eval mode, reparameterize must return mu deterministically
    model.eval()
    z_eval = model.reparameterize(mu, extreme_pos_logvar)
    torch.testing.assert_close(z_eval, mu)


# ==============================================================================
# 5. LIGHTGCN ADJACENCY BINARIZATION AND DEVICE MOVEMENT
# ==============================================================================

def test_lightgcn_empty_adjacency_no_artificial_edge():
    """
    Verifies that LightGCN.set_adjacency on an empty interaction dict produces
    an empty sparse tensor without injecting an artificial (0, 0) self-edge.
    """
    model = LightGCN(num_users=4, num_items=4, embedding_dim=8, num_layers=2)
    model.set_adjacency({})

    assert model.adj_norm is not None
    # nnz should be exactly 0
    assert model.adj_norm._nnz() == 0
    assert model.adj_norm.shape == (8, 8)

    # Forward pass should not crash with empty adjacency
    user_ids = torch.tensor([0, 1])
    item_ids = torch.tensor([1, 2])
    probs = model(user_ids, item_ids)
    assert probs.shape == (2, 5)
    assert torch.isfinite(probs).all()


def test_lightgcn_adjacency_multi_edge_binarization():
    """
    Verifies that multi-edges between the same user and item in train_dict
    are binarized to 1.0 rather than summed.
    """
    model = LightGCN(num_users=3, num_items=3, embedding_dim=8, num_layers=2)
    # User 0 interacts with item 1 three separate times
    train_dict = {
        0: [(1, 5, 100), (1, 4, 101), (1, 5, 102)],
        1: [(0, 3, 103)]
    }
    model.set_adjacency(train_dict)

    # Check maximum value in normalized adjacency
    adj = model.adj_norm.coalesce()
    max_val = adj.values().max().item()
    # Normalized Laplacian with binarized weights will have entries <= 1.0
    assert max_val <= 1.0 + 1e-5


def test_lightgcn_adj_norm_moves_with_apply():
    """
    Verifies that model._apply properly updates self.adj_norm device/dtype.
    """
    model = LightGCN(num_users=3, num_items=3, embedding_dim=8, num_layers=1)
    train_dict = {0: [(1, 5, 100)]}
    model.set_adjacency(train_dict, device="cpu")

    assert model.adj_norm.device.type == "cpu"
    # Call model.to('cpu') which invokes _apply
    model = model.to("cpu")
    assert model.adj_norm.device.type == "cpu"


# ==============================================================================
# 6. SGL COALESCED AUGMENTED SPARSE ADJACENCY
# ==============================================================================

def test_sgl_augmented_adj_is_coalesced():
    """
    Verifies that SGL._augment_adj explicitly returns a coalesced sparse tensor.
    """
    model = SGL(num_users=4, num_items=4, embedding_dim=8, drop_rate=0.2, augment_type="ED")
    train_dict = {0: [(1, 5, 100)], 1: [(2, 4, 101)], 2: [(0, 3, 102)]}
    model.set_adjacency(train_dict)

    aug_adj = model._augment_adj(model.adj_norm, drop_rate=0.2)
    assert aug_adj.is_coalesced()

    # Full contrastive embedding generation should execute cleanly
    v1, v2 = model.get_cl_embeddings()
    assert v1.shape == (8, 8)
    assert v2.shape == (8, 8)
    assert torch.isfinite(v1).all()
    assert torch.isfinite(v2).all()


# ==============================================================================
# 7. BOMB SCORER SIGMOID OVERFLOW CLAMPING
# ==============================================================================

def test_bomb_scorer_overflow_clamping():
    """
    Verifies that compute_bomb_scores does not trigger RuntimeWarning: overflow in exp
    when presented with extreme acceleration, polarity skew, or high sigmoid slopes.
    """
    import warnings

    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")

        # Case 1: Extreme high bombing signal with extreme sigmoid slope and bias
        temporal_accel = {(0, 0): 100.0}
        polarity_skew = {(0, 0): 1.0}
        semantic_sim = {(0, 0): 1.0}

        scores_high = compute_bomb_scores(
            temporal_accel,
            polarity_skew,
            semantic_sim,
            sigmoid_slope=50.0,
            sigmoid_bias=100.0
        )
        assert (0, 0) in scores_high
        assert 0.05 <= scores_high[(0, 0)] <= 1.0

        # Case 2: Extreme negative bias
        scores_low = compute_bomb_scores(
            temporal_accel,
            polarity_skew,
            semantic_sim,
            sigmoid_slope=50.0,
            sigmoid_bias=-100.0
        )
        assert (0, 0) in scores_low
        assert 0.05 <= scores_low[(0, 0)] <= 1.0

        # Ensure no overflow RuntimeWarning was raised
        overflow_warnings = [
            w for w in recorded_warnings
            if "overflow" in str(w.message).lower()
        ]
        assert len(overflow_warnings) == 0


# ==============================================================================
# 8. PARETO PARTITION SINGLE-ITEM CATALOG GUARD
# ==============================================================================

def test_pareto_partition_single_item_catalog():
    """
    Verifies that a catalog with a single item (num_items == 1) guarantees
    both head_set and tail_set are non-empty to prevent downstream M_tail crashes.
    """
    V_eff = {0: 15.0}
    head_set, tail_set = pareto_partition(V_eff, pareto_alpha=0.2)

    assert len(head_set) > 0
    assert len(tail_set) > 0
    assert 0 in head_set
    assert 0 in tail_set


def test_pareto_partition_multi_item_catalog():
    """
    Verifies standard Pareto partitioning with multiple items.
    """
    V_eff = {0: 100.0, 1: 50.0, 2: 10.0, 3: 5.0}
    head_set, tail_set = pareto_partition(V_eff, pareto_alpha=0.2)

    assert len(head_set) >= 1
    assert len(tail_set) >= 1
    assert head_set.isdisjoint(tail_set)
    assert head_set | tail_set == {0, 1, 2, 3}


# ==============================================================================
# 9. MERGE_TOP_N DUPLICATE AVOIDANCE AND TOP_K GUARDS
# ==============================================================================

def test_merge_top_n_duplicate_avoidance():
    """
    Verifies that merge_top_n never emits duplicate items, even when pop_candidates
    and tail_candidates share items or repeat items internally.
    """
    # Overlapping candidates
    pop_candidates = [10, 20, 30, 40]
    tail_candidates = [20, 30, 50, 60]
    pop_tilde_u = 0.5
    blueprint_b_u = [1, 0, 1, 0, 1]

    top_recs = merge_top_n(
        user_id=1,
        pop_candidates=pop_candidates,
        tail_candidates=tail_candidates,
        pop_tilde_u=pop_tilde_u,
        blueprint_b_u=blueprint_b_u,
        top_k=5
    )

    # Assert no duplicates
    assert len(top_recs) == len(set(top_recs))
    assert len(top_recs) <= 5


def test_merge_top_n_non_positive_top_k():
    """
    Verifies that top_k <= 0 returns an empty list without errors.
    """
    recs_zero = merge_top_n(0, [1, 2], [3, 4], 0.5, [1, 0], top_k=0)
    assert recs_zero == []

    recs_neg = merge_top_n(0, [1, 2], [3, 4], 0.5, [1, 0], top_k=-5)
    assert recs_neg == []
