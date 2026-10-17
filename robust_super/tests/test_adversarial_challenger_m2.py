"""
robust_super/tests/test_adversarial_challenger_m2.py
---------------------------------------------------
Adversarial Stress Test Suite for Milestone M2 Baselines:
1. Numerical Bounds on Predicted Scores across all 4 baselines (LightGCN, SGL, TCE_SGDL, TrustGRS)
   under extreme weights, large batches, broadcasted candidates, and degenerate inputs.
2. Degenerate Transition Matrix Inversion in TCE_SGDL:
   Singular, zero, extreme magnitude, rank-1, rank-deficient, and near-zero/negative lambda regularizations.
3. Extreme Shilling Injection (rho = 0.50, 50% synthetic bots):
   Verification of bot edge attenuation, non-divergence, finite loss, and stable training dynamics in TrustGRS.
4. Sparse COO Coalescing in TrustGRS:
   Ensuring model.adj_norm.is_coalesced() is True under diverse topologies: empty, disconnected, duplicate edges,
   and extreme shill attacks.
"""

import copy
import pytest
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from src.models.lightgcn import LightGCN
from src.models.sgl import SGL
from src.models.tce_sgdl import TCE_SGDL
from src.models.trust_grs import TrustGRS
from src.trainer.baseline_trainer import BaselineTrainer
from src.trainer.dual_trainer import build_data_loader
from src.data.preprocessor import preprocess_dataset
from src.data.attack_simulator import AttackSimulator


# ==============================================================================
# 1. NUMERICAL BOUNDS ON PREDICTED SCORES (STRESS TESTS)
# ==============================================================================

@pytest.mark.parametrize("model_cls", [LightGCN, SGL, TCE_SGDL, TrustGRS])
def test_score_candidates_batch_bounds_and_finiteness(model_cls):
    """
    Verify score_candidates_batch outputs are finite and within Likert range [1.0, 5.0]
    across standard, extreme, and perturbed embedding regimes.
    """
    num_users, num_items = 20, 40
    if model_cls == SGL:
        model = model_cls(num_users=num_users, num_items=num_items, embedding_dim=16, num_layers=2)
    elif model_cls == TCE_SGDL:
        model = model_cls(num_users=num_users, num_items=num_items, embedding_dim=16, num_layers=2)
    elif model_cls == TrustGRS:
        model = model_cls(num_users=num_users, num_items=num_items, embedding_dim=16, num_layers=2)
    else:
        model = model_cls(num_users=num_users, num_items=num_items, embedding_dim=16, num_layers=2)

    train_dict = {u: [((u + i) % num_items, 4, 100) for i in range(3)] for u in range(num_users)}
    model.set_adjacency(train_dict, device="cpu")

    # Standard batch
    user_ids = torch.arange(num_users)
    candidates_2d = torch.randint(0, num_items, (num_users, 15))
    scores_2d = model.score_candidates_batch(user_ids, candidates_2d)
    assert scores_2d.shape == (num_users, 15)
    assert torch.isfinite(scores_2d).all()
    assert (scores_2d >= 1.0 - 1e-5).all(), f"Min score {scores_2d.min()} < 1.0"
    assert (scores_2d <= 5.0 + 1e-5).all(), f"Max score {scores_2d.max()} > 5.0"

    # 1D broadcasted candidate matrix
    candidates_1d = torch.randint(0, num_items, (10,))
    scores_1d = model.score_candidates_batch(user_ids, candidates_1d)
    assert scores_1d.shape == (num_users, 10)
    assert torch.isfinite(scores_1d).all()
    assert (scores_1d >= 1.0 - 1e-5).all()
    assert (scores_1d <= 5.0 + 1e-5).all()

    # Empty candidate tensor
    scores_empty_c = model.score_candidates_batch(user_ids, torch.empty((num_users, 0), dtype=torch.long))
    assert scores_empty_c.shape == (num_users, 0)

    # Empty user tensor
    scores_empty_u = model.score_candidates_batch(torch.empty(0, dtype=torch.long), torch.empty((0, 5), dtype=torch.long))
    assert scores_empty_u.shape == (0, 5)


@pytest.mark.parametrize("model_cls", [LightGCN, SGL, TCE_SGDL, TrustGRS])
def test_score_candidates_batch_extreme_perturbed_embeddings(model_cls):
    """
    Adversarial test: perturb user and item embeddings with extreme magnitudes (+-1000)
    to check for softmax overflow or numerical collapse.
    """
    num_users, num_items = 10, 20
    model = model_cls(num_users=num_users, num_items=num_items, embedding_dim=16, num_layers=2)
    train_dict = {u: [(u % num_items, 3, 100)] for u in range(num_users)}
    model.set_adjacency(train_dict, device="cpu")

    # Inject extreme values into embeddings
    with torch.no_grad():
        model.user_embedding.weight[0].fill_(1000.0)
        model.user_embedding.weight[1].fill_(-1000.0)
        model.item_embedding.weight[0].fill_(1000.0)
        model.item_embedding.weight[1].fill_(-1000.0)

    model.eval()
    user_ids = torch.tensor([0, 1, 2])
    candidates = torch.tensor([[0, 1], [0, 1], [0, 1]])
    scores = model.score_candidates_batch(user_ids, candidates)

    assert torch.isfinite(scores).all(), "Scores contain NaN or Inf under extreme embeddings"
    assert (scores >= 1.0 - 1e-5).all()
    assert (scores <= 5.0 + 1e-5).all()


# ==============================================================================
# 2. DEGENERATE TRANSITION MATRIX INVERSION IN TCE_SGDL (STRESS TESTS)
# ==============================================================================

def test_tce_sgdl_degenerate_matrix_inversion_singular_and_zeros():
    """
    Adversarially test Tikhonov regularized pseudo-inversion under:
    - Pure singular matrix (rank 1)
    - Zero matrix (rank 0)
    - Extreme magnitude slack matrix delta_T
    - Near-zero and zero regularization parameter lambda
    """
    model = TCE_SGDL(num_users=10, num_items=10, num_classes=5)

    # 1. Singular matrix: all rows identical (rank 1)
    with torch.no_grad():
        model.transition.T_hat.fill_(0.2)
        model.transition.delta_T.zero_()
    
    T = model.get_transition_matrix()
    assert torch.linalg.matrix_rank(T).item() == 1  # Verify rank 1
    T_inv_singular = model.get_transition_inverse(reg_lambda=1e-4)
    assert T_inv_singular.shape == (5, 5)
    assert torch.isfinite(T_inv_singular).all()
    assert not torch.isnan(T_inv_singular).any()

    # 2. Extreme slack parameters: delta_T has huge values (e.g., 1000.0, -1000.0)
    with torch.no_grad():
        model.transition.delta_T.copy_(torch.randn(5, 5) * 500.0)
    
    T_extreme = model.get_transition_matrix()
    assert torch.isfinite(T_extreme).all()
    row_sums = T_extreme.sum(dim=-1)
    np.testing.assert_allclose(row_sums.detach().numpy(), np.ones(5), atol=1e-4)
    
    T_inv_extreme = model.get_transition_inverse(reg_lambda=1e-3)
    assert torch.isfinite(T_inv_extreme).all()

    # 3. Near-zero and zero lambda: guard must ensure reg_lambda >= 1e-8
    with torch.no_grad():
        model.transition.T_hat.copy_(torch.eye(5))
        model.transition.delta_T.zero_()
    
    T_inv_zero_lam = model.get_transition_inverse(reg_lambda=0.0)
    assert torch.isfinite(T_inv_zero_lam).all()
    T_curr = model.get_transition_matrix()
    diff_from_eye = torch.norm(T_curr @ T_inv_zero_lam - torch.eye(5)).item()
    assert diff_from_eye < 1e-3

    # 4. Stress direct pseudo-inversion helper with arbitrary rank-0 / zero matrix
    # Using solver directly with T_zero
    T_zero = torch.zeros(5, 5)
    reg_lambda = max(float(0.0), 1e-8)
    I = torch.eye(5)
    T_inv_zero = torch.linalg.solve(T_zero.T @ T_zero + reg_lambda * I, T_zero.T)
    assert torch.all(T_inv_zero == 0.0)
    assert torch.isfinite(T_inv_zero).all()


# ==============================================================================
# 3. EXTREME SHILLING INJECTION (rho = 0.50) (STRESS TESTS)
# ==============================================================================

def test_trust_grs_extreme_shilling_rho_50():
    """
    Adversarial test: Inject extreme 50% shilling attack (rho = 0.50).
    Verify TrustGRS:
    1. Attenuates bot edge trust significantly.
    2. Builds valid normalized Laplacian without NaN or Inf.
    3. Executes training loop for 2 epochs without loss divergence or NaN.
    """
    num_auth_users = 40
    num_items = 30
    num_bot_users = int(num_auth_users * 0.50)  # 20 bot users
    total_users = num_auth_users + num_bot_users

    # Authentic users: natural behavior across items with variance
    train_dict = {}
    for u in range(num_auth_users):
        interactions = []
        for i in range(5):
            item = (u * 3 + i) % num_items
            rating = int(2 + (u + i) % 4)  # Ratings in 2..5
            interactions.append((item, rating, 1000 + i))
        train_dict[u] = interactions

    # Target item pushed by bots
    target_items = [num_items - 1, num_items - 2]
    # Filler items: popular head items 0, 1, 2
    filler_items = [0, 1, 2]

    # Bots: extreme shilling, 5-star ratings on filler and target items
    for bot_idx in range(num_bot_users):
        bot_u = num_auth_users + bot_idx
        bot_interactions = []
        for fi in filler_items:
            bot_interactions.append((fi, 5, 2000))
        for ti in target_items:
            bot_interactions.append((ti, 5, 2000))
        train_dict[bot_u] = bot_interactions

    model = TrustGRS(num_users=total_users, num_items=num_items, embedding_dim=16, num_layers=2)
    trust_weights = model.compute_trust_weights(train_dict)

    assert len(trust_weights) > 0
    auth_weights = [trust_weights[(u, i)] for u in range(num_auth_users) for i, _, _ in train_dict[u]]
    bot_weights = [trust_weights[(u, i)] for u in range(num_auth_users, total_users) for i, _, _ in train_dict[u]]

    mean_auth_w = float(np.mean(auth_weights))
    mean_bot_w = float(np.mean(bot_weights))

    # Attenuation check: bot edge trust must be attenuated relative to authentic users
    assert mean_bot_w < mean_auth_w, f"Bot weight {mean_bot_w:.4f} >= Auth weight {mean_auth_w:.4f}"
    # Verify bot profiles have degraded user trust
    bot_user_trusts = [model.user_trust_scores[u] for u in range(num_auth_users, total_users)]
    assert np.mean(bot_user_trusts) < 0.5, f"Bots not flagged: mean bot user trust {np.mean(bot_user_trusts):.4f}"

    # Set adjacency under 50% attack
    model.set_adjacency(train_dict, device="cpu")
    assert model.adj_norm.is_coalesced()
    assert torch.isfinite(model.adj_norm.values()).all()

    # Train under extreme shilling for 2 epochs
    weights_dict = {(u, i): trust_weights.get((u, i), 1.0) for u in train_dict for i, _, _ in train_dict[u]}
    val_dict = {u: ((u + 1) % num_items, 4, 3000) for u in range(num_auth_users)}
    loader = build_data_loader(train_dict, weights_dict, batch_size=32)

    trainer = BaselineTrainer(device="cpu", lr=0.01, epochs=2, early_stopping_patience=2, baseline_type="trust_grs")
    trained_model = trainer.train(model, loader, val_dict, train_dict=train_dict)

    assert len(trained_model.loss_history) == 2
    for epoch_loss in trained_model.loss_history:
        assert np.isfinite(epoch_loss), f"Loss diverged to {epoch_loss}"
        assert epoch_loss > 0.0


# ==============================================================================
# 4. SPARSE COO COALESCING IN TRUST-GRS (STRESS TESTS)
# ==============================================================================

@pytest.mark.parametrize("scenario", [
    "empty",
    "disconnected",
    "duplicate_edges",
    "extreme_density"
])
def test_trust_grs_sparse_coo_coalesced_invariants(scenario):
    """
    Stress-test sparse COO coalescing in TrustGRS across abnormal graph topologies:
    1. Empty graph
    2. Disconnected graph with isolated nodes
    3. Multi-edges / duplicate interactions
    4. Extreme dense graph
    """
    num_users, num_items = 25, 35
    model = TrustGRS(num_users=num_users, num_items=num_items, embedding_dim=8, num_layers=2)

    if scenario == "empty":
        train_dict = {}
    elif scenario == "disconnected":
        # Only 2 users interact, rest are completely isolated
        train_dict = {
            0: [(0, 4, 100), (1, 5, 101)],
            1: [(1, 3, 102), (2, 4, 103)]
        }
    elif scenario == "duplicate_edges":
        # Same user-item pair repeated multiple times with different ratings
        train_dict = {
            0: [(5, 2, 100), (5, 5, 200), (5, 3, 300)],
            1: [(5, 4, 150), (10, 4, 250), (10, 5, 350)]
        }
    elif scenario == "extreme_density":
        # Fully bipartite connected
        train_dict = {u: [(i, 3 + (u + i) % 3, 100) for i in range(num_items)] for u in range(num_users)}

    model.set_adjacency(train_dict, device="cpu")

    # Invariant checks
    assert model.adj_norm is not None, "adj_norm is None"
    assert model.adj_norm.is_coalesced(), f"Scenario {scenario}: adj_norm is NOT coalesced!"
    assert model.adj_norm.shape == (num_users + num_items, num_users + num_items)
    
    # Sparse-Dense matrix multiplication test
    total_nodes = num_users + num_items
    test_vecs = torch.randn(total_nodes, 8)
    out_mm = torch.sparse.mm(model.adj_norm, test_vecs)
    assert out_mm.shape == (total_nodes, 8)
    assert torch.isfinite(out_mm).all(), f"Sparse mm produced non-finite values in {scenario}"


# ==============================================================================
# 5. ADDITIONAL ADVERSARIAL STRESS TESTS
# ==============================================================================

def test_tce_sgdl_extreme_loss_truncation_and_rating_indexing():
    """
    Test TCE_SGDL with extreme loss_trunc (all samples truncated)
    and various rating indexing conventions (0-indexed, 1-indexed, boundary values).
    """
    model = TCE_SGDL(num_users=10, num_items=10, embedding_dim=8, loss_trunc=1e-6, min_weight=0.05)
    u_ids = torch.tensor([0, 1, 2])
    i_ids = torch.tensor([1, 2, 3])

    # 1. Extreme truncation: weights should clamp to min_weight rather than 0
    obs_ratings_1indexed = torch.tensor([1, 3, 5])
    loss, weights = model.compute_loss(u_ids, i_ids, obs_ratings_1indexed, return_weights=True)
    assert torch.isfinite(loss)
    assert loss.item() > 0.0
    assert torch.all(weights >= model.min_weight)

    # 2. 0-indexed ratings [0, 2, 4]
    obs_ratings_0indexed = torch.tensor([0, 2, 4])
    loss_0, weights_0 = model.compute_loss(u_ids, i_ids, obs_ratings_0indexed, return_weights=True)
    assert torch.isfinite(loss_0)
    assert torch.all(weights_0 >= model.min_weight)

    # 3. Boundary / out-of-range ratings clamped safely
    obs_ratings_out = torch.tensor([-1, 0, 10])
    loss_out, weights_out = model.compute_loss(u_ids, i_ids, obs_ratings_out, return_weights=True)
    assert torch.isfinite(loss_out)
    assert torch.all(weights_out >= model.min_weight)


def test_trust_grs_single_interaction_zero_variance_isolation():
    """
    Test TrustGRS with isolated single-interaction items and users.
    Empirical prior variance is 0.0, which must be protected by eps_rating / tau_var.
    """
    num_users, num_items = 10, 10
    model = TrustGRS(num_users=num_users, num_items=num_items, embedding_dim=8, eps_rating=1e-2, tau_var=0.5)

    # Each user and item has exactly 1 interaction
    train_dict = {u: [(u, 5, 100)] for u in range(num_users)}
    weights = model.compute_trust_weights(train_dict)

    assert len(weights) == num_users
    for (u, i), w in weights.items():
        assert np.isfinite(w), f"Weight for ({u}, {i}) is non-finite: {w}"
        assert w >= model.eps_min
        assert w <= 1.0

    model.set_adjacency(train_dict, device="cpu")
    assert model.adj_norm.is_coalesced()
    assert torch.isfinite(model.adj_norm.values()).all()


def test_score_candidates_batch_duplicate_candidates():
    """
    Verify candidate scoring consistency when duplicate item IDs exist in candidate tensor.
    Duplicate candidate items must receive identical expected ratings.
    """
    model = LightGCN(num_users=5, num_items=10, embedding_dim=8)
    user_ids = torch.tensor([0, 1])
    # Item 3 repeated 3 times
    candidates = torch.tensor([[3, 3, 3, 7], [1, 3, 3, 3]])
    scores = model.score_candidates_batch(user_ids, candidates)

    assert scores.shape == (2, 4)
    # User 0, item 3 scores across positions 0, 1, 2 must be identical
    assert torch.allclose(scores[0, 0], scores[0, 1])
    assert torch.allclose(scores[0, 1], scores[0, 2])
    # User 1, item 3 scores across positions 1, 2, 3 must be identical
    assert torch.allclose(scores[1, 1], scores[1, 2])
    assert torch.allclose(scores[1, 2], scores[1, 3])


def test_baseline_trainer_generate_recommendations_ranking_invariance():
    """
    Verify BaselineTrainer.generate_recommendations produces valid top-K rankings
    of exact length K, without NaN or unranked candidates.
    """
    import types
    num_users, num_items = 10, 20
    model = LightGCN(num_users=num_users, num_items=num_items, embedding_dim=8)
    train_dict = {u: [((u + 1) % num_items, 4, 100)] for u in range(num_users)}
    model.set_adjacency(train_dict, device="cpu")

    clean_split = types.SimpleNamespace(
        num_items=num_items,
        test_dict={u: (u % num_items, 5, 200) for u in range(num_users)},
        train_dict=train_dict
    )

    trainer = BaselineTrainer(device="cpu", epochs=1, baseline_type="standard")
    eval_users = list(range(num_users))
    recs = trainer.generate_recommendations(model, eval_users, clean_split, top_k=5)

    assert len(recs) == num_users
    for u, rec_list in recs.items():
        assert len(rec_list) == 5
        # All recommended items must be in [0, num_items)
        assert all(0 <= item < num_items for item in rec_list)
        # All recommended items must be unique
        assert len(set(rec_list)) == 5
