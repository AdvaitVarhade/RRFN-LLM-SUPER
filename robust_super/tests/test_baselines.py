"""
robust_super/tests/test_baselines.py
------------------------------------
Comprehensive regression and unit test suite for Milestone M2:
1. Mathematical invariants: LightGCN, SGL, T-CE/SGDL, Trust-GRS
2. BaselineTrainer execution across all 4 baseline modes (standard, sgl, tce_sgdl, trust_grs)
3. Zero-leakage bandwagon attack isolation contract
4. Candidate ranking and interface parity
5. Numerical stability and stress corner cases
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
# 1. MATHEMATICAL INVARIANTS: LIGHTGCN & SGL
# ==============================================================================

def test_plain_lightgcn_graph_laplacian_normalization():
    num_users, num_items = 10, 15
    model = LightGCN(num_users=num_users, num_items=num_items, embedding_dim=16, num_layers=3)
    train_dict = {u: [(i, 5, 100) for i in range(5)] for u in range(num_users)}
    model.set_adjacency(train_dict, device="cpu")

    assert model.adj_norm is not None
    assert model.adj_norm.is_coalesced()
    assert model.adj_norm.shape == (num_users + num_items, num_users + num_items)

    # In eval mode, evaluate caching
    model.eval()
    u_all, i_all = model.compute_all_embeddings()
    assert u_all.shape == (num_users, 16)
    assert i_all.shape == (num_items, 16)
    model.score_items(0, [1])
    assert model._cached_user_embs is not None
    assert model._cached_item_embs is not None

    # Cache must clear on train(True)
    model.train(True)
    assert model._cached_user_embs is None
    assert model._cached_item_embs is None


def test_plain_lightgcn_scoring_candidates_batch():
    model = LightGCN(num_users=15, num_items=30, embedding_dim=8, num_layers=2)
    train_dict = {u: [((u + 1) % 30, 5, 100)] for u in range(15)}
    model.set_adjacency(train_dict, device="cpu")

    user_ids = torch.tensor([0, 1, 2])
    candidates = torch.tensor([
        [5, 10, 15, 20],
        [6, 11, 16, 21],
        [7, 12, 17, 22]
    ])

    scores = model.score_candidates_batch(user_ids, candidates)
    assert scores.shape == (3, 4)
    assert torch.isfinite(scores).all()
    # Expected rating must lie in [1.0, 5.0]
    assert (scores >= 1.0 - 1e-4).all()
    assert (scores <= 5.0 + 1e-4).all()


def test_plain_sgl_topological_dropout_augmentation():
    num_users, num_items = 15, 20
    model = SGL(num_users=num_users, num_items=num_items, embedding_dim=16, num_layers=2, drop_rate=0.2, augment_type="ED")
    train_dict = {u: [(u % num_items, 5, 100), ((u + 1) % num_items, 4, 100)] for u in range(num_users)}
    model.set_adjacency(train_dict, device="cpu")

    v1, v2 = model.get_cl_embeddings()
    assert v1.shape == (num_users + num_items, 16)
    assert v2.shape == (num_users + num_items, 16)
    assert torch.isfinite(v1).all()
    assert torch.isfinite(v2).all()

    # Test Node Dropout mode as well
    model_nd = SGL(num_users=num_users, num_items=num_items, embedding_dim=16, num_layers=2, drop_rate=0.2, augment_type="ND")
    model_nd.set_adjacency(train_dict, device="cpu")
    v1_nd, v2_nd = model_nd.get_cl_embeddings()
    assert v1_nd.shape == (num_users + num_items, 16)
    assert v2_nd.shape == (num_users + num_items, 16)
    assert torch.isfinite(v1_nd).all()


# ==============================================================================
# 2. MATHEMATICAL INVARIANTS: T-CE / SGDL
# ==============================================================================

def test_tce_sgdl_initialization_and_forward():
    num_users, num_items = 25, 50
    model = TCE_SGDL(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=16,
        num_layers=2,
        num_classes=5,
        tau_loss=1.0,
        delta_T_reg=0.01
    )

    u_ids = torch.tensor([0, 1, 2, 3])
    i_ids = torch.tensor([10, 20, 30, 40])
    probs = model(u_ids, i_ids)

    assert probs.shape == (4, 5)
    assert torch.isfinite(probs).all()
    np.testing.assert_allclose(probs.sum(dim=-1).detach().numpy(), np.ones(4), atol=1e-5)


def test_tce_sgdl_transition_matrix_and_anchor_estimation():
    model = TCE_SGDL(num_users=10, num_items=20, embedding_dim=8, num_layers=1)

    # Initial transition matrix should be row-stochastic
    T_init = model.get_transition_matrix()
    assert T_init.shape == (5, 5)
    row_sums = T_init.sum(dim=-1).detach().numpy()
    np.testing.assert_allclose(row_sums, np.ones(5), atol=1e-5)

    anchor_sets = {
        0: [(0, 1, 1, np.array([0.9, 0.05, 0.02, 0.02, 0.01])) for _ in range(5)],
        1: [(1, 2, 2, np.array([0.05, 0.85, 0.05, 0.03, 0.02])) for _ in range(5)],
        2: [(2, 3, 3, np.array([0.02, 0.03, 0.9, 0.03, 0.02])) for _ in range(5)],
        3: [(3, 4, 4, np.array([0.01, 0.02, 0.02, 0.9, 0.05])) for _ in range(5)],
        4: [(4, 5, 5, np.array([0.01, 0.01, 0.02, 0.06, 0.9])) for _ in range(5)],
    }
    model.estimate_transition_from_anchors(anchor_sets)

    T_est = model.get_transition_matrix()
    assert T_est.shape == (5, 5)
    np.testing.assert_allclose(T_est.sum(dim=-1).detach().numpy(), np.ones(5), atol=1e-5)
    for k in range(5):
        assert T_est[k, k] > T_est[k, (k + 1) % 5]


def test_tce_sgdl_forward_risk_loss_and_weights():
    model = TCE_SGDL(num_users=10, num_items=20, embedding_dim=8, num_layers=1, tau_loss=1.0)

    u_ids = torch.tensor([0, 1, 2])
    i_ids = torch.tensor([2, 3, 4])
    observed_ratings = torch.tensor([5, 1, 3])  # 1-indexed

    loss, weights = model.compute_loss(u_ids, i_ids, observed_ratings, return_weights=True)
    assert torch.isfinite(loss)
    assert loss.item() > 0.0
    assert weights is not None
    assert weights.shape == (3,)
    assert (weights >= model.min_weight).all()
    assert (weights <= 1.0).all()
    assert weights.requires_grad is False  # Explicit detach assertion

    loss.backward()
    assert model.user_embedding.weight.grad is not None
    assert model.transition.delta_T.grad is not None


def test_tce_sgdl_tikhonov_inversion_stability():
    model = TCE_SGDL(num_users=10, num_items=20, embedding_dim=8)

    lam = 1e-3
    T_inv = model.get_transition_inverse(reg_lambda=lam)
    assert T_inv.shape == (5, 5)
    assert torch.isfinite(T_inv).all()

    spectral_norm = float(torch.linalg.norm(T_inv, 2).item())
    theoretical_bound = 1.0 / (2.0 * np.sqrt(lam))
    assert spectral_norm <= theoretical_bound + 1e-4

    # Degenerate case: uniform singular matrix
    with torch.no_grad():
        model.transition.T_hat.fill_(0.2)
        model.transition.delta_T.zero_()

    T_inv_singular = model.get_transition_inverse(reg_lambda=1e-2)
    assert T_inv_singular.shape == (5, 5)
    assert torch.isfinite(T_inv_singular).all()
    assert not torch.isnan(T_inv_singular).any()


def test_tce_sgdl_empty_batch_guard():
    model = TCE_SGDL(num_users=5, num_items=5, embedding_dim=8)
    empty_u = torch.empty(0, dtype=torch.long)
    empty_i = torch.empty(0, dtype=torch.long)
    empty_y = torch.empty(0, dtype=torch.long)

    loss, weights = model.compute_loss(empty_u, empty_i, empty_y, return_weights=True)
    assert loss.item() == 0.0
    assert loss.requires_grad is True
    assert weights.shape == (0,)


# ==============================================================================
# 3. MATHEMATICAL INVARIANTS: TRUST-GRS
# ==============================================================================

def test_trust_grs_anomaly_attenuation_and_coalesce():
    num_users, num_items = 20, 30
    model = TrustGRS(num_users=num_users, num_items=num_items, embedding_dim=16, num_layers=2)
    # Authentic users rate normally with some variance
    train_dict = {u: [(i, 2 + (u + i) % 4, 100) for i in range(4)] for u in range(15)}
    # Bot users: exclusively 5-star ratings on head items and target items
    for bot_u in range(15, 20):
        train_dict[bot_u] = [(0, 5, 100), (1, 5, 100), (28, 5, 100), (29, 5, 100)]

    trust_weights = model.compute_trust_weights(train_dict)
    assert len(trust_weights) > 0

    # Authentic user edge trust should exceed bot edge trust
    auth_w = np.mean([trust_weights[(u, i)] for u, inters in train_dict.items() if u < 15 for i, _, _ in inters])
    bot_w = np.mean([trust_weights[(u, i)] for u, inters in train_dict.items() if u >= 15 for i, _, _ in inters])
    assert auth_w > bot_w

    model.set_adjacency(train_dict, device="cpu")
    assert model.adj_norm.is_coalesced()
    assert model.adj_norm.shape == (num_users + num_items, num_users + num_items)


def test_trust_grs_loss_and_gradient():
    model = TrustGRS(num_users=10, num_items=15, embedding_dim=8, num_layers=1)
    train_dict = {u: [(i, 3, 100) for i in range(3)] for u in range(10)}
    model.set_adjacency(train_dict, device="cpu")

    u_ids = torch.tensor([0, 1, 2])
    i_ids = torch.tensor([1, 2, 3])
    ratings = torch.tensor([4, 5, 2])
    weights = torch.tensor([0.9, 0.8, 0.1])

    probs = model(u_ids, i_ids)
    loss = model.compute_trust_weighted_loss(probs, ratings, weights=weights)

    assert torch.isfinite(loss)
    assert loss.item() > 0.0

    loss.backward()
    assert model.user_embedding.weight.grad is not None
    assert model.item_embedding.weight.grad is not None


# ==============================================================================
# 4. BASELINE TRAINER END-TO-END EXECUTION
# ==============================================================================

@pytest.mark.parametrize("baseline_type", ["standard", "sgl", "tce_sgdl", "trust_grs"])
def test_baseline_trainer_all_modes_e2e(baseline_type):
    num_users, num_items = 12, 20
    train_dict = {u: [((u + i) % num_items, 4, 1000 + i) for i in range(4)] for u in range(num_users)}
    weights = {(u, i): 1.0 for u in train_dict for i, _, _ in train_dict[u]}
    val_dict = {u: ((u + 5) % num_items, 5, 2000) for u in range(num_users)}
    loader = build_data_loader(train_dict, weights, batch_size=16)

    if baseline_type == "standard":
        model = LightGCN(num_users, num_items, embedding_dim=16, num_layers=2)
        model.set_adjacency(train_dict, device="cpu")
    elif baseline_type == "sgl":
        model = SGL(num_users, num_items, embedding_dim=16, num_layers=2, drop_rate=0.1)
        model.set_adjacency(train_dict, device="cpu")
    elif baseline_type == "tce_sgdl":
        model = TCE_SGDL(num_users, num_items, embedding_dim=16, num_layers=2)
        model.set_adjacency(train_dict, device="cpu")
    else:
        model = TrustGRS(num_users, num_items, embedding_dim=16, num_layers=2)

    trainer = BaselineTrainer(device="cpu", lr=0.005, epochs=2, early_stopping_patience=2, baseline_type=baseline_type)
    trained = trainer.train(model, loader, val_dict, train_dict=train_dict)
    assert trained is not None
    assert len(trained.loss_history) == 2


# ==============================================================================
# 5. ZERO-LEAKAGE ATTACK ISOLATION CONTRACT
# ==============================================================================

def test_zero_leakage_attack_isolation():
    records = []
    for u in range(30):
        for step, i in enumerate(range(8)):
            records.append({"user_id": u, "item_id": i, "rating": 4, "timestamp": 1000 + step})
    df = pd.DataFrame(records)
    items_df = pd.DataFrame({"item_id": list(range(8))})
    users_df = pd.DataFrame({"user_id": list(range(30))})

    clean_split = preprocess_dataset(df, items_df, users_df)
    clean_val_copy = copy.deepcopy(clean_split.val_dict)
    clean_test_copy = copy.deepcopy(clean_split.test_dict)
    num_clean_users = clean_split.num_users

    sim = AttackSimulator(seed=42)
    attacked_split = sim.inject_attack(clean_split, attack_type="bandwagon", noise_rate=0.15)

    # 1. Validation dict strictly unmodified
    assert clean_val_copy == attacked_split.val_dict
    # 2. Test dict strictly unmodified
    assert clean_test_copy == attacked_split.test_dict
    # 3. Fake users added exclusively to train_dict
    fake_users = [u for u in attacked_split.train_dict if u >= num_clean_users]
    assert len(fake_users) > 0
    # 4. Zero fake users present in validation or test
    assert not any(u in attacked_split.val_dict for u in fake_users)
    assert not any(u in attacked_split.test_dict for u in fake_users)


# ==============================================================================
# 6. ADVERSARIAL CORNER CASES & NUMERICAL STABILITY
# ==============================================================================

def test_adversarial_disconnected_graph_handling():
    num_users, num_items = 10, 15
    empty_train_dict = {}
    model = LightGCN(num_users=num_users, num_items=num_items, embedding_dim=16)
    model.set_adjacency(empty_train_dict, device="cpu")

    model.eval()
    scores = model.score_candidates_batch(torch.tensor([0, 1]), torch.tensor([[1, 2], [3, 4]]))
    assert scores.shape == (2, 2)
    assert torch.isfinite(scores).all()


def test_adversarial_candidate_scoring_expected_ratings_bound():
    model = LightGCN(num_users=5, num_items=10, embedding_dim=8)
    user_ids = torch.tensor([0, 1])
    candidates = torch.tensor([[0, 1, 2], [3, 4, 5]])
    scores = model.score_candidates_batch(user_ids, candidates)

    assert (scores >= 1.0 - 1e-4).all()
    assert (scores <= 5.0 + 1e-4).all()
