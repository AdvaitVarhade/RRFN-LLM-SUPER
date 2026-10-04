import torch
import pytest
from src.models.simgcl import SimGCL

def test_simgcl_initialization_and_forward():
    num_users = 20
    num_items = 40
    model = SimGCL(num_users=num_users, num_items=num_items, embedding_dim=16, num_layers=2, noise_eps=0.1)

    u_ids = torch.tensor([0, 1, 2])
    i_ids = torch.tensor([5, 10, 15])

    probs = model(u_ids, i_ids)
    assert probs.shape == (3, 5)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(3), atol=1e-5)

def test_simgcl_contrastive_views():
    num_users = 20
    num_items = 40
    model = SimGCL(num_users=num_users, num_items=num_items, embedding_dim=16, num_layers=2, noise_eps=0.2)

    # Set mock adjacency
    train_dict = {0: [(1, 5, 100)], 1: [(2, 4, 100)]}
    model.set_adjacency(train_dict, device="cpu")

    v1, v2 = model.get_cl_embeddings()
    assert v1.shape == (num_users + num_items, 16)
    assert v2.shape == (num_users + num_items, 16)
    # With noise_eps > 0, the two perturbed views must not be identical
    assert not torch.allclose(v1, v2)
