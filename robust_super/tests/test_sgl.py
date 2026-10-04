import torch
import pytest
from src.models.sgl import SGL

def test_sgl_initialization_and_forward():
    num_users = 20
    num_items = 40
    model = SGL(num_users=num_users, num_items=num_items, embedding_dim=16, num_layers=2, drop_rate=0.2, augment_type="ED")

    u_ids = torch.tensor([0, 1, 2])
    i_ids = torch.tensor([5, 10, 15])

    probs = model(u_ids, i_ids)
    assert probs.shape == (3, 5)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(3), atol=1e-5)

def test_sgl_edge_dropout_contrastive_views():
    num_users = 20
    num_items = 40
    model = SGL(num_users=num_users, num_items=num_items, embedding_dim=16, num_layers=2, drop_rate=0.3, augment_type="ED")

    # Set mock adjacency with multiple edges
    train_dict = {
        0: [(1, 5, 100), (2, 4, 100), (3, 3, 100)],
        1: [(2, 4, 100), (4, 5, 100), (5, 2, 100)],
        2: [(1, 5, 100), (3, 4, 100)]
    }
    model.set_adjacency(train_dict, device="cpu")

    v1, v2 = model.get_cl_embeddings()
    assert v1.shape == (num_users + num_items, 16)
    assert v2.shape == (num_users + num_items, 16)
