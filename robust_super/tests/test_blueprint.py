import pytest
from src.super_engine.blueprint import build_denoised_blueprints
from src.super_engine.merger import merge_top_n

def test_blueprint_and_merger():
    train_dict = {
        0: [(0, 5, 100), (1, 1, 101), (2, 4, 102)]
    }
    weights = {(0, 0): 0.95, (0, 1): 0.10, (0, 2): 0.90} # item 1 is suspicious
    user_means = {0: 3.5}
    head_items = {0}

    blueprints = build_denoised_blueprints(train_dict, weights, user_means, head_items, filter_threshold=0.30)
    l_tilde, b_u = blueprints[0]

    # Item 1 should have been pruned due to w < 0.30
    assert 1 not in l_tilde
    assert len(l_tilde) == 2

    # Test Top-N merger
    pop_candidates = [0, 10, 20]
    tail_candidates = [2, 30, 40, 50, 60, 70, 80]
    rec_list = merge_top_n(0, pop_candidates, tail_candidates, pop_tilde_u=0.30, blueprint_b_u=b_u, top_k=5)

    assert len(rec_list) == 5
    assert len(set(rec_list)) == 5
