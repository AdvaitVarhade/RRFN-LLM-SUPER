import pytest
from src.super_engine.inclination import compute_robust_inclination

def test_robust_inclination():
    train_dict = {
        0: [(0, 5, 100), (1, 4, 101), (2, 3, 102)], # 2 head, 1 tail
        1: [(2, 3, 103)], # 1 tail
        2: [] # empty
    }
    head_items = {0, 1}
    weights = {(0, 0): 1.0, (0, 1): 1.0, (0, 2): 1.0, (1, 2): 1.0}

    inclinations = compute_robust_inclination(train_dict, head_items, weights, shrinkage_tau=5.0, global_head_prior=0.20)

    # User 0 has mostly head -> higher inclination
    # User 1 has only tail -> lower inclination pulled toward prior
    assert inclinations[0] > inclinations[1]
    assert 0.0 <= inclinations[0] <= 1.0
    assert 0.0 <= inclinations[1] <= 1.0
    assert inclinations[2] == 0.20
