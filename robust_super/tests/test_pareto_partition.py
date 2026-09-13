import pytest
from src.catalog.pareto_partition import compute_effective_volume, pareto_partition

def test_pareto_partition():
    train_dict = {
        0: [(0, 5, 100), (1, 4, 101), (2, 3, 102)],
        1: [(0, 5, 103), (0, 4, 104)],
        2: [(3, 4, 105)]
    }
    weights = {(0, 0): 1.0, (0, 1): 1.0, (0, 2): 1.0, (1, 0): 1.0, (2, 3): 1.0}

    V_eff = compute_effective_volume(train_dict, weights, num_items=4)
    assert V_eff[0] > V_eff[1]

    head, tail = pareto_partition(V_eff, pareto_alpha=0.40)
    assert len(head) > 0
    assert len(tail) > 0
    assert head.isdisjoint(tail)
    assert head.union(tail) == set(range(4))
