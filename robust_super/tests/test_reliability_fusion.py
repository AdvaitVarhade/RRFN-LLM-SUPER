import pytest
from src.fusion.reliability_fusion import fuse_reliability_scores

def test_reliability_fusion():
    R_RRFN = {(1, 10): 0.90, (1, 20): 0.20}
    R_LLM  = {(1, 10): 0.95, (1, 20): 0.15}
    R_bomb = {(1, 10): 0.98, (1, 20): 0.10}

    weights = fuse_reliability_scores(
        R_RRFN, R_LLM, R_bomb,
        alpha=0.5, beta=0.3, gamma=0.2, min_weight=0.02
    )

    w_genuine = weights[(1, 10)]
    w_noisy = weights[(1, 20)]

    assert w_genuine > 0.90
    assert w_noisy < 0.30
    assert w_noisy >= 0.02
