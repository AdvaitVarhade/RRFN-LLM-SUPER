import numpy as np
from typing import Dict, List, Tuple, Set

def compute_robust_inclination(
    train_dict: Dict[int, List[Tuple[int, int, int]]],
    head_items: Set[int],
    weights: Dict[Tuple[int, int], float],
    shrinkage_tau: float = 5.0,
    global_head_prior: float = 0.20
) -> Dict[int, float]:
    """
    Computes reliability-weighted user popularity inclination Pop_tilde_u:
    Pop_u^robust = sum_{i in C_u ∩ H} w_{ui} / (sum_{i in C_u} w_{ui} + eps)
    lambda_u = sum_w / (sum_w + tau)
    Pop_tilde_u = lambda_u * Pop_u^robust + (1 - lambda_u) * global_head_prior
    """
    inclinations: Dict[int, float] = {}

    for u, interactions in train_dict.items():
        if not interactions:
            inclinations[u] = global_head_prior
            continue

        sum_w_head = 0.0
        sum_w_total = 0.0

        for item, rating, ts in interactions:
            w = weights.get((u, item), 1.0)
            sum_w_total += w
            if item in head_items:
                sum_w_head += w

        pop_u_robust = sum_w_head / max(1e-6, sum_w_total)
        lambda_u = sum_w_total / (sum_w_total + shrinkage_tau)
        pop_tilde_u = (lambda_u * pop_u_robust) + ((1.0 - lambda_u) * global_head_prior)

        inclinations[u] = float(np.clip(pop_tilde_u, 0.01, 0.99))

    return inclinations
