import numpy as np
from typing import Dict, List, Tuple, Set

def merge_top_n(
    user_id: int,
    pop_candidates: List[int],    # sorted by M_pop score descending
    tail_candidates: List[int],   # sorted by M_tail score descending
    pop_tilde_u: float,
    blueprint_b_u: List[int],
    top_k: int = 10
) -> List[int]:
    """
    Implements SUPER Hard-Quota / Soft-Blueprint merge:
    1. Hard quotas: N_pop = floor(top_k * pop_tilde_u), N_tail = top_k - N_pop.
    2. Truncates candidate pools: C_pop of size N_pop, C_tail of size N_tail.
    3. Interleaves candidates following blueprint b_u as soft rank guide.
    4. Deterministically switches if requested pool quota is exhausted.
    5. Fills remaining slots if blueprint covers fewer than top_k items.
    """
    # 1. Hard Quotas
    n_pop = int(np.floor(top_k * pop_tilde_u))
    # Ensure at least 1 tail item if possible unless pop_tilde_u == 1.0
    if pop_tilde_u < 1.0 and n_pop == top_k:
        n_pop = top_k - 1
    if pop_tilde_u > 0.0 and n_pop == 0 and len(pop_candidates) > 0:
        n_pop = 1
    n_tail = top_k - n_pop

    # 2. Truncate pools
    c_pop = pop_candidates[:n_pop]
    c_tail = tail_candidates[:n_tail]

    rec_list: List[int] = []
    idx_pop = 0
    idx_tail = 0

    # 3. Interleave using blueprint
    b_limit = min(top_k, len(blueprint_b_u))
    for k in range(b_limit):
        b_k = blueprint_b_u[k]
        if b_k == 1 and idx_pop < len(c_pop):
            rec_list.append(c_pop[idx_pop])
            idx_pop += 1
        elif b_k == 0 and idx_tail < len(c_tail):
            rec_list.append(c_tail[idx_tail])
            idx_tail += 1
        elif idx_pop < len(c_pop):
            rec_list.append(c_pop[idx_pop])
            idx_pop += 1
        elif idx_tail < len(c_tail):
            rec_list.append(c_tail[idx_tail])
            idx_tail += 1

    # 4. Residual filling
    while len(rec_list) < top_k and idx_pop < len(c_pop):
        rec_list.append(c_pop[idx_pop])
        idx_pop += 1
    while len(rec_list) < top_k and idx_tail < len(c_tail):
        rec_list.append(c_tail[idx_tail])
        idx_tail += 1

    # If still fewer than top_k (e.g. pool ran out of items), fill with remaining candidates
    if len(rec_list) < top_k:
        seen = set(rec_list)
        for item in pop_candidates + tail_candidates:
            if item not in seen:
                rec_list.append(item)
                seen.add(item)
                if len(rec_list) == top_k:
                    break

    return rec_list[:top_k]
