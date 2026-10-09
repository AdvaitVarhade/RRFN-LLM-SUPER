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
    Implements Rank-Calibrated SUPER Soft-Blueprint Merge:
    1. Calibrated quotas: N_pop = round(top_k * pop_tilde_u), N_tail = top_k - N_pop.
    2. Dynamic trajectory tracking: at each rank position k, chooses from Head or Tail pool
       to minimize cumulative calibration drift |(head_count / k) - pop_tilde_u| while
       incorporating user preference and purified blueprint guidance.
    3. Guarantees non-empty candidate fallback and strict top-K deduplication.
    """
    if top_k <= 0:
        return []

    # 1. Calibrated Quota Allocation
    n_pop = int(round(top_k * pop_tilde_u))
    if pop_tilde_u < 1.0 and n_pop >= top_k and len(tail_candidates) > 0:
        n_pop = top_k - 1
    if pop_tilde_u > 0.0 and n_pop == 0 and len(pop_candidates) > 0:
        n_pop = 1
    n_tail = top_k - n_pop

    c_pop = pop_candidates[:n_pop]
    c_tail = tail_candidates[:n_tail]

    rec_list: List[int] = []
    seen: Set[int] = set()
    idx_pop = 0
    idx_tail = 0
    head_count = 0

    def add_item(item: int, is_head: bool) -> bool:
        nonlocal head_count
        if item not in seen:
            rec_list.append(item)
            seen.add(item)
            if is_head:
                head_count += 1
            return True
        return False

    # 2. Smooth Rank-Calibrated Interleaving
    for step in range(1, top_k + 1):
        if len(rec_list) >= top_k:
            break
        can_pop = (idx_pop < len(c_pop))
        can_tail = (idx_tail < len(c_tail))

        if can_pop and can_tail:
            ratio_if_pop = (head_count + 1) / step
            ratio_if_tail = head_count / step
            diff_pop = abs(ratio_if_pop - pop_tilde_u)
            diff_tail = abs(ratio_if_tail - pop_tilde_u)

            b_pref = blueprint_b_u[step - 1] if step - 1 < len(blueprint_b_u) else None
            if b_pref == 1 and diff_pop <= diff_tail + 0.12:
                add_item(c_pop[idx_pop], is_head=True)
                idx_pop += 1
            elif b_pref == 0 and diff_tail <= diff_pop + 0.12:
                add_item(c_tail[idx_tail], is_head=False)
                idx_tail += 1
            elif diff_pop < diff_tail:
                add_item(c_pop[idx_pop], is_head=True)
                idx_pop += 1
            else:
                add_item(c_tail[idx_tail], is_head=False)
                idx_tail += 1
        elif can_pop:
            add_item(c_pop[idx_pop], is_head=True)
            idx_pop += 1
        elif can_tail:
            add_item(c_tail[idx_tail], is_head=False)
            idx_tail += 1

    # 3. Residual filling from truncated pools
    while len(rec_list) < top_k and idx_pop < len(c_pop):
        add_item(c_pop[idx_pop], is_head=True)
        idx_pop += 1
    while len(rec_list) < top_k and idx_tail < len(c_tail):
        add_item(c_tail[idx_tail], is_head=False)
        idx_tail += 1

    # 4. Fallback from remaining candidates
    if len(rec_list) < top_k:
        for item in pop_candidates:
            if len(rec_list) >= top_k:
                break
            add_item(item, is_head=True)
        for item in tail_candidates:
            if len(rec_list) >= top_k:
                break
            add_item(item, is_head=False)

    return rec_list[:top_k]
