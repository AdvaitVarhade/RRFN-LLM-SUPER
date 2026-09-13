from typing import Dict, List, Tuple, Set

def compute_effective_volume(
    train_dict: Dict[int, List[Tuple[int, int, int]]],
    weights: Dict[Tuple[int, int], float],
    num_items: int
) -> Dict[int, float]:
    """
    Computes reliability-weighted effective volume for each item:
    V_eff(i) = sum_{u: (u,i) in train} w_{ui}
    """
    V_eff: Dict[int, float] = {i: 0.0 for i in range(num_items)}

    for u, interactions in train_dict.items():
        for item, rating, ts in interactions:
            w = weights.get((u, item), 1.0)
            V_eff[item] = V_eff.get(item, 0.0) + w

    return V_eff

def pareto_partition(
    V_eff: Dict[int, float],
    pareto_alpha: float = 0.20
) -> Tuple[Set[int], Set[int]]:
    """
    Applies the Pareto principle over effective volume:
    Sort items descending by V_eff.
    Head (H) = minimal prefix whose cumulative volume reaches alpha (e.g. 20%) of total volume.
    Tail (T) = remaining items.
    """
    sorted_items = sorted(V_eff.items(), key=lambda x: -x[1])
    total_volume = sum(V_eff.values())
    threshold = pareto_alpha * total_volume

    head_set: Set[int] = set()
    cum_vol = 0.0

    for item, vol in sorted_items:
        cum_vol += vol
        head_set.add(item)
        if cum_vol >= threshold:
            break

    # If all items or no items selected, ensure valid non-empty split
    if len(head_set) == 0 and len(sorted_items) > 0:
        head_set.add(sorted_items[0][0])

    all_items = set(V_eff.keys())
    tail_set = all_items - head_set

    # If tail is empty, ensure at least one tail item
    if len(tail_set) == 0 and len(head_set) > 1:
        last_item = sorted_items[-1][0]
        head_set.remove(last_item)
        tail_set.add(last_item)

    return head_set, tail_set
