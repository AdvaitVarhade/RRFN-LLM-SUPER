import numpy as np
from typing import Dict, List, Tuple

def compute_temporal_acceleration(
    train_dict: Dict[int, List[Tuple[int, int, int]]],
    time_window_hours: int = 24
) -> Dict[Tuple[int, int], float]:
    """
    Computes temporal rating burst acceleration A(i, t) for each interaction.
    Fully vectorized per-item binary search (np.searchsorted) for high performance.
    """
    window_seconds = time_window_hours * 3600
    item_interactions: Dict[int, List[Tuple[int, int]]] = {}

    # 1. Group interactions by item: item -> [(user, timestamp)]
    for u, interactions in train_dict.items():
        for item, rating, ts in interactions:
            item_interactions.setdefault(item, []).append((u, ts))

    acceleration_scores: Dict[Tuple[int, int], float] = {}

    # 2. Vectorized per-item window search
    for item, pairs in item_interactions.items():
        n = len(pairs)
        if n <= 3:
            for u, _ in pairs:
                acceleration_scores[(u, item)] = 1.0
            continue

        # Sort pairs by timestamp
        pairs.sort(key=lambda x: x[1])
        users = [p[0] for p in pairs]
        ts_arr = np.array([p[1] for p in pairs], dtype=np.int64)

        # Vectorized binary search over entire timestamp array simultaneously
        curr_r = np.searchsorted(ts_arr, ts_arr, side='right')
        curr_l = np.searchsorted(ts_arr, ts_arr - window_seconds, side='left')
        window_count = curr_r - curr_l

        b1_l = np.searchsorted(ts_arr, ts_arr - 2 * window_seconds, side='left')
        b1 = curr_l - b1_l

        b2_l = np.searchsorted(ts_arr, ts_arr - 3 * window_seconds, side='left')
        b2 = b1_l - b2_l

        b3_l = np.searchsorted(ts_arr, ts_arr - 4 * window_seconds, side='left')
        b3 = b2_l - b3_l

        b_avg = (b1 + b2 + b3) / 3.0
        accel = np.minimum(10.0, window_count / (b_avg + 1.0))

        for u, a_val in zip(users, accel):
            acceleration_scores[(u, item)] = float(a_val)

    return acceleration_scores
