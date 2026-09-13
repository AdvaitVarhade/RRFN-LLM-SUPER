import numpy as np
from typing import Dict, List, Tuple

def compute_temporal_acceleration(
    train_dict: Dict[int, List[Tuple[int, int, int]]],
    time_window_hours: int = 24
) -> Dict[Tuple[int, int], float]:
    """
    Computes temporal rating burst acceleration A(i, t) for each interaction.
    Optimized with O(log N) binary search (np.searchsorted) for high performance.
    """
    window_seconds = time_window_hours * 3600
    item_timestamps: Dict[int, List[int]] = {}

    # 1. Collect all interaction timestamps per item
    for u, interactions in train_dict.items():
        for item, rating, ts in interactions:
            item_timestamps.setdefault(item, []).append(ts)

    # Convert to sorted numpy arrays once
    item_ts_arrays: Dict[int, np.ndarray] = {
        item: np.sort(np.array(ts_list, dtype=np.int64))
        for item, ts_list in item_timestamps.items()
    }

    acceleration_scores: Dict[Tuple[int, int], float] = {}

    # 2. Vectorized / binary-search density calculation
    for u, interactions in train_dict.items():
        for item, rating, ts in interactions:
            arr = item_ts_arrays[item]
            if len(arr) <= 3:
                acceleration_scores[(u, item)] = 1.0
                continue

            # Binary search range counts
            idx_curr_r = np.searchsorted(arr, ts, side='right')
            idx_curr_l = np.searchsorted(arr, ts - window_seconds, side='left')
            window_count = idx_curr_r - idx_curr_l

            idx_b1_l = np.searchsorted(arr, ts - 2 * window_seconds, side='left')
            b1 = idx_curr_l - idx_b1_l

            idx_b2_l = np.searchsorted(arr, ts - 3 * window_seconds, side='left')
            b2 = idx_b1_l - idx_b2_l

            idx_b3_l = np.searchsorted(arr, ts - 4 * window_seconds, side='left')
            b3 = idx_b2_l - idx_b3_l

            b_avg = (b1 + b2 + b3) / 3.0
            accel = window_count / (b_avg + 1.0)
            acceleration_scores[(u, item)] = float(min(10.0, accel))

    return acceleration_scores
