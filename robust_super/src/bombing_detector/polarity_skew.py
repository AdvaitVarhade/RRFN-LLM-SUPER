import numpy as np
from typing import Dict, List, Tuple

def compute_polarity_skew(
    train_dict: Dict[int, List[Tuple[int, int, int]]],
    time_window_hours: int = 24
) -> Dict[Tuple[int, int], float]:
    """
    Computes polarity skew S(i, t) - the fraction of extreme ratings (1 or 5)
    in the temporal neighborhood of each interaction.
    Optimized with binary search and prefix sum counting.
    """
    window_seconds = time_window_hours * 3600
    item_events: Dict[int, List[Tuple[int, int]]] = {} # item -> [(ts, rating)]

    for u, interactions in train_dict.items():
        for item, rating, ts in interactions:
            item_events.setdefault(item, []).append((ts, rating))

    # Precompute sorted timestamp arrays and extreme rating indicator arrays
    item_ts_arr: Dict[int, np.ndarray] = {}
    item_extreme_prefix: Dict[int, np.ndarray] = {}

    for item, events in item_events.items():
        events.sort(key=lambda x: x[0])
        ts_arr = np.array([e[0] for e in events], dtype=np.int64)
        extreme_arr = np.array([1 if e[1] in (1, 5) else 0 for e in events], dtype=np.int32)
        # Prefix sum for O(1) range count of extreme ratings
        prefix = np.zeros(len(events) + 1, dtype=np.int32)
        prefix[1:] = np.cumsum(extreme_arr)

        item_ts_arr[item] = ts_arr
        item_extreme_prefix[item] = prefix

    polarity_scores: Dict[Tuple[int, int], float] = {}

    half_window = window_seconds // 2
    for u, interactions in train_dict.items():
        for item, rating, ts in interactions:
            ts_arr = item_ts_arr[item]
            if len(ts_arr) <= 2:
                polarity_scores[(u, item)] = 0.5
                continue

            l_idx = np.searchsorted(ts_arr, ts - half_window, side='left')
            r_idx = np.searchsorted(ts_arr, ts + half_window, side='right')
            n_window = r_idx - l_idx

            if n_window <= 0:
                polarity_scores[(u, item)] = 0.5
                continue

            prefix = item_extreme_prefix[item]
            n_extreme = prefix[r_idx] - prefix[l_idx]
            skew = n_extreme / float(n_window)
            polarity_scores[(u, item)] = float(skew)

    return polarity_scores
