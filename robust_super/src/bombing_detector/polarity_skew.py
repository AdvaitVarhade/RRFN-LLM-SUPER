import numpy as np
from typing import Dict, List, Tuple

def compute_polarity_skew(
    train_dict: Dict[int, List[Tuple[int, int, int]]],
    time_window_hours: int = 24
) -> Dict[Tuple[int, int], float]:
    """
    Computes polarity skew S(i, t) - the fraction of extreme ratings (1 or 5)
    in the temporal neighborhood of each interaction.
    Vectorized per-item prefix sums and window search for high performance.
    """
    window_seconds = time_window_hours * 3600
    half_window = window_seconds // 2

    item_interactions: Dict[int, List[Tuple[int, int, int]]] = {} # item -> [(u, rating, ts)]

    for u, interactions in train_dict.items():
        for item, rating, ts in interactions:
            item_interactions.setdefault(item, []).append((u, rating, ts))

    polarity_scores: Dict[Tuple[int, int], float] = {}

    for item, events in item_interactions.items():
        n = len(events)
        if n <= 2:
            for u, _, _ in events:
                polarity_scores[(u, item)] = 0.5
            continue

        # Sort events by timestamp
        events.sort(key=lambda x: x[2])
        users = [e[0] for e in events]
        ts_arr = np.array([e[2] for e in events], dtype=np.int64)
        extreme_arr = np.array([1 if e[1] in (1, 5) else 0 for e in events], dtype=np.int32)

        prefix = np.zeros(n + 1, dtype=np.int32)
        prefix[1:] = np.cumsum(extreme_arr)

        l_idx = np.searchsorted(ts_arr, ts_arr - half_window, side='left')
        r_idx = np.searchsorted(ts_arr, ts_arr + half_window, side='right')
        n_window = r_idx - l_idx

        valid_mask = n_window > 0
        skew = np.full(n, 0.5, dtype=np.float32)
        if np.any(valid_mask):
            n_extreme = prefix[r_idx[valid_mask]] - prefix[l_idx[valid_mask]]
            skew[valid_mask] = n_extreme / n_window[valid_mask].astype(np.float32)

        for u, s_val in zip(users, skew):
            polarity_scores[(u, item)] = float(s_val)

    return polarity_scores
