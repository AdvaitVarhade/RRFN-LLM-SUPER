from typing import Dict, List, Tuple, Set

def build_denoised_blueprints(
    train_dict: Dict[int, List[Tuple[int, int, int]]],
    weights: Dict[Tuple[int, int], float],
    user_mean_ratings: Dict[int, float],
    head_items: Set[int],
    filter_threshold: float = 0.30
) -> Dict[int, Tuple[List[int], List[int]]]:
    """
    Constructs purified user consumption blueprints (L_tilde_u, b_u):
    1. Prunes interactions with reliability weight < filter_threshold.
    2. Computes reliability-adjusted preference score: s_tilde = w * r + (1-w) * mean_r.
    3. Sorts interactions by s_tilde descending -> L_tilde_u.
    4. Extracts binary blueprint indicator: b_k = 1 if item in Head else 0.
    """
    blueprints: Dict[int, Tuple[List[int], List[int]]] = {}

    for u, interactions in train_dict.items():
        if not interactions:
            blueprints[u] = ([], [])
            continue

        mean_r = user_mean_ratings.get(u, 3.5)
        filtered_items: List[Tuple[int, float]] = []

        for item, rating, ts in interactions:
            w = weights.get((u, item), 1.0)
            if w < filter_threshold:
                continue # Skip suspicious interaction in blueprint construction

            # Adjusted score
            s_tilde = (w * float(rating)) + ((1.0 - w) * mean_r)
            filtered_items.append((item, s_tilde))

        # If all interactions were filtered out, fall back to unfiltered with default weights
        if not filtered_items:
            filtered_items = [(item, float(rating)) for item, rating, ts in interactions]

        # Sort descending by adjusted score
        filtered_items.sort(key=lambda x: -x[1])
        l_tilde = [item for item, _ in filtered_items]
        b_u = [1 if item in head_items else 0 for item in l_tilde]

        blueprints[u] = (l_tilde, b_u)

    return blueprints
