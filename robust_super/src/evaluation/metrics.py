import numpy as np
from typing import Dict, List, Tuple, Set, Optional

def evaluate_recommendations(
    recommendations: Dict[int, List[int]], # u -> top-K item list
    test_dict: Dict[int, Tuple[int, int, int]], # u -> (test_item, rating, ts)
    train_dict: Dict[int, List[Tuple[int, int, int]]],
    head_items: Set[int],
    tail_items: Set[int],
    user_inclinations: Dict[int, float],
    top_k: int = 10
) -> Dict[str, float]:
    """
    Computes standard recommendation and debiasing metrics as formulated in SUPER (2026):
    - Accuracy: nDCG@K, Recall@K
    - Calibration: RMSE-PC, MRMC
    - Beyond-Accuracy: APLT, LTC, Entropy, Novelty
    - Holistic Summary: GKPI
    """
    eval_users = [u for u in test_dict if u in recommendations]
    if not eval_users:
        return {}

    hits = 0
    ndcgs = []
    rmse_pc_errors = []
    mrmc_errors = []
    aplt_user_scores = []
    novelties = []
    item_exposure_counts: Dict[int, int] = {}
    recommended_tail_items: Set[int] = set()

    # Precompute item popularity probability for self-information (Novelty)
    total_train_ratings = sum(len(interactions) for interactions in train_dict.values())
    item_train_counts = {}
    for u, interactions in train_dict.items():
        for item, _, _ in interactions:
            item_train_counts[item] = item_train_counts.get(item, 0) + 1

    item_self_info = {}
    for item, count in item_train_counts.items():
        p_i = count / max(1, total_train_ratings)
        item_self_info[item] = -np.log2(max(1e-10, p_i))

    for u in eval_users:
        rec_list = recommendations[u][:top_k]
        test_item, test_rating, _ = test_dict[u]

        # 1. Recall / Hit Rate
        if test_item in rec_list:
            hits += 1
            rank = rec_list.index(test_item)
            ndcgs.append(1.0 / np.log2(rank + 2))
        else:
            ndcgs.append(0.0)

        # 2. Calibration (RMSE-PC)
        pop_u_rec = sum(1 for item in rec_list if item in head_items) / max(1, len(rec_list))
        target_pop_u = user_inclinations.get(u, 0.20)
        rmse_pc_errors.append((pop_u_rec - target_pop_u) ** 2)

        # 3. MRMC (Rank Miscalibration)
        # Cumulative head fraction divergence across ranks
        cum_head_rec = np.cumsum([1.0 if item in head_items else 0.0 for item in rec_list]) / np.arange(1, len(rec_list) + 1)
        rank_divergence = np.mean(np.abs(cum_head_rec - target_pop_u))
        mrmc_errors.append(rank_divergence)

        # 4. APLT (Average Percentage of Long-Tail items)
        tail_count = sum(1 for item in rec_list if item in tail_items)
        aplt_user_scores.append(tail_count / max(1, len(rec_list)))

        # 5. Novelty
        user_novelty = np.mean([item_self_info.get(item, 15.0) for item in rec_list])
        novelties.append(user_novelty)

        # 6. Global Exposure Tracking
        for item in rec_list:
            item_exposure_counts[item] = item_exposure_counts.get(item, 0) + 1
            if item in tail_items:
                recommended_tail_items.add(item)

    n_eval = len(eval_users)
    recall = hits / n_eval
    ndcg = float(np.mean(ndcgs))
    rmse_pc = float(np.sqrt(np.mean(rmse_pc_errors)))
    mrmc = float(np.mean(mrmc_errors))
    aplt = float(np.mean(aplt_user_scores))
    novelty = float(np.mean(novelties))

    # 7. Long-Tail Coverage (LTC)
    total_tail_items = max(1, len(tail_items))
    ltc = len(recommended_tail_items) / total_tail_items

    # 8. Entropy across catalog
    total_recs = sum(item_exposure_counts.values())
    if total_recs > 0:
        probs = np.array(list(item_exposure_counts.values())) / total_recs
        entropy = -np.sum(probs * np.log2(probs + 1e-12))
        max_entropy = np.log2(max(1, len(item_exposure_counts)))
        norm_entropy = float(entropy / max_entropy) if max_entropy > 0 else 0.0
    else:
        norm_entropy = 0.0

    # 9. GKPI (Harmonic Combination with nDCG)
    # H(x, y) = 2xy / (x + y)
    def harmonic_mean(x, y):
        return (2.0 * x * y) / (x + y + 1e-8)

    # Normalize novelty to [0, 1] range for GKPI computation (max ~ 20 bits)
    norm_novelty = min(1.0, novelty / 20.0)

    h_aplt = harmonic_mean(ndcg, aplt)
    h_nov = harmonic_mean(ndcg, norm_novelty)
    h_ent = harmonic_mean(ndcg, norm_entropy)
    h_ltc = harmonic_mean(ndcg, ltc)

    gkpi = float((h_aplt + h_nov + h_ent + h_ltc) / 4.0)

    return {
        f"Recall@{top_k}": recall,
        f"nDCG@{top_k}": ndcg,
        f"RMSE-PC": rmse_pc,
        f"MRMC": mrmc,
        f"APLT@{top_k}": aplt,
        f"LTC@{top_k}": ltc,
        f"Entropy": norm_entropy,
        f"Novelty": novelty,
        f"GKPI": gkpi
    }
