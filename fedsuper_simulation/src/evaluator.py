"""
fedsuper_simulation/src/evaluator.py
Comprehensive evaluation metrics engine for Privacy-Preserving Federated SUPER.
Computes beyond-accuracy (fairness, calibration, diversity, novelty) and accuracy metrics.
"""
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def calculate_rmse_pc(
    user_blueprints: np.ndarray,
    rec_distributions: np.ndarray
) -> float:
    """
    Calculates Macro-averaged Root Mean Squared Error of Popularity Calibration (Rmse-PC).

    Parameters:
      user_blueprints: (U, C) array of user target popularity preferences P_u.
                       Typically C=3 for [Head, Torso, Tail].
      rec_distributions: (U, C) array of recommendation popularity shares Q_u.

    Returns:
      rmse_pc: Macro-averaged calibration error across all users in [0.0, 1.0].
    """
    p = np.asarray(user_blueprints, dtype=np.float64)
    q = np.asarray(rec_distributions, dtype=np.float64)

    if p.ndim == 1:
        p = p.reshape(1, -1)
    if q.ndim == 1:
        q = q.reshape(1, -1)

    if len(p) == 0 or len(q) == 0:
        return 0.0

    if p.shape != q.shape:
        raise ValueError(f"Shape mismatch in calculate_rmse_pc: {p.shape} vs {q.shape}")

    # Per-user Root Mean Squared Error across popularity classes
    # CE(u) = sqrt( mean_c( (p_u,c - q_u,c)^2 ) )
    ce_per_user = np.sqrt(np.mean((p - q) ** 2, axis=1))
    
    return float(np.mean(ce_per_user))


def calculate_gini_index(exposure_counts: np.ndarray) -> float:
    """
    Calculates Gini coefficient of item recommendation exposures across catalog.
    G = 0: Perfectly fair / uniform exposure across all items.
    G -> 1: Severe popularity bias / monopoly exposure.

    Parameters:
      exposure_counts: (M,) array of exposure counts per item.

    Returns:
      gini: Float in [0.0, 1.0].
    """
    counts = np.asarray(exposure_counts, dtype=np.float64).flatten()
    n = len(counts)
    if n <= 1:
        return 0.0

    total = np.sum(counts)
    if total <= 0.0:
        return 0.0

    sorted_counts = np.sort(counts)
    ranks = np.arange(1, n + 1, dtype=np.float64)
    
    # Standard rank-weighted Gini formula: (2 * sum(j * E_j)) / (n * total) - (n + 1) / n
    gini = (2.0 * np.sum(ranks * sorted_counts)) / (n * total) - (n + 1.0) / n
    return float(np.clip(gini, 0.0, 1.0))


def calculate_long_tail_exposure_ratio(
    exposure_counts: np.ndarray,
    tail_idx: np.ndarray
) -> float:
    """
    Calculates Long-Tail Exposure Ratio (LTER): fraction of total recommendation
    slots allocated to items in the Tail partition.

    Parameters:
      exposure_counts: (M,) array of item recommendation counts.
      tail_idx: (N_tail,) array of tail item indices.

    Returns:
      lter: Float in [0.0, 1.0].
    """
    counts = np.asarray(exposure_counts, dtype=np.float64).flatten()
    total_exp = np.sum(counts)
    if total_exp <= 0.0 or len(tail_idx) == 0:
        return 0.0

    valid_tail_idx = np.asarray(tail_idx, dtype=np.int64)
    valid_tail_idx = valid_tail_idx[(valid_tail_idx >= 0) & (valid_tail_idx < len(counts))]
    if len(valid_tail_idx) == 0:
        return 0.0
    
    tail_exp = np.sum(counts[valid_tail_idx])
    return float(np.clip(tail_exp / total_exp, 0.0, 1.0))


def calculate_catalog_coverage(exposure_counts: np.ndarray) -> float:
    """
    Calculates Catalog Coverage: fraction of catalog items receiving >= 1 exposure.

    Parameters:
      exposure_counts: (M,) array of item exposure counts.

    Returns:
      coverage: Float ratio in [0.0, 1.0].
    """
    counts = np.asarray(exposure_counts, dtype=np.float64).flatten()
    m = len(counts)
    if m == 0:
        return 0.0

    exposed = np.count_nonzero(counts > 0)
    return float(exposed / m)


def calculate_recall_at_k(
    recommendations: np.ndarray,
    test_dict: Dict[int, List[int]],
    top_k: int = 10
) -> float:
    """
    Calculates Macro-averaged Recall@K across evaluation users.

    Parameters:
      recommendations: (U, top_k) array of recommended item IDs.
      test_dict: Dict mapping user_id -> list of true positive test item IDs.
      top_k: Cutoff rank K.

    Returns:
      recall: Float in [0.0, 1.0].
    """
    recs = np.asarray(recommendations)
    if recs.ndim == 1:
        recs = recs.reshape(1, -1)
    k_eff = min(top_k, recs.shape[1]) if recs.ndim == 2 and recs.shape[1] > 0 else 0
    if k_eff == 0:
        return 0.0

    hits = 0.0
    evaluated_users = 0

    for u, ground_truth in test_dict.items():
        if u >= len(recs) or not ground_truth:
            continue
        gt_set = set(ground_truth)
        rec_list = recs[u, :k_eff]
        num_hits = len(gt_set.intersection(rec_list))
        hits += num_hits / max(1, len(gt_set))
        evaluated_users += 1

    if evaluated_users == 0:
        return 0.0
    return float(hits / evaluated_users)


def calculate_ndcg_at_k(
    recommendations: np.ndarray,
    test_dict: Dict[int, List[int]],
    top_k: int = 10
) -> float:
    """
    Calculates Macro-averaged Normalized Discounted Cumulative Gain at K (nDCG@K).

    Parameters:
      recommendations: (U, top_k) array of recommended item IDs.
      test_dict: Dict mapping user_id -> list of true positive test item IDs.
      top_k: Cutoff rank K.

    Returns:
      ndcg: Float in [0.0, 1.0].
    """
    recs = np.asarray(recommendations)
    if recs.ndim == 1:
        recs = recs.reshape(1, -1)
    k_eff = min(top_k, recs.shape[1]) if recs.ndim == 2 and recs.shape[1] > 0 else 0
    if k_eff == 0:
        return 0.0

    total_ndcg = 0.0
    evaluated_users = 0

    for u, ground_truth in test_dict.items():
        if u >= len(recs) or not ground_truth:
            continue
        gt_set = set(ground_truth)
        rec_list = recs[u, :k_eff]

        # Calculate DCG@K
        dcg = 0.0
        for rank_idx, item_id in enumerate(rec_list):
            if item_id in gt_set:
                dcg += 1.0 / np.log2(rank_idx + 2.0)  # rank_idx 0 -> position 1 -> log2(2)

        # Calculate IDCG@K
        num_pos = min(len(gt_set), k_eff)
        idcg = sum(1.0 / np.log2(r + 2.0) for r in range(num_pos))

        if idcg > 0.0:
            total_ndcg += dcg / idcg
        evaluated_users += 1

    if evaluated_users == 0:
        return 0.0
    return float(total_ndcg / evaluated_users)


def calculate_novelty(
    recommendations: np.ndarray,
    item_popularity: np.ndarray,
    total_interactions: Optional[int] = None
) -> float:
    """
    Calculates Mean Self-Information / Novelty of recommendations.
    Novelty = -log2( P(item) ) in bits.

    Parameters:
      recommendations: (U, top_k) array of recommended item IDs.
      item_popularity: (M,) array of historical interaction counts per item.
      total_interactions: Total training interaction count (sum(item_popularity)).

    Returns:
      novelty: Average novelty in bits (higher = more novel / unpopular items recommended).
    """
    recs = np.asarray(recommendations)
    pop = np.asarray(item_popularity, dtype=np.float64)
    m = len(pop)
    if recs.size == 0 or m == 0:
        return 0.0

    if total_interactions is None or total_interactions <= 0:
        total_interactions = int(np.sum(pop))
    if total_interactions <= 0:
        return 0.0

    # Laplace smoothing to prevent log2(0)
    eps = 1e-12
    p_item = (pop + eps) / (float(total_interactions) + m * eps)
    self_info = -np.log2(p_item)

    # Lookup self-information for all recommended items
    valid_recs = np.clip(recs, 0, m - 1)
    nov_matrix = self_info[valid_recs]
    return float(np.mean(nov_matrix))


def calculate_ild(
    recommendations: np.ndarray,
    item_embeddings: np.ndarray
) -> float:
    """
    Calculates Macro-averaged Intra-List Diversity (ILD) using pairwise cosine distance.

    Parameters:
      recommendations: (U, top_k) array of recommended item IDs.
      item_embeddings: (M, d) matrix of normalized item semantic/latent vectors.

    Returns:
      ild: Float in [0.0, 1.0].
    """
    recs = np.asarray(recommendations)
    if recs.ndim == 1:
        recs = recs.reshape(1, -1)
    embs = np.asarray(item_embeddings, dtype=np.float64)
    u_count, k = recs.shape

    if u_count == 0 or k <= 1 or len(embs) == 0:
        return 0.0

    ild_per_user = []
    triu_i, triu_j = np.triu_indices(k, k=1)

    for u in range(u_count):
        user_items = np.clip(recs[u], 0, len(embs) - 1)
        sub_embs = embs[user_items]  # (K, d)
        
        norms = np.linalg.norm(sub_embs, axis=1, keepdims=True)
        norms = np.where(norms > 1e-9, norms, 1.0)
        norm_embs = sub_embs / norms
        
        sim_matrix = norm_embs @ norm_embs.T  # (K, K)
        dist_matrix = 1.0 - sim_matrix
        
        pairwise_dists = dist_matrix[triu_i, triu_j]
        ild_per_user.append(np.mean(pairwise_dists))

    return float(np.clip(np.mean(ild_per_user), 0.0, 1.0))


def compute_exposure_counts(recommendations: np.ndarray, num_items: int) -> np.ndarray:
    """
    Computes total recommendation exposure counts for all items in catalog.
    """
    recs = np.asarray(recommendations).flatten()
    valid_recs = recs[(recs >= 0) & (recs < num_items)]
    return np.bincount(valid_recs, minlength=num_items).astype(np.int64)


def compute_recommendation_distributions(
    recommendations: np.ndarray,
    head_idx: np.ndarray,
    torso_idx: np.ndarray,
    tail_idx: np.ndarray
) -> np.ndarray:
    """
    Computes (U, 3) matrix of recommendation shares [q_head, q_torso, q_tail] per user.
    """
    recs = np.asarray(recommendations)
    if recs.ndim == 1:
        recs = recs.reshape(1, -1)
    u_count, k = recs.shape
    if k == 0:
        return np.zeros((u_count, 3), dtype=np.float64)

    is_head = np.isin(recs, head_idx).mean(axis=1)
    is_torso = np.isin(recs, torso_idx).mean(axis=1)
    is_tail = np.isin(recs, tail_idx).mean(axis=1)

    return np.column_stack([is_head, is_torso, is_tail])


def evaluate_all_metrics(
    dataset: Any,
    recs_uncalib: np.ndarray,
    recs_calib: np.ndarray,
    top_k: int = 10
) -> Dict[str, Any]:
    """
    Comprehensive evaluation pipeline comparing uncalibrated vs SUPER calibrated recommendations.

    Parameters:
      dataset: SyntheticDataset instance containing metadata, train_matrix, test_dict, partitions.
      recs_uncalib: (U, top_k) array of uncalibrated recommendations.
      recs_calib: (U, top_k) array of SUPER calibrated recommendations.
      top_k: Cutoff rank K.

    Returns:
      Dict structured for dashboard charts and KPI cards.
    """
    # 1. Compute user blueprints
    train_mat = dataset.train_matrix
    user_tot = train_mat.sum(axis=1)
    
    head_cnt = train_mat[:, dataset.head_idx].sum(axis=1)
    torso_cnt = train_mat[:, dataset.torso_idx].sum(axis=1)
    tail_cnt = train_mat[:, dataset.tail_idx].sum(axis=1)

    with np.errstate(divide='ignore', invalid='ignore'):
        p_head = np.where(user_tot > 0, head_cnt / user_tot, 1.0 / 3.0)
        p_torso = np.where(user_tot > 0, torso_cnt / user_tot, 1.0 / 3.0)
        p_tail = np.where(user_tot > 0, tail_cnt / user_tot, 1.0 / 3.0)
    user_blueprints = np.column_stack([p_head, p_torso, p_tail])

    # 2. Recommendation distributions
    q_uncalib = compute_recommendation_distributions(recs_uncalib, dataset.head_idx, dataset.torso_idx, dataset.tail_idx)
    q_calib = compute_recommendation_distributions(recs_calib, dataset.head_idx, dataset.torso_idx, dataset.tail_idx)

    # 3. Exposure counts
    exp_uncalib = compute_exposure_counts(recs_uncalib, dataset.num_items)
    exp_calib = compute_exposure_counts(recs_calib, dataset.num_items)

    # 4. Item popularity counts
    if hasattr(dataset, "item_metadata") and hasattr(dataset.item_metadata, "__getitem__") and "popularity_count" in dataset.item_metadata:
        item_pop = dataset.item_metadata["popularity_count"].values
    else:
        item_pop = dataset.train_matrix.sum(axis=0)
    tot_inter = int(np.sum(item_pop))

    # 5. Evaluate Metrics for Uncalibrated
    rmse_pc_uncalib = calculate_rmse_pc(user_blueprints, q_uncalib)
    gini_uncalib = calculate_gini_index(exp_uncalib)
    lter_uncalib = calculate_long_tail_exposure_ratio(exp_uncalib, dataset.tail_idx)
    cov_uncalib = calculate_catalog_coverage(exp_uncalib)
    rec_uncalib = calculate_recall_at_k(recs_uncalib, dataset.test_dict, top_k)
    ndcg_uncalib = calculate_ndcg_at_k(recs_uncalib, dataset.test_dict, top_k)
    nov_uncalib = calculate_novelty(recs_uncalib, item_pop, tot_inter)
    ild_uncalib = calculate_ild(recs_uncalib, dataset.item_llm_embeddings)

    # 6. Evaluate Metrics for Calibrated
    rmse_pc_calib = calculate_rmse_pc(user_blueprints, q_calib)
    gini_calib = calculate_gini_index(exp_calib)
    lter_calib = calculate_long_tail_exposure_ratio(exp_calib, dataset.tail_idx)
    cov_calib = calculate_catalog_coverage(exp_calib)
    rec_calib = calculate_recall_at_k(recs_calib, dataset.test_dict, top_k)
    ndcg_calib = calculate_ndcg_at_k(recs_calib, dataset.test_dict, top_k)
    nov_calib = calculate_novelty(recs_calib, item_pop, tot_inter)
    ild_calib = calculate_ild(recs_calib, dataset.item_llm_embeddings)

    # Per-user calibration errors
    ce_uncalib = np.sqrt(np.mean((user_blueprints - q_uncalib) ** 2, axis=1))
    ce_calib = np.sqrt(np.mean((user_blueprints - q_calib) ** 2, axis=1))

    # Exposure shares
    sum_exp_un = max(float(np.sum(exp_uncalib)), 1.0)
    sum_exp_ca = max(float(np.sum(exp_calib)), 1.0)

    # KPI percentage improvements
    rmse_pc_imp = ((rmse_pc_uncalib - rmse_pc_calib) / max(rmse_pc_uncalib, 1e-6)) * 100.0
    gini_red = ((gini_uncalib - gini_calib) / max(gini_uncalib, 1e-6)) * 100.0
    lter_boost = ((lter_calib - lter_uncalib) / max(lter_uncalib, 1e-6)) * 100.0
    cov_boost = ((cov_calib - cov_uncalib) / max(cov_uncalib, 1e-6)) * 100.0

    return {
        "uncalibrated": {
            "rmse_pc": float(rmse_pc_uncalib),
            "gini_index": float(gini_uncalib),
            "long_tail_exposure_ratio": float(lter_uncalib),
            "catalog_coverage": float(cov_uncalib),
            "recall_at_k": float(rec_uncalib),
            "ndcg_at_k": float(ndcg_uncalib),
            "novelty": float(nov_uncalib),
            "ild": float(ild_uncalib),
            "head_exposure_ratio": float(np.sum(exp_uncalib[dataset.head_idx]) / sum_exp_un),
            "torso_exposure_ratio": float(np.sum(exp_uncalib[dataset.torso_idx]) / sum_exp_un),
            "tail_exposure_ratio": float(np.sum(exp_uncalib[dataset.tail_idx]) / sum_exp_un),
            "exposure_counts": exp_uncalib,
            "per_user_ce": ce_uncalib,
        },
        "calibrated": {
            "rmse_pc": float(rmse_pc_calib),
            "gini_index": float(gini_calib),
            "long_tail_exposure_ratio": float(lter_calib),
            "catalog_coverage": float(cov_calib),
            "recall_at_k": float(rec_calib),
            "ndcg_at_k": float(ndcg_calib),
            "novelty": float(nov_calib),
            "ild": float(ild_calib),
            "head_exposure_ratio": float(np.sum(exp_calib[dataset.head_idx]) / sum_exp_ca),
            "torso_exposure_ratio": float(np.sum(exp_calib[dataset.torso_idx]) / sum_exp_ca),
            "tail_exposure_ratio": float(np.sum(exp_calib[dataset.tail_idx]) / sum_exp_ca),
            "exposure_counts": exp_calib,
            "per_user_ce": ce_calib,
        },
        "blueprint": {
            "head_ratio": float(np.mean(user_blueprints[:, 0])),
            "torso_ratio": float(np.mean(user_blueprints[:, 1])),
            "tail_ratio": float(np.mean(user_blueprints[:, 2])),
            "user_blueprints": user_blueprints,
        },
        "kpis": {
            "rmse_pc_improvement_pct": float(rmse_pc_imp),
            "gini_reduction_pct": float(gini_red),
            "lter_boost_pct": float(lter_boost),
            "coverage_boost_pct": float(cov_boost),
        }
    }
