"""
fedsuper_simulation/src/super_engine.py
SUPER Popularity Blueprint Calibration & LLM Semantic Cosine Fusion Engine.
Fuses intra-pool z-score standardized collaborative filtering scores with LLM semantic similarities,
and enforces user historical popularity blueprints via Largest Remainder integer quota balancing.
"""
from typing import Tuple, Dict, Any, Optional, Union, List
import numpy as np


def compute_user_popularity_blueprint(
    train_matrix: np.ndarray,
    head_idx: np.ndarray,
    torso_idx: np.ndarray,
    tail_idx: np.ndarray
) -> np.ndarray:
    """
    Computes user target popularity preference blueprint:
    P_u = [p_u(head), p_u(torso), p_u(tail)] in Delta^2.

    Parameters:
        train_matrix: (num_users, num_items) binary interaction matrix.
        head_idx: Indices of Head items.
        torso_idx: Indices of Torso items.
        tail_idx: Indices of Tail items.

    Returns:
        (num_users, 3) float32 matrix summing to 1.0 per row.
    """
    num_users, num_items = train_matrix.shape
    head_set = set(np.asarray(head_idx, dtype=np.int64).tolist())
    torso_set = set(np.asarray(torso_idx, dtype=np.int64).tolist())
    tail_set = set(np.asarray(tail_idx, dtype=np.int64).tolist())

    head_cols = np.array(list(head_set), dtype=np.int64)
    torso_cols = np.array(list(torso_set), dtype=np.int64)
    tail_cols = np.array(list(tail_set), dtype=np.int64)

    c_head = train_matrix[:, head_cols].sum(axis=1) if len(head_cols) > 0 else np.zeros(num_users)
    c_torso = train_matrix[:, torso_cols].sum(axis=1) if len(torso_cols) > 0 else np.zeros(num_users)
    c_tail = train_matrix[:, tail_cols].sum(axis=1) if len(tail_cols) > 0 else np.zeros(num_users)

    c_total = c_head + c_torso + c_tail

    blueprints = np.zeros((num_users, 3), dtype=np.float32)
    active_mask = c_total > 0

    blueprints[active_mask, 0] = c_head[active_mask] / c_total[active_mask]
    blueprints[active_mask, 1] = c_torso[active_mask] / c_total[active_mask]
    blueprints[active_mask, 2] = c_tail[active_mask] / c_total[active_mask]

    # Cold-start uniform fallback for users with 0 interactions
    blueprints[~active_mask] = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float32)

    return blueprints


def intra_pool_zscore_standardization(
    cf_scores: np.ndarray,
    pool_idx: np.ndarray
) -> np.ndarray:
    """
    Standardizes CF scores within specified item pool across items for each user.
    z_CF(u, i) = (S_CF(u, i) - mu_u^pool) / (sigma_u^pool + 1e-8)

    Parameters:
        cf_scores: (num_users, num_items) CF score matrix.
        pool_idx: 1D array of item indices in the target tier.

    Returns:
        (num_users, len(pool_idx)) standardized scores.
    """
    if len(pool_idx) == 0:
        return np.zeros((cf_scores.shape[0], 0), dtype=np.float32)

    pool_sub = cf_scores[:, pool_idx].astype(np.float32)
    mean = np.mean(pool_sub, axis=1, keepdims=True)
    std = np.std(pool_sub, axis=1, keepdims=True)
    safe_std = np.where(std > 1e-8, std, 1.0)
    return (pool_sub - mean) / safe_std


def fuse_cf_and_llm_scores(
    cf_scores: np.ndarray,
    user_profiles: np.ndarray,
    item_embeddings: np.ndarray,
    head_idx: np.ndarray,
    torso_idx: np.ndarray,
    tail_idx: np.ndarray,
    llm_lambda: float = 0.70
) -> np.ndarray:
    """
    Computes intra-pool z-score standardized CF scores and LLM semantic cosine similarities,
    fusing them via convex combination: S_fused = (1 - lambda) * z_CF + lambda * S_LLM.

    Parameters:
        cf_scores: (num_users, num_items) predicted CF scores.
        user_profiles: (num_users, llm_dim) normalized user semantic vectors s_u.
        item_embeddings: (num_items, llm_dim) normalized item semantic vectors v_i.
        head_idx: Indices of Head items.
        torso_idx: Indices of Torso items.
        tail_idx: Indices of Tail items.
        llm_lambda: Weight for LLM semantic score in [0, 1].

    Returns:
        (num_users, num_items) fused score matrix.
    """
    num_users, num_items = cf_scores.shape
    lam = float(np.clip(llm_lambda, 0.0, 1.0))

    # 1. LLM Cosine Similarity: S_LLM(u, i) = s_u . v_i in [-1, 1]
    u_norm = np.linalg.norm(user_profiles, axis=1, keepdims=True)
    safe_u = np.where(u_norm > 1e-12, user_profiles / u_norm, 0.0)

    v_norm = np.linalg.norm(item_embeddings, axis=1, keepdims=True)
    safe_v = np.where(v_norm > 1e-12, item_embeddings / v_norm, 0.0)

    s_llm = np.dot(safe_u, safe_v.T).astype(np.float32)  # (num_users, num_items)

    # 2. Intra-Pool Standardization and Convex Fusion
    fused = np.zeros((num_users, num_items), dtype=np.float32)

    for pool in [head_idx, torso_idx, tail_idx]:
        if len(pool) == 0:
            continue
        z_cf_pool = intra_pool_zscore_standardization(cf_scores, pool)
        s_llm_pool = s_llm[:, pool]
        fused[:, pool] = (1.0 - lam) * z_cf_pool + lam * s_llm_pool

    return fused


def calibrated_blueprint_merge(
    fused_scores: np.ndarray,
    user_blueprints: np.ndarray,
    *args,
    **kwargs
) -> np.ndarray:
    """
    Executes SUPER Blueprint Quota Allocation and Candidate Interleaving.
    Guarantees exact sum-to-K integer quota balancing via Largest Remainder method.

    Flexible signature accepts either:
      calibrated_blueprint_merge(fused_scores, user_blueprints, train_matrix, head_idx, torso_idx, tail_idx, top_k=10, calibration_alpha=0.40)
      OR
      calibrated_blueprint_merge(fused_scores, user_blueprints, head_idx, torso_idx, tail_idx, top_k=10, calibration_alpha=0.40, train_matrix=None)
    """
    num_users, num_items = fused_scores.shape

    # Parse polymorphic arguments
    train_matrix = kwargs.get("train_matrix", None)
    head_idx = kwargs.get("head_idx", None)
    torso_idx = kwargs.get("torso_idx", None)
    tail_idx = kwargs.get("tail_idx", None)
    top_k = kwargs.get("top_k", 10)
    calibration_alpha = kwargs.get("calibration_alpha", 0.40)

    # Check positional args
    pos_args = list(args)
    if pos_args:
        # If first positional arg is 2D array, it's train_matrix
        if hasattr(pos_args[0], "ndim") and pos_args[0].ndim == 2:
            train_matrix = pos_args.pop(0)
        
        # Remaining positional args map to head_idx, torso_idx, tail_idx, top_k, calibration_alpha
        if len(pos_args) >= 1 and head_idx is None:
            head_idx = pos_args[0]
        if len(pos_args) >= 2 and torso_idx is None:
            torso_idx = pos_args[1]
        if len(pos_args) >= 3 and tail_idx is None:
            tail_idx = pos_args[2]
        if len(pos_args) >= 4 and "top_k" not in kwargs:
            top_k = pos_args[3]
        if len(pos_args) >= 5 and "calibration_alpha" not in kwargs:
            calibration_alpha = pos_args[4]

    k_val = int(top_k)
    alpha = float(np.clip(calibration_alpha, 0.0, 1.0))

    head_idx = np.asarray(head_idx if head_idx is not None else [], dtype=np.int64)
    torso_idx = np.asarray(torso_idx if torso_idx is not None else [], dtype=np.int64)
    tail_idx = np.asarray(tail_idx if tail_idx is not None else [], dtype=np.int64)

    head_set = set(head_idx.tolist())
    torso_set = set(torso_idx.tolist())
    tail_set = set(tail_idx.tolist())

    # Mask training interactions if train_matrix is provided
    masked_scores = fused_scores.copy()
    if train_matrix is not None:
        masked_scores[train_matrix > 0] = -np.inf

    recommendations = np.zeros((num_users, k_val), dtype=np.int64)

    for u in range(num_users):
        u_scores = masked_scores[u]
        p_target = user_blueprints[u]  # [p_head, p_torso, p_tail]

        # 1. Determine Uncalibrated Distribution q_u^uncalib
        valid_items = np.where(u_scores > -np.inf)[0]
        if len(valid_items) < k_val:
            # Fallback if catalog has fewer unrated items than K
            top_raw = np.argsort(-u_scores)[:k_val]
            recommendations[u] = top_raw
            continue

        raw_top_k = valid_items[np.argsort(-u_scores[valid_items])[:k_val]]
        n_uncalib_h = sum(1 for it in raw_top_k if it in head_set)
        n_uncalib_t = sum(1 for it in raw_top_k if it in torso_set)
        n_uncalib_l = sum(1 for it in raw_top_k if it in tail_set)
        q_uncalib = np.array([n_uncalib_h / k_val, n_uncalib_t / k_val, n_uncalib_l / k_val], dtype=np.float64)

        # 2. Interpolate Continuous Proportions: pi_u = (1 - alpha) * q_uncalib + alpha * p_target
        pi = (1.0 - alpha) * q_uncalib + alpha * p_target.astype(np.float64)
        sum_pi = np.sum(pi)
        if sum_pi > 0:
            pi = pi / sum_pi
        else:
            pi = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float64)

        # 3. Largest Remainder / Hamilton-Hare Integer Quota Balancing
        exact_quotas = pi * k_val
        floor_quotas = np.floor(exact_quotas).astype(np.int64)
        residuals = exact_quotas - floor_quotas
        deficit = k_val - int(np.sum(floor_quotas))

        # Tie-breaker priority: tail (2) -> torso (1) -> head (0) for diversity
        order_idx = np.lexsort(([0, 1, 2], -residuals))
        for r_rank in range(deficit):
            floor_quotas[order_idx[r_rank]] += 1

        k_head = int(floor_quotas[0])
        k_torso = int(floor_quotas[1])
        k_tail = int(floor_quotas[2])

        # 4. Extract Candidate Pools Sorted by Score
        cand_head = [it for it in head_idx if u_scores[it] > -np.inf]
        cand_head.sort(key=lambda x: u_scores[x], reverse=True)

        cand_torso = [it for it in torso_idx if u_scores[it] > -np.inf]
        cand_torso.sort(key=lambda x: u_scores[x], reverse=True)

        cand_tail = [it for it in tail_idx if u_scores[it] > -np.inf]
        cand_tail.sort(key=lambda x: u_scores[x], reverse=True)

        # Pool exhaustion safety clamps
        k_head_act = min(k_head, len(cand_head))
        k_torso_act = min(k_torso, len(cand_torso))
        k_tail_act = min(k_tail, len(cand_tail))

        selected = cand_head[:k_head_act] + cand_torso[:k_torso_act] + cand_tail[:k_tail_act]

        # 5. Greedy Residual Backfill if pools ran short
        if len(selected) < k_val:
            selected_set = set(selected)
            all_sorted = np.argsort(-u_scores)
            for it in all_sorted:
                it_id = int(it)
                if u_scores[it_id] > -np.inf and it_id not in selected_set:
                    selected.append(it_id)
                    selected_set.add(it_id)
                    if len(selected) >= k_val:
                        break

        # Edge fallback if still short (include interacted items if necessary to reach K)
        if len(selected) < k_val:
            selected_set = set(selected)
            for it_id in range(num_items):
                if it_id not in selected_set:
                    selected.append(it_id)
                    selected_set.add(it_id)
                    if len(selected) >= k_val:
                        break

        # Re-sort final selected slate by fused score descending
        selected_k = selected[:k_val]
        selected_k.sort(key=lambda x: u_scores[x] if u_scores[x] > -np.inf else -1e9, reverse=True)
        recommendations[u] = np.array(selected_k, dtype=np.int64)

    return recommendations
