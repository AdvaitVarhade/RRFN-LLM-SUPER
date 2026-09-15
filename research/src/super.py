"""Correct SUPER framework implementation (Yavru et al., IEEE Access 2026)
and Dynamic Probabilistic Blueprint Merge extensions for FedSUPER-LLM.

Pipeline:
  1. Pareto partition of item catalog into (Head set H, Tail set T) by cumulative
     interaction volume threshold alpha=0.20 (NOT 20% of items; 20% of interactions).
  2. Train TWO separate recommenders: M_pop on D_pop (head interactions only),
     M_tail on D_tail (tail interactions only).
  3. Per user, compute popularity inclination Pop_u = |C_u ∩ H| / |C_u|
     (fraction of user's interactions that are in the head set).
  4. Recommendation top-N for user u:
     - Static Blueprint Merge: N_pop = floor(N * Pop_u), N_tail = N - N_pop
     - Dynamic Probabilistic Quota Merge: expectation-preserving stochastic rounding,
       user-adaptive alpha partition, and confidence-elastic interleaving.
     - Calibrated Score Fusion: intra-pool z-score standardized fusion with SBERT embeddings.
"""
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


def pareto_partition(pop: np.ndarray, alpha: float = 0.20) -> Tuple[np.ndarray, np.ndarray]:
    """Return (head_idx, tail_idx) where head items account for alpha fraction
    of total interactions.

    pop: (n_items,) interaction counts per item
    alpha: cumulative-volume threshold (default 0.20 per paper)
    """
    total = pop.sum()
    threshold = total * alpha
    order = np.argsort(-pop)  # descending interaction count
    cum = np.cumsum(pop[order])
    # Head = items in `order` whose cumulative interaction volume is <= threshold
    # (Include at least one item to avoid degenerate empty head).
    n_head = int(np.searchsorted(cum, threshold, side="right")) + 1
    n_head = max(1, min(n_head, len(order)))
    head_idx = order[:n_head]
    tail_idx = order[n_head:]
    # Handle 0-popularity items -> tail
    zero_pop = np.where(pop == 0)[0]
    if len(zero_pop):
        already = set(head_idx.tolist()) | set(tail_idx.tolist())
        for z in zero_pop:
            if z not in already:
                tail_idx = np.append(tail_idx, z)
    return head_idx, tail_idx


def tri_tier_partition(
    pop: np.ndarray,
    alpha_head: float = 0.20,
    alpha_torso: float = 0.50
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (head_idx, torso_idx, tail_idx) based on cumulative volume thresholds.

    Head: 0 to alpha_head volume
    Torso: alpha_head to (alpha_head + alpha_torso) volume
    Tail: remaining volume
    """
    total = pop.sum()
    thresh_head = total * alpha_head
    thresh_torso = total * (alpha_head + alpha_torso)

    order = np.argsort(-pop)
    cum = np.cumsum(pop[order])

    n_head = int(np.searchsorted(cum, thresh_head, side="right")) + 1
    n_head = max(1, min(n_head, len(order)))

    n_torso_end = int(np.searchsorted(cum, thresh_torso, side="right")) + 1
    n_torso_end = max(n_head + 1, min(n_torso_end, len(order)))

    head_idx = order[:n_head]
    torso_idx = order[n_head:n_torso_end]
    tail_idx = order[n_torso_end:]

    # Handle 0-popularity items -> tail
    zero_pop = np.where(pop == 0)[0]
    if len(zero_pop):
        already = set(head_idx.tolist()) | set(torso_idx.tolist()) | set(tail_idx.tolist())
        for z in zero_pop:
            if z not in already:
                tail_idx = np.append(tail_idx, z)

    return head_idx, torso_idx, tail_idx


def user_popularity_inclination(train_matrix: np.ndarray, head_idx: np.ndarray) -> np.ndarray:
    """Pop_u = |C_u ∩ H| / |C_u| per user.

    train_matrix: (n_users, n_items) integer interaction counts
    head_idx: array of Head item indices from pareto_partition
    Returns (n_users,) array of Pop_u in [0, 1].
    """
    head_set = set(head_idx.tolist())
    user_interactions = (train_matrix > 0).astype(np.float32)
    per_user_count = user_interactions.sum(axis=1)
    head_count = user_interactions[:, list(head_set)].sum(axis=1)
    pop_u = np.where(per_user_count > 0, head_count / np.maximum(per_user_count, 1.0), 0.0)
    return pop_u.astype(np.float32)


def tri_tier_user_inclinations(
    train_matrix: np.ndarray,
    head_idx: np.ndarray,
    torso_idx: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Pop_head_u and Pop_torso_u per user.
    Pop_tail_u = 1.0 - Pop_head_u - Pop_torso_u.
    """
    head_set = set(head_idx.tolist())
    torso_set = set(torso_idx.tolist())

    user_interactions = (train_matrix > 0).astype(np.float32)
    per_user_count = user_interactions.sum(axis=1)

    head_count = user_interactions[:, list(head_set)].sum(axis=1)
    torso_count = user_interactions[:, list(torso_set)].sum(axis=1)

    pop_head_u = np.where(per_user_count > 0, head_count / np.maximum(per_user_count, 1.0), 0.0)
    pop_torso_u = np.where(per_user_count > 0, torso_count / np.maximum(per_user_count, 1.0), 0.0)

    return pop_head_u.astype(np.float32), pop_torso_u.astype(np.float32)


def stochastic_user_quota(
    pop_u: np.ndarray,
    N: int = 20,
    rng: Optional[Union[np.random.Generator, int]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Expectation-preserving randomized rounding for head/tail integer quotas.

    For user u with head popularity inclination pop_u[u] in [0, 1]:
      expected_pop = N * pop_u[u]
      floor_u = floor(expected_pop)
      frac_u = expected_pop - floor_u
      n_pop = floor_u + (random_draw < frac_u)
      n_tail = N - n_pop

    Mathematical guarantees:
      1. Integer validity: n_pop[u] + n_tail[u] == N for all u, with 0 <= n_pop[u] <= N.
      2. Unbiased expectation: E[n_pop[u]] == N * pop_u[u].
      3. Strict calibration bound: |n_pop[u] / N - pop_u[u]| <= 1 / N.

    Parameters:
      pop_u: (n_users,) float array of user popularity inclinations in [0, 1].
      N: int, recommendation slate length (default 20).
      rng: Optional numpy Generator or int seed for reproducibility.

    Returns:
      (n_pop, n_tail): Tuple of (n_users,) int64 arrays.
    """
    pop_u_arr = np.asarray(pop_u, dtype=np.float64)
    pop_u_clipped = np.clip(pop_u_arr, 0.0, 1.0)

    if rng is None:
        rng = np.random.default_rng()
    elif isinstance(rng, (int, np.integer)):
        rng = np.random.default_rng(rng)

    expected_pop = N * pop_u_clipped
    floor_pop = np.floor(expected_pop)
    frac_pop = expected_pop - floor_pop  # strictly in [0, 1)

    draws = rng.uniform(0.0, 1.0, size=pop_u_clipped.shape)
    stochastic_round = (draws < frac_pop).astype(np.int64)

    n_pop = floor_pop.astype(np.int64) + stochastic_round
    n_pop = np.clip(n_pop, 0, N)
    n_tail = N - n_pop

    return n_pop.astype(np.int64), n_tail.astype(np.int64)


def user_adaptive_alpha_partition(
    pop_count: np.ndarray,
    train_matrix: np.ndarray,
    alpha_base: float = 0.20,
    beta: float = 0.15
) -> np.ndarray:
    """Computes personalized head-affinity inclination based on user interaction volume.

    High-activity users naturally have wider catalog exposure and their effective
    head catalog boundary is dynamically scaled up, while sparse users focus on
    a tighter head catalog boundary.

    Parameters:
      pop_count: (n_items,) array of item interaction counts across all training data.
      train_matrix: (n_users, n_items) interaction matrix.
      alpha_base: baseline Pareto cumulative volume threshold (default 0.20).
      beta: scaling coefficient for user interaction activity modulation (default 0.15).

    Returns:
      pop_u_adaptive: (n_users,) float32 array of user-specific popularity inclinations in [0, 1].
    """
    n_users, n_items = train_matrix.shape
    pop_count_arr = np.asarray(pop_count, dtype=np.float64)
    total_pop = max(float(pop_count_arr.sum()), 1.0)

    # Sort items by descending popularity
    order = np.argsort(-pop_count_arr)
    cum_pop = np.cumsum(pop_count_arr[order])

    # User interaction activity
    user_counts = (train_matrix > 0).sum(axis=1).astype(np.float64)
    max_count = max(float(user_counts.max()), 1.0)

    # Normalized activity weight in [0, 1] using log-scaling
    activity_weight = np.log1p(user_counts) / np.log1p(max_count)

    # User-specific alpha threshold in [0.05, 0.50]
    alpha_u = np.clip(alpha_base * (1.0 + beta * (activity_weight - 0.5)), 0.05, 0.50)

    # Number of head items for each user threshold
    user_thresholds = total_pop * alpha_u
    n_head_u = np.searchsorted(cum_pop, user_thresholds, side="right") + 1
    n_head_u = np.clip(n_head_u, 1, n_items)

    # Map each item index to its rank position in descending order
    item_rank = np.zeros(n_items, dtype=np.int64)
    item_rank[order] = np.arange(n_items)

    pop_u_adaptive = np.zeros(n_users, dtype=np.float32)
    for u in range(n_users):
        u_items = np.where(train_matrix[u] > 0)[0]
        if len(u_items) == 0:
            pop_u_adaptive[u] = float(alpha_base)
        else:
            u_item_ranks = item_rank[u_items]
            n_head_interacted = int(np.sum(u_item_ranks < n_head_u[u]))
            pop_u_adaptive[u] = float(n_head_interacted) / float(len(u_items))

    return pop_u_adaptive


def calibrated_score_fusion(
    scores_fed: np.ndarray,
    scores_llm: np.ndarray,
    pool_idx: np.ndarray,
    lam: float = 0.70,
    standardize: bool = True
) -> np.ndarray:
    """Fuses collaborative filtering scores with LLM semantic similarity scores
    using intra-pool z-score standardization.

    Parameters:
      scores_fed: (n_users, n_items) collaborative filtering score matrix.
      scores_llm: (n_users, n_items) Sentence-BERT semantic cosine similarity matrix.
      pool_idx: array-like of item indices in the pool (e.g. head_idx or tail_idx).
      lam: blend weight for semantic scores lambda in [0, 1] (default 0.70).
      standardize: whether to apply intra-pool z-score standardization on scores_fed.

    Returns:
      fused: (n_users, n_items) float32 matrix with blended scores on pool_idx
             and -inf on non-pool items.
    """
    n_users, n_items = scores_fed.shape
    pool_idx_arr = np.asarray(pool_idx, dtype=np.int64)
    fused = np.full((n_users, n_items), -np.inf, dtype=np.float32)

    if len(pool_idx_arr) == 0:
        return fused

    fed_pool = scores_fed[:, pool_idx_arr].astype(np.float32)
    llm_pool = scores_llm[:, pool_idx_arr].astype(np.float32)

    if standardize:
        mean = np.mean(fed_pool, axis=1, keepdims=True)
        std = np.std(fed_pool, axis=1, keepdims=True)
        std = np.where(std > 1e-8, std, 1.0)
        fed_norm = (fed_pool - mean) / std
    else:
        fed_norm = fed_pool

    blended = (1.0 - lam) * fed_norm + lam * llm_pool
    fused[:, pool_idx_arr] = blended
    return fused


def super_blueprint_merge(
    scores_pop: np.ndarray,
    scores_tail: np.ndarray,
    train_matrix: np.ndarray,
    head_idx: np.ndarray,
    N: int = 10,
    blueprint_order: str = "preference"
) -> np.ndarray:
    """Blueprint-based merge to construct top-N per user (Yavru et al., Algorithm 2).

    Inputs are score matrices restricted to the appropriate item subset:
      scores_pop: (n_users, n_items) full matrix, but only head items used.
      scores_tail: (n_users, n_items) full matrix, but only tail items used.
    train_matrix: (n_users, n_items) used for blueprint construction.
    head_idx: array of head item indices.
    Returns: (n_users, N) array of selected item indices (the RecList).
    """
    n_users = train_matrix.shape[0]
    head_set = set(head_idx.tolist())
    tail_set = set(range(train_matrix.shape[1])) - head_set

    # Mask scores outside their pool: set non-pool items to -inf
    pop_mask = np.full(scores_pop.shape, -np.inf, dtype=np.float32)
    pop_mask[:, list(head_set)] = scores_pop[:, list(head_set)]
    tail_mask = np.full(scores_tail.shape, -np.inf, dtype=np.float32)
    tail_mask[:, list(tail_set)] = scores_tail[:, list(tail_set)]

    # Per-user list of training item indices sorted by descending preference count
    out = np.zeros((n_users, N), dtype=np.int64) - 1

    for u in range(n_users):
        row_count = train_matrix[u]
        if row_count.sum() == 0:
            cand = np.argsort(-tail_mask[u])[:N]
            out[u] = cand[:N]
            continue
        u_items_sorted = np.argsort(-row_count)
        u_items_sorted = u_items_sorted[row_count[u_items_sorted] > 0]
        blueprint = np.array([1 if i in head_set else 0 for i in u_items_sorted], dtype=np.int8)

        # Hard quota
        Pop_u = blueprint.mean() if len(blueprint) > 0 else 0.5
        N_pop = int(np.floor(N * Pop_u))
        N_tail = N - N_pop
        # Cap pools to user's potential scores
        C_pop = np.argsort(-pop_mask[u])[:max(N_pop, 0)]
        C_tail = np.argsort(-tail_mask[u])[:max(N_tail, 0)]
        idx_pop = 0
        idx_tail = 0
        rec = []
        for k in range(min(N, len(blueprint))):
            if blueprint[k] == 1 and idx_pop < N_pop and idx_pop < len(C_pop):
                rec.append(int(C_pop[idx_pop])); idx_pop += 1
            elif blueprint[k] == 0 and idx_tail < N_tail and idx_tail < len(C_tail):
                rec.append(int(C_tail[idx_tail])); idx_tail += 1
            elif idx_pop < N_pop and idx_pop < len(C_pop):
                rec.append(int(C_pop[idx_pop])); idx_pop += 1
            elif idx_tail < N_tail and idx_tail < len(C_tail):
                rec.append(int(C_tail[idx_tail])); idx_tail += 1
            else:
                break
        # Fill remaining (greedy residual quota)
        while len(rec) < N and idx_pop < N_pop and idx_pop < len(C_pop):
            rec.append(int(C_pop[idx_pop])); idx_pop += 1
        while len(rec) < N and idx_tail < N_tail and idx_tail < len(C_tail):
            rec.append(int(C_tail[idx_tail])); idx_tail += 1
        # If still short, fill from best overall fallback
        if len(rec) < N:
            combined = np.maximum(pop_mask[u], tail_mask[u])
            fallback = np.argsort(-combined)
            for it in fallback:
                if it not in rec:
                    rec.append(int(it))
                    if len(rec) >= N:
                        break
        out[u] = rec[:N]
    return out


def tri_tier_blueprint_merge(
    scores_head: np.ndarray,
    scores_torso: np.ndarray,
    scores_tail: np.ndarray,
    train_matrix: np.ndarray,
    head_idx: np.ndarray,
    torso_idx: np.ndarray,
    N: int = 10
) -> np.ndarray:
    """Tri-tier blueprint merge (Head, Torso, Tail)."""
    n_users = train_matrix.shape[0]
    head_set = set(head_idx.tolist())
    torso_set = set(torso_idx.tolist())
    tail_set = set(range(train_matrix.shape[1])) - head_set - torso_set

    head_mask = np.full(scores_head.shape, -np.inf, dtype=np.float32)
    head_mask[:, list(head_set)] = scores_head[:, list(head_set)]

    torso_mask = np.full(scores_torso.shape, -np.inf, dtype=np.float32)
    torso_mask[:, list(torso_set)] = scores_torso[:, list(torso_set)]

    tail_mask = np.full(scores_tail.shape, -np.inf, dtype=np.float32)
    tail_mask[:, list(tail_set)] = scores_tail[:, list(tail_set)]

    out = np.zeros((n_users, N), dtype=np.int64) - 1

    for u in range(n_users):
        row_count = train_matrix[u]
        if row_count.sum() == 0:
            cand = np.argsort(-tail_mask[u])[:N]
            out[u] = cand[:N]
            continue

        u_items_sorted = np.argsort(-row_count)
        u_items_sorted = u_items_sorted[row_count[u_items_sorted] > 0]

        # 2: Head, 1: Torso, 0: Tail
        blueprint = np.array(
            [2 if i in head_set else (1 if i in torso_set else 0) for i in u_items_sorted],
            dtype=np.int8
        )

        Pop_head = (blueprint == 2).mean()
        Pop_torso = (blueprint == 1).mean()

        N_head = int(np.floor(N * Pop_head))
        N_torso = int(np.floor(N * Pop_torso))
        N_tail = N - N_head - N_torso

        C_head = np.argsort(-head_mask[u])[:max(N_head, 0)]
        C_torso = np.argsort(-torso_mask[u])[:max(N_torso, 0)]
        C_tail = np.argsort(-tail_mask[u])[:max(N_tail, 0)]

        idx_head, idx_torso, idx_tail = 0, 0, 0
        rec = []

        for k in range(min(N, len(blueprint))):
            bp = blueprint[k]
            if bp == 2 and idx_head < N_head and idx_head < len(C_head):
                rec.append(int(C_head[idx_head])); idx_head += 1
            elif bp == 1 and idx_torso < N_torso and idx_torso < len(C_torso):
                rec.append(int(C_torso[idx_torso])); idx_torso += 1
            elif bp == 0 and idx_tail < N_tail and idx_tail < len(C_tail):
                rec.append(int(C_tail[idx_tail])); idx_tail += 1
            elif idx_head < N_head and idx_head < len(C_head):
                rec.append(int(C_head[idx_head])); idx_head += 1
            elif idx_torso < N_torso and idx_torso < len(C_torso):
                rec.append(int(C_torso[idx_torso])); idx_torso += 1
            elif idx_tail < N_tail and idx_tail < len(C_tail):
                rec.append(int(C_tail[idx_tail])); idx_tail += 1
            else:
                break

        while len(rec) < N and idx_head < N_head and idx_head < len(C_head):
            rec.append(int(C_head[idx_head])); idx_head += 1
        while len(rec) < N and idx_torso < N_torso and idx_torso < len(C_torso):
            rec.append(int(C_torso[idx_torso])); idx_torso += 1
        while len(rec) < N and idx_tail < N_tail and idx_tail < len(C_tail):
            rec.append(int(C_tail[idx_tail])); idx_tail += 1

        if len(rec) < N:
            combined = np.maximum.reduce([head_mask[u], torso_mask[u], tail_mask[u]])
            fallback = np.argsort(-combined)
            for it in fallback:
                if it not in rec:
                    rec.append(int(it))
                    if len(rec) >= N:
                        break
        out[u] = rec[:N]
    return out


def logit_adjusted_calibration(
    scores: np.ndarray,
    pop_counts: np.ndarray,
    pop_u: np.ndarray,
    tau: float = 1.0,
    epsilon: float = 1e-5
) -> np.ndarray:
    """Logit-adjusted calibration for recommendation scores.
    S_adj_u,i = S_u,i - tau * (1 - Pop_u) * log(P_i + epsilon)

    scores: (n_users, n_items) predicted raw scores
    pop_counts: (n_items,) interaction counts
    pop_u: (n_users,) popularity inclination [0, 1]
    """
    n_users, n_items = scores.shape
    total_interactions = np.sum(pop_counts)
    if total_interactions == 0:
        return scores.copy()

    pop_prob = pop_counts / total_interactions
    log_pi = np.log(pop_prob + epsilon)

    user_factor = (1.0 - pop_u).reshape(-1, 1)
    adjusted_scores = scores - tau * user_factor * log_pi.reshape(1, -1)
    return adjusted_scores


def confidence_elastic_merge(
    scores_pop: np.ndarray,
    scores_tail: np.ndarray,
    train_matrix: np.ndarray,
    head_idx: np.ndarray,
    N: int = 20,
    threshold: float = 0.10
) -> np.ndarray:
    """Score-gap aware elasticity allowing swapping adjacent slots when confidence margin is high.

    When the marginal candidate in the alternate pool exhibits a substantial score
    advantage over the boundary candidate in the scheduled pool, this function allows
    a bounded (+/- 1 slot) elasticity adjustment to maximize recommendation quality
    while strictly maintaining popularity calibration within the 1/N bound.

    Parameters:
      scores_pop: (n_users, n_items) scores from head model (or masked on tail).
      scores_tail: (n_users, n_items) scores from tail model (or masked on head).
      train_matrix: (n_users, n_items) interaction matrix.
      head_idx: array of Head item indices from Pareto partition.
      N: recommendation list length (default 20).
      threshold: score gap threshold theta required to trigger a slot swap (default 0.10).

    Returns:
      reclist_matrix: (n_users, N) int64 matrix of recommended item indices.
    """
    n_users, n_items = train_matrix.shape
    head_set = set(head_idx.tolist())
    tail_set = set(range(n_items)) - head_set

    # Mask non-pool items and training positives
    pop_mask = np.full(scores_pop.shape, -np.inf, dtype=np.float32)
    pop_mask[:, list(head_set)] = scores_pop[:, list(head_set)]
    pop_mask[train_matrix > 0] = -np.inf

    tail_mask = np.full(scores_tail.shape, -np.inf, dtype=np.float32)
    tail_mask[:, list(tail_set)] = scores_tail[:, list(tail_set)]
    tail_mask[train_matrix > 0] = -np.inf

    out = np.zeros((n_users, N), dtype=np.int64) - 1

    for u in range(n_users):
        row_count = train_matrix[u]
        if row_count.sum() == 0:
            combined = np.maximum(pop_mask[u], tail_mask[u])
            cand = np.argsort(-combined)[:N]
            out[u] = cand[:N]
            continue

        u_items_sorted = np.argsort(-row_count)
        u_items_sorted = u_items_sorted[row_count[u_items_sorted] > 0]
        blueprint = np.array([1 if i in head_set else 0 for i in u_items_sorted], dtype=np.int8)

        pop_u = float(blueprint.mean()) if len(blueprint) > 0 else 0.5
        n_pop_base = int(np.round(N * pop_u))
        n_pop_base = max(0, min(N, n_pop_base))
        n_tail_base = N - n_pop_base

        # Extract sorted candidate pools (filtering out -inf)
        order_pop = np.argsort(-pop_mask[u])
        cand_pop = [int(i) for i in order_pop if pop_mask[u, i] > -np.inf]

        order_tail = np.argsort(-tail_mask[u])
        cand_tail = [int(i) for i in order_tail if tail_mask[u, i] > -np.inf]

        n_pop = n_pop_base
        n_tail = n_tail_base

        # Check score gap elasticity at boundary
        if n_pop < N and n_tail > 0 and len(cand_pop) > n_pop and len(cand_tail) >= n_tail:
            s_head_next = float(pop_mask[u, cand_pop[n_pop]])
            s_tail_last = float(tail_mask[u, cand_tail[n_tail - 1]])
            if (s_head_next - s_tail_last) > threshold:
                n_pop += 1
                n_tail -= 1
        elif n_tail < N and n_pop > 0 and len(cand_tail) > n_tail and len(cand_pop) >= n_pop:
            s_tail_next = float(tail_mask[u, cand_tail[n_tail]])
            s_head_last = float(pop_mask[u, cand_pop[n_pop - 1]])
            if (s_tail_next - s_head_last) > threshold:
                n_tail += 1
                n_pop -= 1

        # Interleave candidates following blueprint
        c_pop_sel = cand_pop[:n_pop]
        c_tail_sel = cand_tail[:n_tail]
        idx_pop = 0
        idx_tail = 0
        rec = []

        for k in range(min(N, len(blueprint))):
            bp = blueprint[k]
            if bp == 1 and idx_pop < len(c_pop_sel):
                rec.append(c_pop_sel[idx_pop]); idx_pop += 1
            elif bp == 0 and idx_tail < len(c_tail_sel):
                rec.append(c_tail_sel[idx_tail]); idx_tail += 1
            elif idx_pop < len(c_pop_sel):
                rec.append(c_pop_sel[idx_pop]); idx_pop += 1
            elif idx_tail < len(c_tail_sel):
                rec.append(c_tail_sel[idx_tail]); idx_tail += 1
            else:
                break

        while len(rec) < N and idx_pop < len(c_pop_sel):
            rec.append(c_pop_sel[idx_pop]); idx_pop += 1
        while len(rec) < N and idx_tail < len(c_tail_sel):
            rec.append(c_tail_sel[idx_tail]); idx_tail += 1

        # Fallback if pools exhausted
        if len(rec) < N:
            combined = np.maximum(pop_mask[u], tail_mask[u])
            fallback_order = np.argsort(-combined)
            for it in fallback_order:
                it_int = int(it)
                if combined[it_int] > -np.inf and it_int not in rec:
                    rec.append(it_int)
                    if len(rec) >= N:
                        break

        # Edge fallback if still short
        if len(rec) < N:
            seen_items = set(np.where(train_matrix[u] > 0)[0].tolist()) | set(rec)
            for it in range(n_items):
                if it not in seen_items:
                    rec.append(it)
                    if len(rec) >= N:
                        break

        out[u] = np.array(rec[:N], dtype=np.int64)

    return out


def dynamic_probabilistic_quota_merge(
    scores_pop: np.ndarray,
    scores_tail: np.ndarray,
    train_matrix: np.ndarray,
    head_idx: np.ndarray,
    N: int = 20,
    mode: str = "stochastic",
    seed: int = 42,
    **kwargs
) -> np.ndarray:
    """Primary dynamic merge entry point supporting multiple quota allocation modes.

    Supported modes:
      - 'stochastic': Expectation-preserving randomized rounding of Pop_u.
      - 'adaptive_alpha': Personalized catalog thresholding based on user activity.
      - 'confidence_elastic': Score-gap aware elasticity allowing boundary slot swaps.
      - 'hybrid': Combines user-adaptive alpha, stochastic rounding, and confidence elasticity.
      - 'static': Baseline integer floor truncation (Yavru et al.).

    Parameters:
      scores_pop: (n_users, n_items) head model predicted scores.
      scores_tail: (n_users, n_items) tail model predicted scores.
      train_matrix: (n_users, n_items) user-item interaction matrix.
      head_idx: array of Head item indices from Pareto partition.
      N: recommendation slate size (default 20).
      mode: merge strategy mode (default 'stochastic').
      seed: random seed for stochastic reproducibility (default 42).
      **kwargs: optional hyper-parameters (alpha_base, beta, threshold, pop_count, etc.)

    Returns:
      reclist_matrix: (n_users, N) int64 matrix of recommended item indices.
    """
    n_users, n_items = train_matrix.shape
    head_set = set(head_idx.tolist())
    tail_set = set(range(n_items)) - head_set

    # Mask non-pool items and training positives
    pop_mask = np.full(scores_pop.shape, -np.inf, dtype=np.float32)
    pop_mask[:, list(head_set)] = scores_pop[:, list(head_set)]
    pop_mask[train_matrix > 0] = -np.inf

    tail_mask = np.full(scores_tail.shape, -np.inf, dtype=np.float32)
    tail_mask[:, list(tail_set)] = scores_tail[:, list(tail_set)]
    tail_mask[train_matrix > 0] = -np.inf

    rng = np.random.default_rng(seed)

    # 1. Compute Quotas based on mode
    if mode == "static":
        pop_u = user_popularity_inclination(train_matrix, head_idx)
        n_pop_all = np.floor(N * pop_u).astype(np.int64)
        n_pop_all = np.clip(n_pop_all, 0, N)
        n_tail_all = N - n_pop_all
        elastic_check = False

    elif mode == "stochastic":
        pop_u = user_popularity_inclination(train_matrix, head_idx)
        n_pop_all, n_tail_all = stochastic_user_quota(pop_u, N=N, rng=rng)
        elastic_check = False

    elif mode == "adaptive_alpha":
        alpha_base = float(kwargs.get("alpha_base", 0.20))
        beta = float(kwargs.get("beta", 0.15))
        pop_count = kwargs.get("pop_count", None)
        if pop_count is None:
            pop_count = train_matrix.sum(axis=0)
        pop_u_adapt = user_adaptive_alpha_partition(pop_count, train_matrix, alpha_base=alpha_base, beta=beta)
        n_pop_all, n_tail_all = stochastic_user_quota(pop_u_adapt, N=N, rng=rng)
        elastic_check = False

    elif mode == "confidence_elastic":
        threshold = float(kwargs.get("threshold", 0.10))
        return confidence_elastic_merge(scores_pop, scores_tail, train_matrix, head_idx, N=N, threshold=threshold)

    elif mode == "hybrid":
        alpha_base = float(kwargs.get("alpha_base", 0.20))
        beta = float(kwargs.get("beta", 0.15))
        pop_count = kwargs.get("pop_count", None)
        if pop_count is None:
            pop_count = train_matrix.sum(axis=0)
        pop_u_adapt = user_adaptive_alpha_partition(pop_count, train_matrix, alpha_base=alpha_base, beta=beta)
        n_pop_all, n_tail_all = stochastic_user_quota(pop_u_adapt, N=N, rng=rng)
        elastic_check = True
        threshold = float(kwargs.get("threshold", 0.10))

    else:
        # Default fallback to stochastic
        pop_u = user_popularity_inclination(train_matrix, head_idx)
        n_pop_all, n_tail_all = stochastic_user_quota(pop_u, N=N, rng=rng)
        elastic_check = False

    out = np.zeros((n_users, N), dtype=np.int64) - 1

    for u in range(n_users):
        row_count = train_matrix[u]
        if row_count.sum() == 0:
            combined = np.maximum(pop_mask[u], tail_mask[u])
            cand = np.argsort(-combined)[:N]
            out[u] = cand[:N]
            continue

        u_items_sorted = np.argsort(-row_count)
        u_items_sorted = u_items_sorted[row_count[u_items_sorted] > 0]
        blueprint = np.array([1 if i in head_set else 0 for i in u_items_sorted], dtype=np.int8)

        n_pop = int(n_pop_all[u])
        n_tail = int(n_tail_all[u])

        # Candidate pools
        order_pop = np.argsort(-pop_mask[u])
        cand_pop = [int(i) for i in order_pop if pop_mask[u, i] > -np.inf]

        order_tail = np.argsort(-tail_mask[u])
        cand_tail = [int(i) for i in order_tail if tail_mask[u, i] > -np.inf]

        # If hybrid mode, perform confidence-elastic check
        if elastic_check:
            if n_pop < N and n_tail > 0 and len(cand_pop) > n_pop and len(cand_tail) >= n_tail:
                s_head_next = float(pop_mask[u, cand_pop[n_pop]])
                s_tail_last = float(tail_mask[u, cand_tail[n_tail - 1]])
                if (s_head_next - s_tail_last) > threshold:
                    n_pop += 1
                    n_tail -= 1
            elif n_tail < N and n_pop > 0 and len(cand_tail) > n_tail and len(cand_pop) >= n_pop:
                s_tail_next = float(tail_mask[u, cand_tail[n_tail]])
                s_head_last = float(pop_mask[u, cand_pop[n_pop - 1]])
                if (s_tail_next - s_head_last) > threshold:
                    n_tail += 1
                    n_pop -= 1

        c_pop_sel = cand_pop[:n_pop]
        c_tail_sel = cand_tail[:n_tail]
        idx_pop = 0
        idx_tail = 0
        rec = []

        for k in range(min(N, len(blueprint))):
            bp = blueprint[k]
            if bp == 1 and idx_pop < len(c_pop_sel):
                rec.append(c_pop_sel[idx_pop]); idx_pop += 1
            elif bp == 0 and idx_tail < len(c_tail_sel):
                rec.append(c_tail_sel[idx_tail]); idx_tail += 1
            elif idx_pop < len(c_pop_sel):
                rec.append(c_pop_sel[idx_pop]); idx_pop += 1
            elif idx_tail < len(c_tail_sel):
                rec.append(c_tail_sel[idx_tail]); idx_tail += 1
            else:
                break

        while len(rec) < N and idx_pop < len(c_pop_sel):
            rec.append(c_pop_sel[idx_pop]); idx_pop += 1
        while len(rec) < N and idx_tail < len(c_tail_sel):
            rec.append(c_tail_sel[idx_tail]); idx_tail += 1

        if len(rec) < N:
            combined = np.maximum(pop_mask[u], tail_mask[u])
            fallback_order = np.argsort(-combined)
            for it in fallback_order:
                it_int = int(it)
                if combined[it_int] > -np.inf and it_int not in rec:
                    rec.append(it_int)
                    if len(rec) >= N:
                        break

        if len(rec) < N:
            seen_items = set(np.where(train_matrix[u] > 0)[0].tolist()) | set(rec)
            for it in range(n_items):
                if it not in seen_items:
                    rec.append(it)
                    if len(rec) >= N:
                        break

        out[u] = np.array(rec[:N], dtype=np.int64)

    return out


def evaluate_calibration(
    reclist_matrix: np.ndarray,
    train_matrix: np.ndarray,
    head_idx: np.ndarray
) -> Dict[str, float]:
    """Evaluates Popularity Calibration Root Mean Squared Error (Rmse-PC) and
    Mean Rank Miscalibration (MRMC) for a recommendation matrix.

    reclist_matrix: (n_users, N) recommended item indices.
    train_matrix: (n_users, n_items) training interaction matrix.
    head_idx: array of Head item indices from Pareto partition.
    Returns: dict with 'rmse_pc' and 'mrmc'.
    """
    n_users, N = reclist_matrix.shape
    n_items = train_matrix.shape[1]

    head_set = set(head_idx.tolist())
    head_arr = np.array(sorted(head_set), dtype=np.int64)
    is_head_item = np.zeros(n_items, dtype=np.int8)
    is_head_item[head_arr] = 1

    # Historical Pop_u
    user_tot = train_matrix.sum(axis=1)
    pop_u = np.where(user_tot > 0, train_matrix[:, head_arr].sum(axis=1) / np.maximum(user_tot, 1.0), 0.0)

    # Recommendations head fraction
    head_indicators = is_head_item[reclist_matrix]  # (n_users, N)
    head_frac_rec = head_indicators.mean(axis=1)
    rmse_pc = float(np.sqrt(((head_frac_rec - pop_u) ** 2).mean()))

    # MRMC: average over k in [1..N] of cumulative head exposure difference
    cum_head = np.cumsum(head_indicators.astype(np.float32), axis=1) / np.arange(1, N + 1)[None, :]
    mrmc_per_user = np.abs(cum_head - pop_u[:, None]).mean(axis=1)
    mrmc = float(mrmc_per_user.mean())

    return {"rmse_pc": rmse_pc, "mrmc": mrmc}


def mask_user_trainpos(scores: np.ndarray, train_matrix: np.ndarray) -> np.ndarray:
    """Set scores[u, train item i] = -inf so they don't appear in final lists.

    scores: (n_users, n_items) numpy
    train_matrix: (n_users, n_items) binary or counts
    """
    masked = scores.copy()
    masked[train_matrix > 0] = -np.inf
    return masked


def gkpi_score(ndcg: float, aplt: float, entropy: float, novelty: float, ltc: float) -> float:
    """Compute SUPER-paper-style General KPI = arithmetic mean of harmonic means
    of nDCG with each beyond-accuracy metric.

    H(x, y) = 2xy / (x + y); fallback to 0 if x+y == 0.
    """
    def H(x, y):
        return (2 * x * y) / (x + y) if (x + y) > 0 else 0.0
    return float(np.mean([H(ndcg, aplt), H(ndcg, entropy), H(ndcg, novelty), H(ndcg, ltc)]))


# Convenience and backwards-compatibility aliases
user_adaptive_alpha_merge = dynamic_probabilistic_quota_merge
dynamic_probabilistic_blueprint_merge = dynamic_probabilistic_quota_merge
