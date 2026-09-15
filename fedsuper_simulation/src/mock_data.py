"""
fedsuper_simulation/src/mock_data.py - Native Standalone Synthetic Dataset Generator
Generates realistic client cohorts, item catalog, power-law interactions, Dirichlet genre preferences,
L2-normalized LLM semantic profiles/embeddings, and Head/Torso/Tail partitions entirely offline.
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import pandas as pd

try:
    from .config import SimulationConfig
except (ImportError, ValueError):
    try:
        from src.config import SimulationConfig
    except ImportError:
        from config import SimulationConfig


GENRE_NAMES = [
    "Action & Adventure",
    "Sci-Fi & Fantasy",
    "Drama & Romance",
    "Comedy & Satire",
    "Documentary & History",
    "Animation & Anime",
    "Thriller & Mystery",
    "Indie & Art House"
]


@dataclass
class SyntheticDataset:
    """Container for synthetic federated recommendation data.

    Attributes:
        num_users: Number of users/clients (U).
        num_items: Number of items in catalog (M).
        user_metadata: DataFrame with columns ['user_id', 'preferred_genres', 'activity_level', 'pop_u_true', 'pop_u_torso', 'pop_u_tail'].
        item_metadata: DataFrame with columns ['item_id', 'title', 'category', 'popularity_count', 'popularity_tier', 'is_head', 'is_torso', 'is_tail', 'popularity_rank'].
        train_matrix: (U, M) float32 binary matrix of training interactions.
        test_dict: Dict mapping user_id -> [held_out_test_item_id] (leave-one-out).
        user_llm_profiles: (U, llm_dim) float32 L2-normalized semantic user profile vectors s_u.
        item_llm_embeddings: (M, llm_dim) float32 L2-normalized semantic item embedding vectors v_i.
        head_idx: 1D int64 array of Head item indices (top 20% most popular).
        torso_idx: 1D int64 array of Torso item indices (next 30% most popular).
        tail_idx: 1D int64 array of Tail item indices (bottom 50% least popular).
    """
    num_users: int
    num_items: int
    user_metadata: pd.DataFrame
    item_metadata: pd.DataFrame
    train_matrix: np.ndarray
    test_dict: Dict[int, List[int]]
    user_llm_profiles: np.ndarray
    item_llm_embeddings: np.ndarray
    head_idx: np.ndarray
    torso_idx: np.ndarray
    tail_idx: np.ndarray


def _normalize_l2(matrix: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Normalizes row vectors to unit L2 norm."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return (matrix / norms).astype(np.float32)


def generate_mock_dataset(config: SimulationConfig, seed: Optional[int] = None) -> SyntheticDataset:
    """Generates a complete standalone synthetic dataset matching simulation specifications.

    Parameters:
        config: SimulationConfig instance providing dimensions, parameters, and defaults.
        seed: Optional explicit random seed (overrides config.seed if provided).

    Returns:
        SyntheticDataset instance containing all matrices, metadata, and partition indices.
    """
    actual_seed = seed if seed is not None else config.seed
    rng = np.random.default_rng(actual_seed)

    num_users = config.num_clients
    num_items = config.num_items
    llm_dim = config.llm_dim
    num_genres = min(config.num_genres, len(GENRE_NAMES))
    genres = GENRE_NAMES[:num_genres]

    # 1. Genre Centroids in LLM semantic space: C in R^(num_genres x llm_dim)
    raw_centroids = rng.normal(0.0, 1.0, size=(num_genres, llm_dim))
    genre_centroids = _normalize_l2(raw_centroids)

    # 2. Item Catalog Generation
    # Assign each item a primary genre
    item_genres = rng.choice(num_genres, size=num_items)
    
    # Generate item semantic embeddings: v_i = normalize(C_{genre} + noise)
    item_noise = rng.normal(0.0, 0.15, size=(num_items, llm_dim))
    raw_item_embs = genre_centroids[item_genres] + item_noise
    item_llm_embeddings = _normalize_l2(raw_item_embs)

    # Inherent Zipf popularity weights
    zipf_exp = config.zipf_exponent
    item_ranks = np.arange(1, num_items + 1)
    base_pop_weights = 1.0 / (item_ranks ** zipf_exp)
    # Permute base weights slightly to mix across genres
    base_pop_weights = rng.permutation(base_pop_weights)
    base_pop_weights /= base_pop_weights.sum()

    # Preliminary Head mask for interaction sampling (top 20% items by base weight)
    prelim_head_count = max(1, int(round(0.20 * num_items)))
    prelim_head_items = np.argsort(-base_pop_weights)[:prelim_head_count]
    is_prelim_head = np.zeros(num_items, dtype=bool)
    is_prelim_head[prelim_head_items] = True

    # 3. Client Cohort Generation
    # Dirichlet genre preferences (alpha=0.5 yields focus on 1-2 favorite genres)
    dirichlet_alpha = np.full(num_genres, 0.5)
    user_genre_probs = rng.dirichlet(dirichlet_alpha, size=num_users)  # (U, num_genres)

    # User semantic profiles: s_u = normalize(sum_k theta_{u,k} C_k + noise)
    user_semantic_centers = user_genre_probs @ genre_centroids  # (U, llm_dim)
    user_noise = rng.normal(0.0, 0.10, size=(num_users, llm_dim))
    user_llm_profiles = _normalize_l2(user_semantic_centers + user_noise)

    # User activity levels (number of interactions)
    # Log-normal distribution with min 8 (or min possible for small catalogs), max min(num_items - 2, 80)
    mu_activity = 3.2
    sigma_activity = 0.5
    raw_activities = rng.lognormal(mean=mu_activity, sigma=sigma_activity, size=num_users)
    min_act = min(8, max(3, num_items - 1)) if num_items > 3 else max(2, num_items)
    max_act = max(min_act, min(80, num_items - 1 if num_items > 2 else num_items))
    user_activities = np.clip(np.round(raw_activities).astype(int), min_act, max_act)

    # Target user popularity inclination: Pop_u* ~ Beta(2, 3) (diverse head vs tail lovers)
    user_target_pop = rng.beta(a=2.0, b=3.0, size=num_users)

    # 4. User Interactions Sampling
    user_interaction_lists: List[List[int]] = []
    
    for u in range(num_users):
        n_act = int(user_activities[u])
        n_act = min(n_act, num_items)
        theta_u = user_genre_probs[u]
        s_u = user_llm_profiles[u]
        target_head_u = user_target_pop[u]

        # Calculate item affinities for user u
        # A. Genre affinity
        genre_aff = theta_u[item_genres] + 0.05
        # B. LLM semantic cosine similarity in [0, 1]
        sem_sim = np.clip((item_llm_embeddings @ s_u + 1.0) / 2.0, 0.01, 1.0)
        # C. Popularity inclination affinity
        pop_pref = np.where(is_prelim_head, target_head_u, 1.0 - target_head_u) + 0.05
        # D. Power-law popularity component
        pop_aff = base_pop_weights ** 0.5

        # Combined interaction probability
        p_ui = genre_aff * sem_sim * pop_pref * pop_aff
        sum_p = p_ui.sum()
        if sum_p > 0:
            p_ui /= sum_p
        else:
            p_ui = np.full(num_items, 1.0 / num_items)

        # Sample n_act unique items
        sampled_items = rng.choice(num_items, size=n_act, replace=False, p=p_ui)
        user_interaction_lists.append(sampled_items.tolist())

    # 5. Catalog Popularity Stratification
    # Count total interactions per item
    interaction_counts = np.zeros(num_items, dtype=int)
    for items in user_interaction_lists:
        for it in items:
            interaction_counts[it] += 1

    # Deterministic sorting (descending count, ascending item_id)
    sorted_order = np.lexsort((np.arange(num_items), -interaction_counts))

    # Strict catalog partitioning: Head (20%), Torso (30%), Tail (50%)
    n_head = max(1, int(round(0.20 * num_items)))
    n_torso = max(1, int(round(0.30 * num_items)))
    if n_head + n_torso >= num_items:
        n_head = max(1, num_items // 3)
        n_torso = max(1, num_items // 3)
        if n_head + n_torso >= num_items:
            n_head = max(1, num_items - 2) if num_items >= 3 else 1
            n_torso = 1 if num_items >= 2 else 0
    n_tail = num_items - n_head - n_torso
    if n_tail < 0:
        n_tail = 0

    head_idx = sorted_order[:n_head].astype(np.int64)
    torso_idx = sorted_order[n_head : n_head + n_torso].astype(np.int64)
    tail_idx = sorted_order[n_head + n_torso :].astype(np.int64)

    head_set = set(head_idx.tolist())
    torso_set = set(torso_idx.tolist())
    tail_set = set(tail_idx.tolist())

    # 6. Build Train Matrix & Leave-One-Out Test Dict
    train_matrix = np.zeros((num_users, num_items), dtype=np.float32)
    test_dict: Dict[int, List[int]] = {}

    pop_u_true = np.zeros(num_users, dtype=np.float32)
    pop_u_torso = np.zeros(num_users, dtype=np.float32)
    pop_u_tail = np.zeros(num_users, dtype=np.float32)

    for u in range(num_users):
        u_items = user_interaction_lists[u]
        if len(u_items) > 1:
            # Leave-one-out: last item is test, remainder are train
            test_item = u_items[-1]
            train_items = u_items[:-1]
            test_dict[u] = [test_item]
            train_matrix[u, train_items] = 1.0
        elif len(u_items) == 1:
            test_dict[u] = [u_items[0]]
            train_matrix[u, u_items[0]] = 1.0
        else:
            test_dict[u] = []

        # Calculate exact historical inclination over all user interactions
        total_u = max(1, len(u_items))
        n_h = sum(1 for it in u_items if it in head_set)
        n_t = sum(1 for it in u_items if it in torso_set)
        n_l = sum(1 for it in u_items if it in tail_set)

        pop_u_true[u] = n_h / total_u
        pop_u_torso[u] = n_t / total_u
        pop_u_tail[u] = n_l / total_u

    # 7. Construct User Metadata DataFrame
    preferred_genres_list = []
    for u in range(num_users):
        top_genre_indices = np.argsort(-user_genre_probs[u])[:2]
        top_genre_str = ", ".join([genres[g] for g in top_genre_indices])
        preferred_genres_list.append(top_genre_str)

    user_metadata = pd.DataFrame({
        "user_id": np.arange(num_users, dtype=int),
        "preferred_genres": preferred_genres_list,
        "activity_level": user_activities,
        "pop_u_true": np.round(pop_u_true, 4),
        "pop_u_torso": np.round(pop_u_torso, 4),
        "pop_u_tail": np.round(pop_u_tail, 4),
    })

    # 8. Construct Item Metadata DataFrame
    popularity_ranks = np.zeros(num_items, dtype=int)
    popularity_ranks[sorted_order] = np.arange(1, num_items + 1)

    tier_labels = []
    for i in range(num_items):
        if i in head_set:
            tier_labels.append("Head")
        elif i in torso_set:
            tier_labels.append("Torso")
        else:
            tier_labels.append("Tail")

    item_titles = [f"Item #{i:03d} ({genres[item_genres[i]]})" for i in range(num_items)]

    item_metadata = pd.DataFrame({
        "item_id": np.arange(num_items, dtype=int),
        "title": item_titles,
        "category": [genres[g] for g in item_genres],
        "popularity_count": interaction_counts,
        "popularity_tier": tier_labels,
        "is_head": [i in head_set for i in range(num_items)],
        "is_torso": [i in torso_set for i in range(num_items)],
        "is_tail": [i in tail_set for i in range(num_items)],
        "popularity_rank": popularity_ranks,
    })

    return SyntheticDataset(
        num_users=num_users,
        num_items=num_items,
        user_metadata=user_metadata,
        item_metadata=item_metadata,
        train_matrix=train_matrix,
        test_dict=test_dict,
        user_llm_profiles=user_llm_profiles,
        item_llm_embeddings=item_llm_embeddings,
        head_idx=head_idx,
        torso_idx=torso_idx,
        tail_idx=tail_idx,
    )
