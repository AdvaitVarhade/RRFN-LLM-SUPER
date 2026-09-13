import numpy as np
from typing import Dict, List, Tuple, Optional

def compute_semantic_similarity(
    train_dict: Dict[int, List[Tuple[int, int, int]]],
    review_texts: Optional[Dict[Tuple[int, int], str]] = None
) -> Dict[Tuple[int, int], float]:
    """
    Computes pairwise semantic similarity among review texts within an item's rating burst.
    If review texts are not present (e.g. standard MovieLens-1M), returns 0.0.
    """
    sim_scores: Dict[Tuple[int, int], float] = {}

    if not review_texts:
        for u, interactions in train_dict.items():
            for item, _, _ in interactions:
                sim_scores[(u, item)] = 0.0
        return sim_scores

    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        model = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        for u, interactions in train_dict.items():
            for item, _, _ in interactions:
                sim_scores[(u, item)] = 0.0
        return sim_scores

    # Group reviews by item
    item_reviews: Dict[int, List[Tuple[int, str]]] = {}
    for (u, item), text in review_texts.items():
        if text and len(text.strip()) > 0:
            item_reviews.setdefault(item, []).append((u, text))

    for item, u_texts in item_reviews.items():
        if len(u_texts) < 2:
            for u, _ in u_texts:
                sim_scores[(u, item)] = 0.0
            continue

        texts = [t for _, t in u_texts]
        embeddings = model.encode(texts)
        sim_mat = cosine_similarity(embeddings)
        upper_idx = np.triu_indices(len(texts), k=1)
        mean_sim = float(sim_mat[upper_idx].mean()) if len(upper_idx[0]) > 0 else 0.0

        for u, _ in u_texts:
            sim_scores[(u, item)] = max(0.0, min(1.0, mean_sim))

    # Fill default for any uncalculated pairs
    for u, interactions in train_dict.items():
        for item, _, _ in interactions:
            if (u, item) not in sim_scores:
                sim_scores[(u, item)] = 0.0

    return sim_scores
