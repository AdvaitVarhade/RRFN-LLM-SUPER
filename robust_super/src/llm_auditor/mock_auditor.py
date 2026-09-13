import numpy as np
from typing import Dict, List, Tuple, Optional

class MockLLMAuditor:
    """
    Deterministic offline mock for the LLM semantic auditor.
    Produces high reliability for genuine interactions and low reliability for injected noise.
    """
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def audit_interaction(
        self,
        user_id: int,
        item_id: int,
        rating: int,
        ground_truth_label: int = 1, # 1 genuine, 0 injected
        user_mean_rating: float = 3.5
    ) -> Tuple[float, str]:
        """
        Simulates LLM evaluation with realistic variance.
        """
        if ground_truth_label == 1:
            # Genuine: high score centered around 0.88
            rating_diff = abs(rating - user_mean_rating)
            base_score = 0.92 - (rating_diff * 0.05)
            noise = self.rng.normal(0, 0.04)
            score = float(np.clip(base_score + noise, 0.45, 0.99))
            reason = "Interaction aligns with user genre and rating preferences."
        else:
            # Injected / corrupted: low score centered around 0.20
            base_score = 0.22
            noise = self.rng.normal(0, 0.05)
            score = float(np.clip(base_score + noise, 0.05, 0.48))
            reason = "Rating contradicts historical genre affinity and shows anomalous pattern."

        return score, reason

    def audit_batch(
        self,
        interactions: List[Tuple[int, int, int]], # (u, i, r)
        ground_truth_labels: Dict[Tuple[int, int], int],
        user_mean_ratings: Dict[int, float]
    ) -> Dict[Tuple[int, int], float]:
        scores = {}
        for u, i, r in interactions:
            gt = ground_truth_labels.get((u, i), 1)
            mean_r = user_mean_ratings.get(u, 3.5)
            score, _ = self.audit_interaction(u, i, r, gt, mean_r)
            scores[(u, i)] = score
        return scores
