import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple

class NoiseTransitionMatrix(nn.Module):
    """
    Manages the noise transition matrix T, initialized from anchor points:
    T_kj = P(Y_tilde = j | Y* = k)
    and refined via a learnable parameter delta_T during risk-consistent training.
    """
    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.num_classes = num_classes
        
        # Initial estimated transition matrix T_hat (frozen buffer)
        self.register_buffer("T_hat", torch.eye(num_classes, dtype=torch.float32))
        
        # Learnable slack matrix delta_T
        self.delta_T = nn.Parameter(torch.zeros(num_classes, num_classes, dtype=torch.float32))

    def estimate_from_anchors(self, anchor_sets: Dict[int, List[Tuple[int, int, int, np.ndarray]]]):
        """
        Estimates T_hat empirically from anchor points:
        T_hat[k, j] = fraction of anchors with predicted true class k having observed rating j.
        """
        T_est = np.zeros((self.num_classes, self.num_classes), dtype=np.float32)

        for k in range(self.num_classes):
            anchors = anchor_sets.get(k, [])
            if len(anchors) == 0:
                # Default to identity if no anchors
                T_est[k, k] = 1.0
                continue

            for _, _, observed_rating, _ in anchors:
                # observed_rating is 1-indexed: 1..5 -> 0..4
                j = int(observed_rating) - 1
                if 0 <= j < self.num_classes:
                    T_est[k, j] += 1.0
                else:
                    T_est[k, k] += 1.0

            row_sum = T_est[k, :].sum()
            if row_sum > 0:
                T_est[k, :] /= row_sum
            else:
                T_est[k, k] = 1.0

        # Clip and re-normalize rows
        T_est = np.clip(T_est, 1e-4, 1.0)
        T_est /= T_est.sum(axis=-1, keepdims=True)

        self.T_hat.copy_(torch.from_numpy(T_est))

    def get_T_final(self) -> torch.Tensor:
        """
        Returns normalized transition matrix: RowSoftmax(T_hat + delta_T)
        """
        # Ensure row-stochastic matrix
        T_combined = self.T_hat + self.delta_T
        T_final = F.softmax(T_combined, dim=-1)
        return T_final

    def forward(self, clean_probs: torch.Tensor) -> torch.Tensor:
        """
        Computes predicted noisy probability distribution:
        P(Y_tilde | x) = clean_probs @ T_final
        clean_probs: [B, num_classes]
        Returns: [B, num_classes]
        """
        T_final = self.get_T_final()
        noisy_probs = torch.matmul(clean_probs, T_final)
        return noisy_probs
