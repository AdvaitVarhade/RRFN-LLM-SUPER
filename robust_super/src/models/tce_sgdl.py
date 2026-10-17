"""
robust_super/src/models/tce_sgdl.py
-----------------------------------
T-CE / SGDL: Transition-Matrix Cross-Entropy & Self-Guided Denoising Learning Baseline
References:
  - Patrini et al., "Making Deep Neural Networks Robust to Label Noise: A Loss Correction Approach", CVPR 2017 / NeurIPS 2022.
  - Gao et al., "Self-Guided Learning to Denoise for Robust Recommendation", SIGIR 2022.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np

from .lightgcn import LightGCN
from ..rrfn.transition_matrix import NoiseTransitionMatrix


class TCE_SGDL(LightGCN):
    """
    Monolithic robust collaborative filtering baseline combining:
    1. LightGCN 3-layer bipartite graph convolution backbone.
    2. Forward noise transition matrix T in R^{K x K} with anchor estimation and learnable slack delta_T.
    3. Forward risk-consistent loss correction: p_tilde = p * T.
    4. Self-guided dynamic instance re-weighting / loss truncation: w_ui = exp(-l_ui / tau_loss).
    5. Tikhonov regularized pseudo-inversion for numerical stability.
    6. Monolithic candidate scoring via score_candidates_batch.
    """
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        num_layers: int = 3,
        num_classes: int = 5,
        tau_loss: float = 1.0,
        loss_trunc: Optional[float] = None,
        delta_T_reg: float = 0.01,
        min_weight: float = 0.01
    ):
        super().__init__(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            num_classes=num_classes
        )
        self.tau_loss = float(tau_loss)
        self.loss_trunc = float(loss_trunc) if loss_trunc is not None else None
        self.delta_T_reg = float(delta_T_reg)
        self.min_weight = float(min_weight)

        # Internal Noise Transition Matrix module
        self.transition = NoiseTransitionMatrix(num_classes=num_classes)

    def estimate_transition_from_anchors(
        self,
        anchor_sets: Dict[int, List[Tuple[int, int, int, np.ndarray]]]
    ):
        """
        Estimates the base transition matrix T_hat from confident anchor interactions.
        """
        self.transition.estimate_from_anchors(anchor_sets)

    def get_transition_matrix(self) -> torch.Tensor:
        """
        Returns the current normalized row-stochastic transition matrix:
        T = softmax(T_hat + delta_T, dim=-1)
        """
        return self.transition.get_T_final()

    def get_transition_inverse(self, reg_lambda: float = 1e-4) -> torch.Tensor:
        """
        Computes the Tikhonov-regularized pseudo-inverse of T:
        T_inv = (T.T @ T + reg_lambda * I)^{-1} @ T.T
        """
        return self.transition.get_inverse(reg_lambda=reg_lambda)

    def compute_loss(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        observed_ratings: torch.Tensor,
        return_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Computes forward risk-consistent loss with self-guided instance re-weighting.
        
        Args:
            user_ids: [B]
            item_ids: [B]
            observed_ratings: [B] (1-indexed 1..K or 0-indexed 0..K-1)
            return_weights: If True, also returns detached instance weights [B]
            
        Returns:
            loss: scalar tensor with gradient
            weights: [B] detached sample weights if return_weights=True, else None
        """
        batch_size = user_ids.size(0)
        if batch_size == 0:
            zero_loss = torch.tensor(0.0, device=user_ids.device, requires_grad=True)
            return (zero_loss, torch.empty(0, device=user_ids.device)) if return_weights else (zero_loss, None)

        # 1. Clean posterior probability prediction
        clean_probs = self.forward(user_ids, item_ids)  # [B, K]

        # 2. Forward risk-consistent correction: p_tilde = clean_probs @ T
        noisy_probs = self.transition(clean_probs)  # [B, K]

        # 3. Align label indexing: convert 1-indexed to 0-indexed
        if observed_ratings.min() >= 1 and observed_ratings.max() <= self.num_classes:
            labels = (observed_ratings - 1).long()
        else:
            labels = observed_ratings.long()
        labels = labels.clamp(0, self.num_classes - 1)

        # 4. Extract observed class probability with numerical floor
        p_observed = noisy_probs[torch.arange(batch_size, device=user_ids.device), labels]
        p_observed = p_observed.clamp(min=1e-12, max=1.0)
        nll_per_sample = -torch.log(p_observed)  # [B]

        # 5. Self-Guided instance re-weighting (SGDL)
        # Detach loss values to avoid circular minimization via weight decay
        with torch.no_grad():
            weights = torch.exp(-nll_per_sample / max(1e-6, self.tau_loss))
            if self.loss_trunc is not None:
                trunc_mask = (nll_per_sample <= self.loss_trunc).float()
                weights = weights * trunc_mask
            weights = torch.clamp(weights, min=self.min_weight, max=1.0)

        # 6. Weighted batch risk loss normalized by weight sum
        weighted_loss = torch.sum(weights * nll_per_sample) / (torch.sum(weights) + 1e-8)

        # 7. Tikhonov/Frobenius regularization on delta_T slack parameter
        reg_loss = self.delta_T_reg * torch.norm(self.transition.delta_T, p="fro") ** 2

        total_loss = weighted_loss + reg_loss
        return (total_loss, weights) if return_weights else (total_loss, None)

    def extract_interaction_weights(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        observed_ratings: torch.Tensor
    ) -> torch.Tensor:
        """
        Inference helper to compute instance reliability weights for Denoising-ROC-AUC evaluation.
        """
        self.eval()
        with torch.no_grad():
            _, weights = self.compute_loss(user_ids, item_ids, observed_ratings, return_weights=True)
            return weights

    def score_candidates_batch(
        self,
        user_ids: torch.Tensor,
        candidate_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Monolithic candidate scoring: computes expected rating scores for candidates.
        Inherits optimized evaluation embedding caching from LightGCN.
        
        Args:
            user_ids: 1D Tensor [B]
            candidate_matrix: 2D Tensor [B, C] or 1D Tensor [C]
            
        Returns:
            scores: [B, C] expected ratings in range [1.0, 5.0]
        """
        return super().score_candidates_batch(user_ids, candidate_matrix)
