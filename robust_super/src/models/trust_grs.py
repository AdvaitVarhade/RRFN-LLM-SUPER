"""
robust_super/src/models/trust_grs.py
------------------------------------
Trust-GRS: Trustworthy Training Framework for Graph Neural Network Based
Recommender Systems Against Shilling Attacks (Mu et al., AAAI 2025).

Implements:
1. Item rating prior estimation: mu_i, sigma_i^2
2. Shill anomaly estimation: s_rating(u, i) = exp(-(r_ui - mu_i)^2 / (2*sigma_i^2 + eps))
3. Profile structural trust: s_user(u) based on head item concentration and rating variance
4. Edge trust weight tensor: W_trust(u, i) = clamp(s_rating * (1 - s_user), eps_min, 1.0)
5. Trust-weighted normalized Laplacian: L_tilde = D_tilde^{-1/2} (A * W_trust) D_tilde^{-1/2} with .coalesce()
6. Trust-weighted training loss and monolithic fast candidate evaluation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
from typing import Dict, List, Tuple, Optional, Set

from .lightgcn import LightGCN


class TrustGRS(LightGCN):
    """
    Trust-GRS robust recommendation model with trust-aware graph convolution
    and anomaly-attenuated objective optimization.
    """
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        num_layers: int = 3,
        num_classes: int = 5,
        eps_rating: float = 1e-2,
        tau_var: float = 0.5,
        head_ratio_threshold: float = 0.20,
        eps_min: float = 1e-3,
    ):
        super().__init__(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            num_classes=num_classes
        )
        self.eps_rating = float(eps_rating)
        self.tau_var = float(tau_var)
        self.head_ratio_threshold = float(head_ratio_threshold)
        self.eps_min = float(eps_min)

        # Trust tracking dictionaries
        self.trust_weights: Dict[Tuple[int, int], float] = {}
        self.user_trust_scores: Dict[int, float] = {}
        self.item_stats: Dict[int, Tuple[float, float]] = {}

    def compute_trust_weights(
        self,
        train_dict: Dict[int, List[Tuple[int, int, int]]],
        head_items: Optional[Set[int]] = None
    ) -> Dict[Tuple[int, int], float]:
        """
        Computes edge trust weights W_trust(u, i) for all observed interactions.
        """
        # 1. Identify head items if not explicitly provided (top Pareto volume items)
        if head_items is None:
            item_counts: Dict[int, int] = {}
            for u, inters in train_dict.items():
                for item, _, _ in inters:
                    item_counts[item] = item_counts.get(item, 0) + 1
            sorted_items = sorted(item_counts.keys(), key=lambda x: item_counts[x], reverse=True)
            k_head = max(1, int(len(sorted_items) * self.head_ratio_threshold))
            head_items = set(sorted_items[:k_head])

        # 2. Compute item rating empirical priors: mu_i and sigma_i^2
        item_ratings: Dict[int, List[float]] = {}
        all_ratings: List[float] = []
        for u, inters in train_dict.items():
            for item, r, _ in inters:
                item_ratings.setdefault(item, []).append(float(r))
                all_ratings.append(float(r))

        global_mu = float(np.mean(all_ratings)) if all_ratings else 3.5
        global_sigma_sq = float(np.var(all_ratings)) if len(all_ratings) > 1 else 1.0

        item_stats: Dict[int, Tuple[float, float]] = {}
        for item, r_list in item_ratings.items():
            mu_i = float(np.mean(r_list))
            sigma_sq_i = float(np.var(r_list)) if len(r_list) > 1 else 0.0
            item_stats[item] = (mu_i, sigma_sq_i)
        self.item_stats = item_stats

        # 3. Compute profile structural trust: s_user(u)
        user_trust_scores: Dict[int, float] = {}
        for u, inters in train_dict.items():
            if not inters:
                user_trust_scores[u] = 1.0
                continue
            
            d_u = len(inters)
            head_cnt = sum(1 for item, _, _ in inters if item in head_items)
            h_u = head_cnt / max(1, d_u)
            
            u_ratings = [float(r) for _, r, _ in inters]
            v_u = float(np.var(u_ratings)) if d_u > 1 else 0.0
            
            # Anomaly suspicion score: high head ratio combined with low rating variance
            s_user = float(np.clip(h_u * np.exp(-v_u / max(1e-4, self.tau_var)), 0.0, 1.0 - self.eps_min))
            user_trust_scores[u] = 1.0 - s_user  # Trustworthiness in [eps_min, 1.0]

        self.user_trust_scores = user_trust_scores

        # 4. Joint edge trust tensor: W_trust(u, i)
        trust_weights: Dict[Tuple[int, int], float] = {}
        for u, inters in train_dict.items():
            t_u = user_trust_scores.get(u, 1.0)
            for item, r, _ in inters:
                mu_i, sigma_sq_i = item_stats.get(item, (global_mu, global_sigma_sq))
                deviation = (float(r) - mu_i) ** 2
                s_rating = float(np.exp(-deviation / (2.0 * sigma_sq_i + self.eps_rating)))
                
                w_trust = float(np.clip(s_rating * t_u, self.eps_min, 1.0))
                trust_weights[(u, item)] = w_trust

        self.trust_weights = trust_weights
        return trust_weights

    def set_adjacency(
        self,
        train_dict: Dict[int, List[Tuple[int, int, int]]],
        device: str = "cpu",
        head_items: Optional[Set[int]] = None
    ):
        """
        Constructs trust-weighted normalized Laplacian:
        L_tilde = D_tilde^{-1/2} (A * W_trust) D_tilde^{-1/2}
        ensuring sparse tensor coalescing (.coalesce()).
        """
        # Precompute trust weights if not already present
        if not self.trust_weights:
            self.compute_trust_weights(train_dict, head_items=head_items)

        rows, cols, weights = [], [], []
        for u, interactions in train_dict.items():
            for item, _, _ in interactions:
                rows.append(u)
                cols.append(item)
                w = self.trust_weights.get((u, item), 1.0)
                weights.append(w)

        rows = np.array(rows, dtype=np.int64)
        cols = np.array(cols, dtype=np.int64)
        weights = np.array(weights, dtype=np.float32)

        if len(rows) == 0:
            indices = torch.empty((2, 0), dtype=torch.int64)
            values = torch.empty(0, dtype=torch.float32)
            total_nodes = self.num_users + self.num_items
            shape = torch.Size((total_nodes, total_nodes))
            self.adj_norm = torch.sparse_coo_tensor(indices, values, shape, device=device).coalesce()
            self._cached_user_embs = None
            self._cached_item_embs = None
            return

        # Trust-weighted bipartite matrix R_trust of shape (num_users, num_items)
        R_trust = sp.coo_matrix((weights, (rows, cols)), shape=(self.num_users, self.num_items)).tocsr()

        # Symmetric block adjacency: A_trust = [[0, R_trust], [R_trust^T, 0]]
        adj = sp.bmat([[None, R_trust], [R_trust.T, None]], format="csr")

        # Symmetrically normalized Laplacian: D_tilde^{-1/2} A_trust D_tilde^{-1/2}
        rowsum = np.array(adj.sum(axis=1)).flatten()
        d_inv = np.zeros_like(rowsum, dtype=np.float32)
        pos_mask = rowsum > 0
        d_inv[pos_mask] = np.power(rowsum[pos_mask], -0.5)
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(adj).dot(d_mat).tocoo()

        # Convert to coalesced PyTorch sparse COO tensor
        indices = torch.from_numpy(np.vstack((norm_adj.row, norm_adj.col)).astype(np.int64))
        values = torch.from_numpy(norm_adj.data.astype(np.float32))
        shape = torch.Size(norm_adj.shape)

        self.adj_norm = torch.sparse_coo_tensor(indices, values, shape, device=device).coalesce()
        self._cached_user_embs = None
        self._cached_item_embs = None

    def compute_trust_weighted_loss(
        self,
        probs: torch.Tensor,
        ratings: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        eps: float = 1e-8
    ) -> torch.Tensor:
        """
        Computes trust-weighted cross entropy loss:
        L = sum(w_k * -log(p_{k, y_k})) / (sum(w_k) + eps)
        """
        # Map Likert ratings (1..5) to 0-indexed classes (0..4)
        targets = (ratings - 1).clamp(0, self.num_classes - 1)
        batch_size = probs.size(0)
        
        # Gather predicted probability for observed target class
        p_target = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        nll = -torch.log(p_target.clamp_min(1e-12))

        if weights is not None:
            w = weights.view(-1).float()
            loss = torch.sum(w * nll) / (torch.sum(w) + eps)
        else:
            loss = torch.mean(nll)

        return loss

    def score_candidates_batch(
        self,
        user_ids: torch.Tensor,
        candidate_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Batched multi-user candidate scoring.
        """
        return super().score_candidates_batch(user_ids, candidate_matrix)
