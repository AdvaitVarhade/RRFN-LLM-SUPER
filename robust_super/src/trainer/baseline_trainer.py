"""
robust_super/src/trainer/baseline_trainer.py
--------------------------------------------
Standardized trainer for monolithic competitive baselines:
1. Plain LightGCN (SIGIR '20) — standard unweighted cross-entropy
2. Plain SGL (SIGIR '21) — unweighted cross-entropy + InfoNCE contrastive loss
3. T-CE / SGDL (NeurIPS '22) — forward risk-consistent loss + self-guided instance re-weighting
4. Trust-GRS (AAAI '25) — trust-weighted graph convolution + trust-weighted loss

Features:
- Full catalog monolithic training without Pareto head/tail catalog partitioning
- Direct candidate ranking without blueprint interleaving or user inclination calibration
- Standardized validation tracking, early stopping, and model checkpoint restoration
"""

import copy
import random
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .scheduler import WarmupReduceLROnPlateau
from ..rrfn.contrastive_loss import infonce_loss, joint_risk_contrastive_loss


class BaselineTrainer:
    """
    Standardized trainer for monolithic collaborative filtering baselines.
    """
    def __init__(
        self,
        device: str = "cpu",
        lr: float = 0.002,
        weight_decay: float = 1e-5,
        gradient_clip_norm: float = 5.0,
        epochs: int = 15,
        early_stopping_patience: int = 6,
        warm_epochs: int = 2,
        baseline_type: str = "standard",  # standard | sgl | tce_sgdl | trust_grs
        lambda_cl: float = 0.15,
        temperature_cl: float = 0.15,
    ):
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.gradient_clip_norm = gradient_clip_norm
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.warm_epochs = warm_epochs
        self.baseline_type = baseline_type.lower()
        self.lambda_cl = lambda_cl
        self.temperature_cl = temperature_cl

    def train(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_dict: Dict[int, Tuple[int, int, int]],
        anchor_selector=None,
        train_dict: Optional[Dict[int, List[Tuple[int, int, int]]]] = None,
        head_items: Optional[Set[int]] = None
    ) -> nn.Module:
        """
        Dispatches training to the designated baseline routine.
        """
        model.to(self.device)

        if self.baseline_type == "standard":
            return self._train_standard(model, train_loader, val_dict)
        elif self.baseline_type == "sgl":
            return self._train_sgl(model, train_loader, val_dict)
        elif self.baseline_type == "tce_sgdl":
            return self._train_tce_sgdl(model, train_loader, val_dict, anchor_selector)
        elif self.baseline_type == "trust_grs":
            return self._train_trust_grs(model, train_loader, val_dict, train_dict, head_items)
        else:
            raise ValueError(f"Unknown baseline_type: {self.baseline_type}")

    def _train_standard(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_dict: Dict[int, Tuple[int, int, int]]
    ) -> nn.Module:
        """Plain LightGCN training with unweighted cross entropy."""
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = WarmupReduceLROnPlateau(optimizer, warmup_epochs=2, base_lr=self.lr, patience=4)
        criterion = nn.NLLLoss()

        best_score = -float("inf")
        best_state = None
        patience = 0
        loss_history, val_history = [], []

        for epoch in range(1, self.epochs + 1):
            model.train()
            epoch_loss = 0.0
            num_batches = 0

            for batch in train_loader:
                u_b = batch[0].to(self.device)
                i_b = batch[1].to(self.device)
                y_b = batch[2].to(self.device)
                targets = (y_b - 1).clamp(0, 4)

                optimizer.zero_grad()
                probs = model(u_b, i_b)
                log_probs = torch.log(probs.clamp_min(1e-12))
                loss = criterion(log_probs, targets)

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip_norm)
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / max(1, num_batches)
            val_recall = self.evaluate_validation(model, val_dict)
            loss_history.append(avg_loss)
            val_history.append(val_recall)
            scheduler.step(val_recall)

            if val_recall > best_score:
                best_score = val_recall
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= self.early_stopping_patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.loss_history = loss_history
        model.val_history = val_history
        return model

    def _train_sgl(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_dict: Dict[int, Tuple[int, int, int]]
    ) -> nn.Module:
        """Plain SGL training: unweighted cross entropy + InfoNCE contrastive loss."""
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = WarmupReduceLROnPlateau(optimizer, warmup_epochs=2, base_lr=self.lr, patience=4)
        criterion = nn.NLLLoss()

        best_score = -float("inf")
        best_state = None
        patience = 0
        loss_history, val_history = [], []

        for epoch in range(1, self.epochs + 1):
            model.train()
            epoch_loss = 0.0
            num_batches = 0

            for batch in train_loader:
                u_b = batch[0].to(self.device)
                i_b = batch[1].to(self.device)
                y_b = batch[2].to(self.device)
                targets = (y_b - 1).clamp(0, 4)

                optimizer.zero_grad()
                probs = model(u_b, i_b)
                log_probs = torch.log(probs.clamp_min(1e-12))
                main_loss = criterion(log_probs, targets)

                # Contrastive graph views
                if hasattr(model, "get_cl_embeddings"):
                    v1, v2 = model.get_cl_embeddings()
                    num_users = getattr(model, "num_users", 0)
                    u_uniq = torch.unique(u_b)
                    i_uniq = torch.unique(i_b)
                    if num_users > 0 and len(u_uniq) > 1 and len(i_uniq) > 1:
                        cl_u = infonce_loss(v1[u_uniq], v2[u_uniq], temperature=self.temperature_cl)
                        cl_i = infonce_loss(v1[num_users + i_uniq], v2[num_users + i_uniq], temperature=self.temperature_cl)
                        cl_total = cl_u + cl_i
                        loss = joint_risk_contrastive_loss(main_loss, cl_total, lambda_cl=self.lambda_cl)
                    else:
                        loss = main_loss
                else:
                    loss = main_loss

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip_norm)
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / max(1, num_batches)
            val_recall = self.evaluate_validation(model, val_dict)
            loss_history.append(avg_loss)
            val_history.append(val_recall)
            scheduler.step(val_recall)

            if val_recall > best_score:
                best_score = val_recall
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= self.early_stopping_patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.loss_history = loss_history
        model.val_history = val_history
        return model

    def _train_tce_sgdl(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_dict: Dict[int, Tuple[int, int, int]],
        anchor_selector=None
    ) -> nn.Module:
        """T-CE / SGDL: unweighted warm start, anchor estimation, forward risk correction."""
        # 1. Warm-start training
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.NLLLoss()

        model.train()
        for _ in range(self.warm_epochs):
            for batch in train_loader:
                u_b = batch[0].to(self.device)
                i_b = batch[1].to(self.device)
                y_b = batch[2].to(self.device)
                targets = (y_b - 1).clamp(0, 4)
                optimizer.zero_grad()
                probs = model(u_b, i_b)
                log_probs = torch.log(probs.clamp_min(1e-12))
                loss = criterion(log_probs, targets)
                loss.backward()
                optimizer.step()

        # 2. Anchor selection and transition estimation
        if anchor_selector is not None and hasattr(model, "estimate_transition_from_anchors"):
            anchors = anchor_selector.select_anchors(model, train_loader, device=self.device)
            model.estimate_transition_from_anchors(anchors)

        # 3. Robust training loop with SGDL loss
        params = list(model.parameters())
        if hasattr(model, "transition") and hasattr(model.transition, "delta_T"):
            if not any(p is model.transition.delta_T for p in params):
                params.append(model.transition.delta_T)

        optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = WarmupReduceLROnPlateau(optimizer, warmup_epochs=2, base_lr=self.lr, patience=4)

        best_score = -float("inf")
        best_state = None
        patience = 0
        loss_history, val_history = [], []

        for epoch in range(1, self.epochs + 1):
            model.train()
            epoch_loss = 0.0
            num_batches = 0

            for batch in train_loader:
                u_b = batch[0].to(self.device)
                i_b = batch[1].to(self.device)
                y_b = batch[2].to(self.device)

                optimizer.zero_grad()
                if hasattr(model, "compute_loss"):
                    loss, _ = model.compute_loss(u_b, i_b, y_b)
                else:
                    targets = (y_b - 1).clamp(0, 4)
                    probs = model(u_b, i_b)
                    loss = criterion(torch.log(probs.clamp_min(1e-12)), targets)

                loss.backward()
                nn.utils.clip_grad_norm_(params, self.gradient_clip_norm)
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / max(1, num_batches)
            val_recall = self.evaluate_validation(model, val_dict)
            loss_history.append(avg_loss)
            val_history.append(val_recall)
            scheduler.step(val_recall)

            if val_recall > best_score:
                best_score = val_recall
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= self.early_stopping_patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.loss_history = loss_history
        model.val_history = val_history
        return model

    def _train_trust_grs(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_dict: Dict[int, Tuple[int, int, int]],
        train_dict: Optional[Dict[int, List[Tuple[int, int, int]]]] = None,
        head_items: Optional[Set[int]] = None
    ) -> nn.Module:
        """Trust-GRS: trust-weighted Laplacian and trust-weighted loss."""
        # 1. Initialize trust weights and trust-weighted normalized Laplacian
        if train_dict is not None and hasattr(model, "set_adjacency"):
            model.set_adjacency(train_dict, device=self.device, head_items=head_items)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = WarmupReduceLROnPlateau(optimizer, warmup_epochs=2, base_lr=self.lr, patience=4)

        best_score = -float("inf")
        best_state = None
        patience = 0
        loss_history, val_history = [], []

        for epoch in range(1, self.epochs + 1):
            model.train()
            epoch_loss = 0.0
            num_batches = 0

            for batch in train_loader:
                u_b = batch[0].to(self.device)
                i_b = batch[1].to(self.device)
                y_b = batch[2].to(self.device)
                w_b = batch[3].to(self.device) if len(batch) > 3 else None

                optimizer.zero_grad()
                probs = model(u_b, i_b)
                if hasattr(model, "compute_trust_weighted_loss"):
                    loss = model.compute_trust_weighted_loss(probs, y_b, weights=w_b)
                else:
                    targets = (y_b - 1).clamp(0, 4)
                    log_probs = torch.log(probs.clamp_min(1e-12))
                    nll = nn.functional.nll_loss(log_probs, targets, reduction='none')
                    if w_b is not None:
                        loss = torch.sum(w_b * nll) / (torch.sum(w_b) + 1e-8)
                    else:
                        loss = torch.mean(nll)

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip_norm)
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / max(1, num_batches)
            val_recall = self.evaluate_validation(model, val_dict)
            loss_history.append(avg_loss)
            val_history.append(val_recall)
            scheduler.step(val_recall)

            if val_recall > best_score:
                best_score = val_recall
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= self.early_stopping_patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.loss_history = loss_history
        model.val_history = val_history
        return model

    def evaluate_validation(
        self,
        model: nn.Module,
        val_dict: Dict[int, Tuple[int, int, int]],
        k: int = 10,
        max_users: int = 200
    ) -> float:
        """Fast validation evaluation across all catalog items."""
        model.eval()
        hits, total = 0, 0
        sample_users = list(val_dict.keys())[:min(max_users, len(val_dict))]
        num_items = getattr(model, "num_items", 0)

        with torch.no_grad():
            for u in sample_users:
                target_item, _, _ = val_dict[u]
                total += 1
                if num_items > 1:
                    pool = [i for i in range(num_items) if i != target_item]
                    negs = pool[:min(99, len(pool))]
                    candidates = [target_item] + negs
                    scores = model.score_items(u, candidates, device=self.device)
                    topk_indices = torch.topk(scores, k=min(k, len(candidates))).indices.cpu().numpy()
                    if 0 in topk_indices:
                        hits += 1
                else:
                    scores = model.score_items(u, [target_item], device=self.device)
                    if scores.numel() > 0:
                        hits += 1
        return hits / max(1, total)

    @staticmethod
    def generate_recommendations(
        model: nn.Module,
        eval_users: List[int],
        clean_split,
        top_k: int = 10,
        rng: Optional[random.Random] = None,
        device: str = "cpu"
    ) -> Dict[int, List[int]]:
        """
        Generates monolithic Top-K recommendations by directly scoring candidate items.
        """
        model.eval()
        all_items_set = set(range(clean_split.num_items))
        if rng is None:
            rng = random.Random(42)

        recs: Dict[int, List[int]] = {}
        with torch.no_grad():
            for u in eval_users:
                test_item, _, _ = clean_split.test_dict[u]
                seen = set(item for item, _, _ in clean_split.train_dict.get(u, [])) | {test_item}
                unseen = list(all_items_set - seen)
                negs = rng.sample(unseen, min(99, len(unseen)))
                candidates = [test_item] + negs

                scores = model.score_items(u, candidates, device=device)
                topk_indices = torch.topk(scores, k=min(top_k, len(candidates))).indices.cpu().numpy()
                recs[u] = [candidates[idx] for idx in topk_indices]
        return recs
