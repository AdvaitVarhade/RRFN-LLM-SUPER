import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Set, Optional

from ..rrfn.transition_matrix import NoiseTransitionMatrix
from ..rrfn.risk_loss import risk_consistent_loss
from .scheduler import WarmupReduceLROnPlateau

class InteractionDataset(Dataset):
    """
    PyTorch Dataset for user-item interactions with observed ratings and reliability weights.
    """
    def __init__(
        self,
        interactions: List[Tuple[int, int, int]], # (u, i, r)
        weights: Dict[Tuple[int, int], float]
    ):
        self.interactions = interactions
        self.weights = weights

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        u, i, r = self.interactions[idx]
        w = self.weights.get((u, i), 1.0)
        return (
            torch.tensor(u, dtype=torch.long),
            torch.tensor(i, dtype=torch.long),
            torch.tensor(r, dtype=torch.long),
            torch.tensor(w, dtype=torch.float32)
        )

def build_data_loader(
    train_dict: Dict[int, List[Tuple[int, int, int]]],
    weights: Dict[Tuple[int, int], float],
    item_filter: Optional[Set[int]] = None,
    batch_size: int = 1024,
    shuffle: bool = True
) -> DataLoader:
    """
    Constructs DataLoader filtered by item set (e.g. Head or Tail).
    """
    interactions_list = []
    for u, u_list in train_dict.items():
        for item, rating, ts in u_list:
            if item_filter is None or item in item_filter:
                interactions_list.append((u, item, rating))

    if len(interactions_list) == 0:
        # Fallback single dummy interaction if empty
        interactions_list.append((0, 0, 3))

    dataset = InteractionDataset(interactions_list, weights)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

class DualModelTrainer:
    """
    Orchestrates decoupled dual training of M_pop and M_tail using risk-consistent loss.
    """
    def __init__(
        self,
        model_factory,
        device: str = "cpu",
        lr: float = 0.001,
        weight_decay: float = 1e-5,
        delta_T_reg: float = 0.01,
        gradient_clip_norm: float = 5.0,
        epochs: int = 25,
        early_stopping_patience: int = 8
    ):
        self.model_factory = model_factory
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.delta_T_reg = delta_T_reg
        self.gradient_clip_norm = gradient_clip_norm
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience

    def warm_train(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        warm_epochs: int = 5
    ):
        """
        Warm-start training with standard unweighted cross-entropy before anchor estimation.
        """
        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for epoch in range(warm_epochs):
            total_loss = 0.0
            for u_b, i_b, y_b, _ in train_loader:
                u_b = u_b.to(self.device)
                i_b = i_b.to(self.device)
                labels = (y_b - 1).clamp(0, 4).to(self.device)

                optimizer.zero_grad()
                probs = model(u_b, i_b)  # [B, 5]
                # Log of probs for cross-entropy
                log_probs = torch.log(probs + 1e-8)
                loss = criterion(log_probs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

    def train_single_model(
        self,
        train_loader: DataLoader,
        val_dict: Dict[int, Tuple[int, int, int]],
        allowed_items: Set[int],
        T_hat: Optional[torch.Tensor] = None
    ) -> Tuple[nn.Module, NoiseTransitionMatrix]:
        """
        Trains one decoupled model (M_pop or M_tail) using the risk-consistent loss.
        """
        model = self.model_factory().to(self.device)
        transition_module = NoiseTransitionMatrix(num_classes=5).to(self.device)

        if T_hat is not None:
            transition_module.T_hat.copy_(T_hat.to(self.device))

        params = list(model.parameters()) + [transition_module.delta_T]
        optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = WarmupReduceLROnPlateau(optimizer, warmup_epochs=2, base_lr=self.lr, patience=4)

        best_val_score = -float("inf")
        best_state = None
        best_trans_state = None
        patience_count = 0

        loss_history = []
        val_history = []

        for epoch in range(1, self.epochs + 1):
            model.train()
            transition_module.train()
            epoch_loss = 0.0
            num_batches = 0

            for u_b, i_b, y_b, w_b in train_loader:
                u_b = u_b.to(self.device)
                i_b = i_b.to(self.device)
                y_b = y_b.to(self.device)
                w_b = w_b.to(self.device)

                optimizer.zero_grad()
                clean_probs = model(u_b, i_b)
                loss = risk_consistent_loss(
                    clean_probs,
                    transition_module,
                    observed_ratings=y_b,
                    weights=w_b,
                    lambda_reg=self.delta_T_reg
                )

                loss.backward()
                nn.utils.clip_grad_norm_(params, self.gradient_clip_norm)
                optimizer.step()
                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / max(1, num_batches)
            loss_history.append(avg_loss)

            # Fast validation evaluation (Recall on subset of allowed items)
            val_score = self._evaluate_val_recall(model, val_dict, allowed_items)
            val_history.append(val_score)
            scheduler.step(val_score)

            if val_score > best_val_score:
                best_val_score = val_score
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_trans_state = {k: v.cpu().clone() for k, v in transition_module.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= self.early_stopping_patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        if best_trans_state is not None:
            transition_module.load_state_dict(best_trans_state)

        # Attach training telemetry for dashboard inspection
        model.loss_history = loss_history
        model.val_history = val_history

        return model, transition_module

    def _evaluate_val_recall(self, model: nn.Module, val_dict: Dict[int, Tuple[int, int, int]], allowed_items: Set[int], k: int = 10) -> float:
        model.eval()
        hits = 0
        total = 0
        sample_users = list(val_dict.keys())[:min(200, len(val_dict))] # subset for fast validation
        allowed_list = list(allowed_items)
        if not allowed_list:
            return 0.0

        with torch.no_grad():
            for u in sample_users:
                target_item, _, _ = val_dict[u]
                if target_item not in allowed_items:
                    continue
                total += 1
                scores = model.score_items(u, allowed_list, device=self.device)
                topk_indices = torch.topk(scores, k=min(k, len(allowed_list))).indices.cpu().numpy()
                topk_items = [allowed_list[idx] for idx in topk_indices]
                if target_item in topk_items:
                    hits += 1

        return hits / max(1, total)
