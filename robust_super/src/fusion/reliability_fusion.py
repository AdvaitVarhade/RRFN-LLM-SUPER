import torch
import numpy as np
from typing import Dict, List, Tuple
from torch.utils.data import DataLoader

def compute_R_RRFN(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: str = "cpu",
    epsilon: float = 1e-8
) -> Dict[Tuple[int, int], float]:
    """
    Computes statistical consistency score R_RRFN(u, i) from the warm-start model:
    R_RRFN(u, i) = P(Y* = y_observed | x_ui) / (max_k P(Y* = k | x_ui) + epsilon)
    """
    model.eval()
    if hasattr(model, "compute_all_embeddings") and hasattr(model, "_cached_user_embs"):
        if model._cached_user_embs is None or model._cached_item_embs is None:
            u_all, i_all = model.compute_all_embeddings()
            model._cached_user_embs = u_all
            model._cached_item_embs = i_all

    R_RRFN: Dict[Tuple[int, int], float] = {}

    with torch.no_grad():
        for batch in data_loader:
            if len(batch) < 3:
                continue
            u_batch, i_batch, y_batch = batch[0], batch[1], batch[2]
            u_b = u_batch.to(device)
            i_b = i_batch.to(device)

            probs = model(u_b, i_b)  # [B, 5]
            max_probs = probs.max(dim=-1).values

            # Map 1-indexed ratings (1..5) to 0-indexed (0..4)
            labels = (y_batch - 1).clamp(0, 4).to(device)
            p_observed = probs[torch.arange(len(u_batch), device=device), labels]

            r_ratio = (p_observed / (max_probs + epsilon)).cpu().numpy()
            r_ratio_clipped = np.clip(r_ratio, 0.05, 1.0)
            u_np = u_batch.cpu().numpy()
            i_np = i_batch.cpu().numpy()

            for u_val, i_val, r_val in zip(u_np, i_np, r_ratio_clipped):
                R_RRFN[(int(u_val), int(i_val))] = float(r_val)

    return R_RRFN

def fuse_reliability_scores(
    R_RRFN: Dict[Tuple[int, int], float],
    R_LLM: Dict[Tuple[int, int], float],
    R_bomb: Dict[Tuple[int, int], float],
    alpha: float = 0.50,
    beta: float = 0.30,
    gamma: float = 0.20,
    min_weight: float = 0.02
) -> Dict[Tuple[int, int], float]:
    """
    Fuses three complementary reliability signals into continuous interaction weights w_{ui}:
    w_{ui} = alpha * R_RRFN + beta * R_LLM + gamma * R_bomb
    Fully vectorized with NumPy array operations and broadcasting.
    """
    all_keys = list(set(R_RRFN.keys()) | set(R_LLM.keys()) | set(R_bomb.keys()))
    if not all_keys:
        return {}

    total_w = alpha + beta + gamma
    a = alpha / max(1e-6, total_w)
    b = beta / max(1e-6, total_w)
    c = gamma / max(1e-6, total_w)

    r_r = np.array([R_RRFN.get(k, 0.50) for k in all_keys], dtype=np.float32)
    r_l = np.array([R_LLM.get(k, 0.50) for k in all_keys], dtype=np.float32)
    r_b = np.array([R_bomb.get(k, 0.50) for k in all_keys], dtype=np.float32)

    fused = (a * r_r) + (b * r_l) + (c * r_b)
    fused_clipped = np.clip(fused, min_weight, 1.0)

    return {k: float(v) for k, v in zip(all_keys, fused_clipped)}
