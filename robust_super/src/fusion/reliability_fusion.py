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
            u_np = u_batch.cpu().numpy()
            i_np = i_batch.cpu().numpy()

            for idx in range(len(u_np)):
                R_RRFN[(int(u_np[idx]), int(i_np[idx]))] = float(np.clip(r_ratio[idx], 0.05, 1.0))

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
    """
    all_keys = set(R_RRFN.keys()) | set(R_LLM.keys()) | set(R_bomb.keys())
    weights: Dict[Tuple[int, int], float] = {}

    total_w = alpha + beta + gamma
    a = alpha / max(1e-6, total_w)
    b = beta / max(1e-6, total_w)
    c = gamma / max(1e-6, total_w)

    for key in all_keys:
        r_r = R_RRFN.get(key, 0.50)
        r_l = R_LLM.get(key, 0.50)
        r_b = R_bomb.get(key, 0.50)

        fused = (a * r_r) + (b * r_l) + (c * r_b)
        weights[key] = float(np.clip(fused, min_weight, 1.0))

    return weights
