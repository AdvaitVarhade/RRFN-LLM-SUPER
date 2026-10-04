import numpy as np
from typing import Dict, Tuple, List, Optional, Any

def compute_bomb_scores(
    temporal_accel: Dict[Tuple[int, int], float],
    polarity_skew: Dict[Tuple[int, int], float],
    semantic_sim: Dict[Tuple[int, int], float],
    omega_1: float = 0.60,
    omega_2: float = 0.40,
    omega_3: float = 0.00,
    sigmoid_bias: float = 0.0,
    burst_threshold: float = 2.5,
    polarity_threshold: float = 0.70,
    sigmoid_slope: float = 4.0,
    sigmoid_midpoint: float = 0.50
) -> Dict[Tuple[int, int], float]:
    """
    Computes R_bomb(u, i) in [0.0, 1.0].
    Higher value = authentic / benign interaction.
    Lower value = anomalous coordinated review bombing.
    Vectorized array operations over all interaction keys.
    """
    all_keys = list(set(temporal_accel.keys()) | set(polarity_skew.keys()))
    if not all_keys:
        return {}

    # Renormalize weights
    total_omega = omega_1 + omega_2 + omega_3
    w1 = omega_1 / max(1e-6, total_omega)
    w2 = omega_2 / max(1e-6, total_omega)
    w3 = omega_3 / max(1e-6, total_omega)

    a = np.array([temporal_accel.get(k, 1.0) for k in all_keys], dtype=np.float32)
    s = np.array([polarity_skew.get(k, 0.5) for k in all_keys], dtype=np.float32)
    sim = np.array([semantic_sim.get(k, 0.0) for k in all_keys], dtype=np.float32)

    norm_a = np.clip((a - 1.0) / (burst_threshold - 1.0 + 1e-6), 0.0, 1.0)
    norm_s = np.clip((s - 0.5) / (polarity_threshold - 0.5 + 1e-6), 0.0, 1.0)

    bombing_signal = (w1 * norm_a) + (w2 * norm_s) + (w3 * sim)
    # Dynamically scaled calibrated logistic transformation with exp argument clamped
    logit = -(sigmoid_slope * (bombing_signal - sigmoid_midpoint) + sigmoid_bias)
    logit_clamped = np.clip(logit, -50.0, 50.0)
    suspicion = 1.0 / (1.0 + np.exp(logit_clamped))

    # Reliability is 1 - suspicion
    r_bomb = np.clip(1.0 - suspicion, 0.05, 1.0)
    return {k: float(v) for k, v in zip(all_keys, r_bomb)}

def _compute_binary_f1(y_true: np.ndarray, y_pred: np.ndarray, total_pos_true: int) -> float:
    tp = int(np.count_nonzero(y_true & y_pred))
    total_pos_pred = int(np.count_nonzero(y_pred))
    denom = total_pos_true + total_pos_pred
    if denom == 0:
        return 0.0
    return float(2.0 * tp / denom)

def _compute_binary_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    desc_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score_sorted = y_score[desc_indices]
    y_true_sorted = y_true[desc_indices]

    diff_indices = np.where(np.diff(y_score_sorted))[0]
    threshold_idxs = np.r_[diff_indices, y_true.size - 1]

    tps = np.cumsum(y_true_sorted)[threshold_idxs]
    fps = (1 + threshold_idxs) - tps

    if tps[-1] == 0 or fps[-1] == 0:
        return 0.5

    tpr = np.r_[0, tps / tps[-1]]
    fpr = np.r_[0, fps / fps[-1]]

    return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2.0))

def sweep_omega_sensitivity(
    temporal_accel: Dict[Tuple[int, int], float],
    polarity_skew: Dict[Tuple[int, int], float],
    semantic_sim: Dict[Tuple[int, int], float],
    ground_truth_labels: Dict[Tuple[int, int], int],
    steps: int = 11
) -> List[Dict[str, float]]:
    """
    Sweeps omega_1 (temporal) vs omega_2 (polarity) from 0.0 to 1.0
    and calculates F1 detection score against ground truth labels.
    Vectorized 2D matrix multiplication broadcasting:
    W @ X computes all steps simultaneously.
    """
    if not ground_truth_labels:
        return []

    n_keys = len(ground_truth_labels)
    gt = np.fromiter(ground_truth_labels.values(), dtype=np.int32, count=n_keys)
    a = np.fromiter((temporal_accel.get(k, 1.0) for k in ground_truth_labels), dtype=np.float32, count=n_keys)
    s = np.fromiter((polarity_skew.get(k, 0.5) for k in ground_truth_labels), dtype=np.float32, count=n_keys)

    norm_a = np.clip((a - 1.0) / (2.5 - 1.0 + 1e-6), 0.0, 1.0)
    norm_s = np.clip((s - 0.5) / (0.70 - 0.5 + 1e-6), 0.0, 1.0)

    w1_vals = np.linspace(0.0, 1.0, steps, dtype=np.float32)
    W = np.stack([w1_vals, 1.0 - w1_vals], axis=1)  # [steps, 2]
    feats = np.stack([norm_a, norm_s], axis=0)      # [2, N]

    # 2D Matrix multiplication computes all steps simultaneously
    signals = W @ feats                             # [steps, N]
    logits = -(4.0 * (signals - 0.50))
    logits_clamped = np.clip(logits, -50.0, 50.0)
    suspicions = 1.0 / (1.0 + np.exp(logits_clamped))
    scores = np.clip(1.0 - suspicions, 0.05, 1.0)   # [steps, N]

    results = []
    has_binary_classes = bool(len(np.unique(gt)) > 1)
    total_pos_true = int(np.count_nonzero(gt)) if has_binary_classes else 0

    for idx, w1 in enumerate(w1_vals):
        sc = scores[idx]
        pred = (sc >= 0.50).astype(np.int32)
        if has_binary_classes:
            f1 = _compute_binary_f1(gt, pred, total_pos_true)
            auc = _compute_binary_roc_auc(gt, sc)
        else:
            f1 = 1.0
            auc = 1.0
        results.append({
            "omega_1_temporal": float(round(float(w1), 2)),
            "omega_2_polarity": float(round(float(1.0 - w1), 2)),
            "Denoising_F1": f1,
            "ROC_AUC": auc
        })

    return results

