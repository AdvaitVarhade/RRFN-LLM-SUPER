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
    polarity_threshold: float = 0.70
) -> Dict[Tuple[int, int], float]:
    """
    Computes R_bomb(u, i) in [0.0, 1.0].
    Higher value = authentic / benign interaction.
    Lower value = anomalous coordinated review bombing.
    """
    R_bomb: Dict[Tuple[int, int], float] = {}
    all_keys = set(temporal_accel.keys()) | set(polarity_skew.keys())

    # Renormalize weights
    total_omega = omega_1 + omega_2 + omega_3
    w1 = omega_1 / max(1e-6, total_omega)
    w2 = omega_2 / max(1e-6, total_omega)
    w3 = omega_3 / max(1e-6, total_omega)

    for key in all_keys:
        a = temporal_accel.get(key, 1.0)
        s = polarity_skew.get(key, 0.5)
        sim = semantic_sim.get(key, 0.0)

        # Normalize acceleration: a in [1, 5] -> normalized signal
        norm_a = min(1.0, max(0.0, (a - 1.0) / (burst_threshold - 1.0 + 1e-6)))
        norm_s = min(1.0, max(0.0, (s - 0.5) / (polarity_threshold - 0.5 + 1e-6)))

        bombing_signal = (w1 * norm_a) + (w2 * norm_s) + (w3 * sim)
        suspicion = 1.0 / (1.0 + np.exp(-(bombing_signal * 4.0 - 2.0 + sigmoid_bias)))
        
        # Reliability is 1 - suspicion
        r_bomb = float(np.clip(1.0 - suspicion, 0.05, 1.0))
        R_bomb[key] = r_bomb

    return R_bomb

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
    """
    from sklearn.metrics import f1_score, roc_auc_score
    results = []
    
    omega_1_vals = np.linspace(0.0, 1.0, steps)
    for w1 in omega_1_vals:
        w2 = 1.0 - w1
        r_bomb = compute_bomb_scores(temporal_accel, polarity_skew, semantic_sim, omega_1=w1, omega_2=w2, omega_3=0.0)
        
        y_true = []
        y_pred = []
        y_scores = []
        for key, gt in ground_truth_labels.items():
            score = r_bomb.get(key, 0.5)
            y_true.append(gt)
            y_scores.append(score)
            y_pred.append(1 if score >= 0.5 else 0)
            
        y_true_arr = np.array(y_true)
        if len(np.unique(y_true_arr)) > 1:
            f1 = float(f1_score(y_true_arr, y_pred, average="binary", zero_division=0))
            auc = float(roc_auc_score(y_true_arr, y_scores))
        else:
            f1 = 1.0
            auc = 1.0
            
        results.append({
            "omega_1_temporal": float(round(w1, 2)),
            "omega_2_polarity": float(round(w2, 2)),
            "Denoising_F1": f1,
            "ROC_AUC": auc
        })
        
    return results

