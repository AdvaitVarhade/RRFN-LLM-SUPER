import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, auc

def compute_robustness_metrics(
    attacked_metrics: Dict[str, float],
    clean_baseline_metrics: Optional[Dict[str, float]],
    ground_truth_labels: Dict[Tuple[int, int], int],
    predicted_weights: Dict[Tuple[int, int], float],
    target_items: Optional[List[int]] = None,
    recommendations: Optional[Dict[int, List[int]]] = None,
    top_k: int = 10
) -> Dict[str, float]:
    """
    Computes adversarial robustness and noise mitigation metrics:
    - Delta-GKPI: % degradation in GKPI under attack
    - CSS: Calibration Stability Score (|RMSE-PC_attacked - RMSE-PC_clean|)
    - Target Promotion Lift (TPL)
    - Denoising F1 and ROC-AUC
    """
    robustness_dict: Dict[str, float] = {}

    # 1. Delta-GKPI & CSS
    if clean_baseline_metrics:
        clean_gkpi = clean_baseline_metrics.get("GKPI", 0.0)
        attack_gkpi = attacked_metrics.get("GKPI", 0.0)
        if clean_gkpi > 0:
            delta_gkpi = ((clean_gkpi - attack_gkpi) / clean_gkpi) * 100.0
        else:
            delta_gkpi = 0.0
        robustness_dict["Delta-GKPI(%)"] = float(delta_gkpi)

        clean_rmse_pc = clean_baseline_metrics.get("RMSE-PC", 0.0)
        attack_rmse_pc = attacked_metrics.get("RMSE-PC", 0.0)
        css = abs(attack_rmse_pc - clean_rmse_pc)
        robustness_dict["CSS"] = float(css)

    # 2. Denoising Quality (F1 and AUC against ground-truth noise)
    if ground_truth_labels and predicted_weights:
        y_true = []
        y_scores = []
        y_pred = []

        for key, gt_label in ground_truth_labels.items():
            w = predicted_weights.get(key, 0.5)
            y_true.append(gt_label) # 1 genuine, 0 injected
            y_scores.append(w)
            y_pred.append(1 if w >= 0.50 else 0)

        y_true_arr = np.array(y_true)
        y_scores_arr = np.array(y_scores)
        y_pred_arr = np.array(y_pred)

        # Check if both classes are present
        if len(np.unique(y_true_arr)) > 1:
            f1 = float(f1_score(y_true_arr, y_pred_arr, average="binary", zero_division=0))
            auc = float(roc_auc_score(y_true_arr, y_scores_arr))
        else:
            f1 = 1.0
            auc = 1.0

        robustness_dict["Denoising-F1"] = f1
        robustness_dict["Denoising-ROC-AUC"] = auc

    return robustness_dict

def compute_roc_pr_data(
    ground_truth_labels: Dict[Tuple[int, int], int],
    predicted_weights: Dict[Tuple[int, int], float]
) -> Dict[str, Any]:
    """
    Computes exact ROC and Precision-Recall curve arrays for UI visualization.
    """
    y_true = []
    y_scores = []
    for key, gt in ground_truth_labels.items():
        w = predicted_weights.get(key, 0.5)
        y_true.append(gt)
        y_scores.append(w)

    y_true_arr = np.array(y_true)
    y_scores_arr = np.array(y_scores)

    if len(np.unique(y_true_arr)) > 1:
        fpr, tpr, roc_thresholds = roc_curve(y_true_arr, y_scores_arr)
        roc_auc = float(auc(fpr, tpr))

        precision, recall, pr_thresholds = precision_recall_curve(y_true_arr, y_scores_arr)
        avg_precision = float(average_precision_score(y_true_arr, y_scores_arr))
    else:
        fpr = np.array([0.0, 1.0])
        tpr = np.array([1.0, 1.0])
        roc_auc = 1.0
        precision = np.array([1.0, 1.0])
        recall = np.array([0.0, 1.0])
        avg_precision = 1.0

    return {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "roc_auc": roc_auc,
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "avg_precision": avg_precision,
        "sample_size": len(y_true_arr)
    }

def generate_latex_benchmark_table(
    table_rows: List[Dict[str, Any]],
    caption: str = "Adversarial Robustness and Popularity Debiasing Performance Comparison on MovieLens-1M.",
    label: str = "tab:benchmark_results",
    full_metrics: bool = True
) -> str:
    """
    Generates publishable LaTeX Table code matching IEEE/ACM paper format with all base paper metrics.
    """
    if full_metrics:
        latex = [
            "\\begin{table*}[t]",
            "\\centering",
            "\\small",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            "\\begin{tabular}{lcccccccccc}",
            "\\toprule",
            "\\textbf{Method / System} & \\textbf{Recall@10} $\\uparrow$ & \\textbf{nDCG@10} $\\uparrow$ & \\textbf{RMSE-PC} $\\downarrow$ & \\textbf{MRMC} $\\downarrow$ & \\textbf{APLT} (\\%) $\\uparrow$ & \\textbf{LTC} (\\%) $\\uparrow$ & \\textbf{Entropy} $\\uparrow$ & \\textbf{Novelty} $\\uparrow$ & \\textbf{GKPI} $\\uparrow$ & \\textbf{$\\Delta$GKPI} (\\%) $\\downarrow$ \\\\",
            "\\midrule"
        ]

        for row in table_rows:
            method = str(row.get("Method", ""))
            recall = f"{row.get('Recall@10', 0.0):.4f}"
            ndcg = f"{row.get('nDCG@10', 0.0):.4f}"
            rmse_pc = f"{row.get('RMSE-PC', 0.0):.4f}"
            mrmc = f"{row.get('MRMC', 0.0):.4f}"
            aplt = f"{row.get('APLT@10', 0.0) * 100:.1f}\\%"
            ltc = f"{row.get('LTC@10', 0.0) * 100:.1f}\\%"
            entropy = f"{row.get('Entropy', 0.0):.4f}"
            novelty = f"{row.get('Novelty', 0.0):.2f}"
            gkpi = f"{row.get('GKPI', 0.0):.4f}"
            delta_gkpi = f"{row.get('Delta-GKPI(%)', 0.0):+.1f}\\%"

            if "Ours" in method or "RRFN" in method:
                latex.append(
                    f"\\textbf{{{method}}} & \\textbf{{{recall}}} & \\textbf{{{ndcg}}} & \\textbf{{{rmse_pc}}} & \\textbf{{{mrmc}}} & \\textbf{{{aplt}}} & \\textbf{{{ltc}}} & \\textbf{{{entropy}}} & \\textbf{{{novelty}}} & \\textbf{{{gkpi}}} & \\textbf{{{delta_gkpi}}} \\\\"
                )
            else:
                latex.append(
                    f"{method} & {recall} & {ndcg} & {rmse_pc} & {mrmc} & {aplt} & {ltc} & {entropy} & {novelty} & {gkpi} & {delta_gkpi} \\\\"
                )

        latex.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table*}"
        ])
    else:
        latex = [
            "\\begin{table*}[t]",
            "\\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            "\\begin{tabular}{lcccccc}",
            "\\toprule",
            "\\textbf{Method / Architecture} & \\textbf{nDCG@10} $\\uparrow$ & \\textbf{Recall@10} $\\uparrow$ & \\textbf{RMSE-PC} $\\downarrow$ & \\textbf{MRMC} $\\downarrow$ & \\textbf{APLT@10} (\\%) & \\textbf{GKPI} $\\uparrow$ \\\\",
            "\\midrule"
        ]

        for row in table_rows:
            method = row.get("Method", "")
            ndcg = f"{row.get('nDCG@10', 0.0):.4f}"
            recall = f"{row.get('Recall@10', 0.0):.4f}"
            rmse_pc = f"{row.get('RMSE-PC', 0.0):.4f}"
            mrmc = f"{row.get('MRMC', 0.0):.4f}"
            aplt = f"{row.get('APLT@10', 0.0) * 100:.1f}\\%"
            gkpi = f"{row.get('GKPI', 0.0):.4f}"

            if "Ours" in method or "RRFN" in method:
                latex.append(f"\\textbf{{{method}}} & \\textbf{{{ndcg}}} & \\textbf{{{recall}}} & \\textbf{{{rmse_pc}}} & \\textbf{{{mrmc}}} & \\textbf{{{aplt}}} & \\textbf{{{gkpi}}} \\\\")
            else:
                latex.append(f"{method} & {ndcg} & {recall} & {rmse_pc} & {mrmc} & {aplt} & {gkpi} \\\\")

        latex.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table*}"
        ])

    return "\n".join(latex)

def generate_metric_comparison_summary(
    base_metrics: Dict[str, float],
    updated_metrics: Dict[str, float],
    clean_metrics: Optional[Dict[str, float]] = None,
    top_k: int = 10
) -> List[Dict[str, Any]]:
    """
    Computes a comprehensive head-to-head comparison across all base paper metrics.
    """
    metric_definitions = [
        # (Metric Key, Display Name, Category, Higher is Better, Format)
        (f"Recall@{top_k}", f"Recall@{top_k}", "Ranking Accuracy", True, "float"),
        (f"nDCG@{top_k}", f"nDCG@{top_k}", "Ranking Accuracy", True, "float"),
        ("RMSE-PC", "RMSE-PC (Calibration Error)", "Popularity Calibration", False, "float"),
        ("MRMC", "MRMC (Rank Miscalibration)", "Popularity Calibration", False, "float"),
        (f"APLT@{top_k}", f"APLT@{top_k} (Long-Tail %)", "Fairness & Discovery", True, "percent"),
        (f"LTC@{top_k}", f"LTC@{top_k} (Catalog Coverage)", "Fairness & Discovery", True, "percent"),
        ("Entropy", "Entropy (Exposure Spread)", "Fairness & Discovery", True, "float"),
        ("Novelty", "Novelty (Self-Information bits)", "Fairness & Discovery", True, "float"),
        ("GKPI", "GKPI (Holistic Harmonic Metric)", "Overall Performance", True, "float"),
        ("Delta-GKPI(%)", "ΔGKPI (Attack Degradation %)", "Adversarial Robustness", False, "percent_delta"),
        ("CSS", "CSS (Calibration Stability Score)", "Adversarial Robustness", False, "float"),
        ("Denoising-F1", "Denoising Detection F1", "Adversarial Robustness", True, "float"),
        ("Denoising-ROC-AUC", "Denoising Detection ROC-AUC", "Adversarial Robustness", True, "float")
    ]

    comparison_rows: List[Dict[str, Any]] = []

    for key, display_name, category, higher_is_better, fmt in metric_definitions:
        base_val = float(base_metrics.get(key, 0.0))
        updated_val = float(updated_metrics.get(key, 0.0))

        # Absolute diff: updated - base
        abs_diff = updated_val - base_val

        # Relative improvement (%)
        if higher_is_better:
            if base_val != 0:
                rel_impr = ((updated_val - base_val) / abs(base_val)) * 100.0
            else:
                rel_impr = 100.0 if updated_val > 0 else 0.0
            is_win = updated_val > base_val
        else:
            # Lower is better (e.g. RMSE-PC, MRMC, Delta-GKPI, CSS)
            if base_val != 0:
                rel_impr = ((base_val - updated_val) / abs(base_val)) * 100.0
            else:
                rel_impr = 100.0 if updated_val < base_val else 0.0
            is_win = updated_val < base_val

        clean_val = float(clean_metrics.get(key, 0.0)) if clean_metrics and key in clean_metrics else None

        comparison_rows.append({
            "Metric": display_name,
            "Category": category,
            "Clean Baseline": clean_val,
            "Base Model (Attacked)": base_val,
            "Updated Model (Ours)": updated_val,
            "Absolute Diff": abs_diff,
            "Relative Improvement (%)": rel_impr,
            "Higher is Better": higher_is_better,
            "Outcome": "🟢 WIN (+)" if is_win else ("⚪ PAR" if abs_diff == 0 else "🔴 LOSS (-)")
        })

    return comparison_rows


