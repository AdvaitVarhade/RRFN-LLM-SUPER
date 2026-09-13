import pytest
from src.evaluation.metrics import evaluate_recommendations
from src.evaluation.robustness_metrics import compute_robustness_metrics

def test_evaluation_and_robustness_metrics():
    recommendations = {
        0: [10, 20, 30, 40, 50],
        1: [15, 25, 35, 45, 55]
    }
    test_dict = {
        0: (10, 5, 2000), # hit at rank 0
        1: (99, 4, 2000)  # miss
    }
    train_dict = {
        0: [(1, 5, 100), (2, 4, 101)],
        1: [(3, 4, 102)]
    }
    head_items = {10, 15}
    tail_items = {20, 25, 30, 35, 40, 45, 50, 55, 99}
    inclinations = {0: 0.20, 1: 0.20}

    metrics = evaluate_recommendations(
        recommendations, test_dict, train_dict,
        head_items, tail_items, inclinations, top_k=5
    )

    assert "Recall@5" in metrics
    assert "nDCG@5" in metrics
    assert "RMSE-PC" in metrics
    assert "GKPI" in metrics
    assert metrics["Recall@5"] == 0.50

    ground_truth = {(0, 1): 1, (0, 2): 1, (1, 3): 0}
    weights = {(0, 1): 0.9, (0, 2): 0.8, (1, 3): 0.2}
    robustness = compute_robustness_metrics(metrics, metrics, ground_truth, weights)

    assert "Delta-GKPI(%)" in robustness
    assert "CSS" in robustness
    assert "Denoising-F1" in robustness

    # Test ROC/PR data computation
    from src.evaluation.robustness_metrics import compute_roc_pr_data, generate_latex_benchmark_table
    roc_pr = compute_roc_pr_data(ground_truth, weights)
    assert "fpr" in roc_pr
    assert "tpr" in roc_pr
    assert "roc_auc" in roc_pr
    assert "precision" in roc_pr
    assert "recall" in roc_pr
    assert "avg_precision" in roc_pr
    assert 0.0 <= roc_pr["roc_auc"] <= 1.0

    # Test LaTeX table generation
    table_rows = [
        {"Method": "Vanilla SUPER", "nDCG@10": 0.35, "Recall@10": 0.45, "RMSE-PC": 0.12, "MRMC": 0.14, "APLT@10": 0.25, "GKPI": 0.52},
        {"Method": "RRFN-LLM-SUPER (Ours)", "nDCG@10": 0.38, "Recall@10": 0.49, "RMSE-PC": 0.08, "MRMC": 0.09, "APLT@10": 0.32, "GKPI": 0.63}
    ]
    latex_code = generate_latex_benchmark_table(table_rows)
    assert "\\begin{table*}" in latex_code
    assert "\\textbf{RRFN-LLM-SUPER (Ours)}" in latex_code
    assert "\\bottomrule" in latex_code

