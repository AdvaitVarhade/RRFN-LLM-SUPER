"""
test_m3_15epoch_and_ablation.py
--------------------------------
Comprehensive unit tests verifying Milestone 3 (Requirement R3):
  1. Full 15-epoch training loss history integrity in loss_history.json
  2. Loss convergence and strictly positive, finite values
  3. Canonical 6-variant ablation table schema and strict monotonicity
  4. DualModelTrainer early stopping patience configuration (None / non-positive support)
"""

import os
import json
import pytest
import numpy as np
import torch
from src.models.neumf import NeuMF
from src.trainer.dual_trainer import DualModelTrainer, build_data_loader
from experiments.run_ablations import verify_monotonicity

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results/run_15epoch_gpu"))

def test_loss_history_15_epoch_completeness():
    """Verify loss_history.json has exactly 15 epochs for all 4 tracked series."""
    history_file = os.path.join(RESULTS_DIR, "loss_history.json")
    assert os.path.exists(history_file), f"Missing {history_file}"

    with open(history_file, "r") as f:
        data = json.load(f)

    for key in ["M_pop_train_loss", "M_tail_train_loss", "M_pop_val_recall", "M_tail_val_recall"]:
        assert key in data, f"Missing key {key} in loss_history.json"
        assert len(data[key]) == 15, f"{key} length is {len(data[key])}, expected 15"

    # Verify all loss values are positive, finite numbers
    pop_loss = data["M_pop_train_loss"]
    tail_loss = data["M_tail_train_loss"]
    assert all(isinstance(v, (int, float)) and np.isfinite(v) and v > 0 for v in pop_loss)
    assert all(isinstance(v, (int, float)) and np.isfinite(v) and v > 0 for v in tail_loss)

    # Verify genuine convergence (initial loss > final loss)
    assert pop_loss[0] > pop_loss[-1], "M_pop initial loss must be higher than final loss"
    assert tail_loss[0] > tail_loss[-1], "M_tail initial loss must be higher than final loss"
    assert pop_loss[-1] < 1.05, f"M_pop converged loss expected < 1.05, got {pop_loss[-1]}"
    assert tail_loss[-1] < 1.20, f"M_tail converged loss expected < 1.20, got {tail_loss[-1]}"

def test_ablation_table_canonical_6_variants_and_monotonicity():
    """Verify ablation_table.json contains all 6 canonical variants and strict monotonicity."""
    ablation_file = os.path.join(RESULTS_DIR, "ablation_table.json")
    assert os.path.exists(ablation_file), f"Missing {ablation_file}"

    with open(ablation_file, "r") as f:
        rows = json.load(f)

    assert len(rows) == 6, f"Expected 6 canonical variants, found {len(rows)}"

    expected_variants = [
        "Full RRFN-LLM-SUPER (Ours)",
        "w/o RRFN (No Transition Matrix T)",
        "w/o LLM Auditing (beta = 0)",
        "w/o Bombing Detector (gamma = 0)",
        "w/o GCL Contrastive Loss (lambda_cl = 0)",
        "w/o SUPER Debiasing (Monolithic)"
    ]

    actual_variants = [r["Variant"] for r in rows]
    assert actual_variants == expected_variants, f"Variant mismatch: {actual_variants}"

    # Verify required metric keys
    required_keys = ["Variant", "Recall@10", "nDCG@10", "LTC@10", "Denoising_ROC", "Delta_Drop_Pct", "Primary_Impact_Description"]
    for r in rows:
        for k in required_keys:
            assert k in r, f"Row {r.get('Variant')} missing key {k}"
        for metric in ["Recall@10", "nDCG@10", "LTC@10", "Denoising_ROC"]:
            val = r[metric]
            assert isinstance(val, (int, float)) and np.isfinite(val), f"Metric {metric} not finite"
            assert 0.0 < val <= 1.0, f"Metric {metric} out of bounds: {val}"
        assert r["Delta_Drop_Pct"] <= 0.0, f"Delta_Drop_Pct must be non-positive, got {r['Delta_Drop_Pct']}"

    # Verify strict monotonicity invariants
    verify_monotonicity(rows)

def test_dual_trainer_early_stopping_disabled_runs_full_epochs():
    """Verify setting early_stopping_patience=None runs all requested epochs without premature break."""
    num_users = 10
    num_items = 15
    train_dict = {u: [(i, 3, 1000 + i) for i in range(4)] for u in range(num_users)}
    weights = {(u, i): 1.0 for u in train_dict for i, _, _ in train_dict[u]}
    val_dict = {u: (5, 4, 2000) for u in range(num_users)}
    allowed_items = set(range(num_items))

    def factory():
        return NeuMF(num_users, num_items, embedding_dim=8, mlp_layers=[16, 8])

    requested_epochs = 4
    trainer = DualModelTrainer(
        model_factory=factory,
        device="cpu",
        epochs=requested_epochs,
        early_stopping_patience=None # Disable early stopping
    )
    loader = build_data_loader(train_dict, weights, batch_size=8)

    model, _ = trainer.train_single_model(loader, val_dict, allowed_items)
    assert len(model.loss_history) == requested_epochs
    assert len(model.val_history) == requested_epochs
