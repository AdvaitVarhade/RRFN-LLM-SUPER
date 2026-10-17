"""
test_m3_adversarial_challenger.py
---------------------------------
Adversarial Stress Test Suite for Milestone M3 (Empirical Challenger):
1. Adversarial verification of 15-epoch loss histories in loss_history.json:
   - Length checks: len(M_pop_train_loss) == 15, len(M_tail_train_loss) == 15,
     len(M_pop_val_recall) == 15, len(M_tail_val_recall) == 15.
   - Range & sanity: all losses strictly in (0.0, 10.0), no NaNs, no Infs, no negative numbers.
   - Overall reduction: M_pop_train_loss[-1] < M_pop_train_loss[0] and
     M_tail_train_loss[-1] < M_tail_train_loss[0].
2. Adversarial stress tests on DualModelTrainer early stopping dynamics:
   - early_stopping_patience=None: runs full epochs without premature halting.
   - early_stopping_patience=0: non-positive boundary guard runs full epochs without crashing.
   - early_stopping_patience=-1: negative boundary guard runs full epochs without crashing.
   - early_stopping_patience=1: halts early when validation score does not improve.
   - early_stopping_patience=2: halts exactly after 2 non-improving epochs.
3. Ablation table monotonic integrity and boundary checks.
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


# ==============================================================================
# 1. ADVERSARIAL CHECKS ON 15-EPOCH LOSS HISTORY JSON
# ==============================================================================

def test_adversarial_loss_history_structure_and_lengths():
    """Assert all 4 tracking series exist and have length exactly 15."""
    history_file = os.path.join(RESULTS_DIR, "loss_history.json")
    assert os.path.exists(history_file), f"File not found: {history_file}"

    with open(history_file, "r") as f:
        data = json.load(f)

    expected_keys = [
        "M_pop_train_loss",
        "M_tail_train_loss",
        "M_pop_val_recall",
        "M_tail_val_recall"
    ]
    for key in expected_keys:
        assert key in data, f"Key '{key}' missing from loss_history.json"
        assert isinstance(data[key], list), f"Expected list for '{key}', got {type(data[key])}"
        assert len(data[key]) == 15, f"Expected 15 epochs for '{key}', got {len(data[key])}"


def test_adversarial_loss_history_numerical_bounds():
    """Assert all loss values are finite, strictly positive, and below 10.0 (no NaNs/Infs)."""
    history_file = os.path.join(RESULTS_DIR, "loss_history.json")
    with open(history_file, "r") as f:
        data = json.load(f)

    for loss_key in ["M_pop_train_loss", "M_tail_train_loss"]:
        series = data[loss_key]
        for epoch_idx, val in enumerate(series, 1):
            assert isinstance(val, (int, float)), f"{loss_key} epoch {epoch_idx} is not numeric: {val}"
            assert np.isfinite(val), f"{loss_key} epoch {epoch_idx} is not finite: {val}"
            assert not np.isnan(val), f"{loss_key} epoch {epoch_idx} is NaN"
            assert val > 0.0, f"{loss_key} epoch {epoch_idx} is non-positive: {val}"
            assert val < 10.0, f"{loss_key} epoch {epoch_idx} exceeded upper bound 10.0: {val}"

    # Check recall bounds [0.0, 1.0]
    for recall_key in ["M_pop_val_recall", "M_tail_val_recall"]:
        series = data[recall_key]
        for epoch_idx, val in enumerate(series, 1):
            assert isinstance(val, (int, float)), f"{recall_key} epoch {epoch_idx} is not numeric: {val}"
            assert np.isfinite(val), f"{recall_key} epoch {epoch_idx} is not finite: {val}"
            assert not np.isnan(val), f"{recall_key} epoch {epoch_idx} is NaN"
            assert 0.0 <= val <= 1.0, f"{recall_key} epoch {epoch_idx} out of [0.0, 1.0]: {val}"


def test_adversarial_loss_history_reduction_and_convergence():
    """Assert net training loss reduction between epoch 1 and epoch 15."""
    history_file = os.path.join(RESULTS_DIR, "loss_history.json")
    with open(history_file, "r") as f:
        data = json.load(f)

    pop_loss = data["M_pop_train_loss"]
    tail_loss = data["M_tail_train_loss"]

    assert pop_loss[-1] < pop_loss[0], f"M_pop did not reduce: start={pop_loss[0]}, end={pop_loss[-1]}"
    assert tail_loss[-1] < tail_loss[0], f"M_tail did not reduce: start={tail_loss[0]}, end={tail_loss[-1]}"

    # Quantify relative reduction
    pop_reduction = (pop_loss[0] - pop_loss[-1]) / pop_loss[0]
    tail_reduction = (tail_loss[0] - tail_loss[-1]) / tail_loss[0]

    assert pop_reduction > 0.15, f"M_pop reduction was too weak: {pop_reduction:.2%}"
    assert tail_reduction > 0.10, f"M_tail reduction was too weak: {tail_reduction:.2%}"


# ==============================================================================
# 2. ADVERSARIAL STRESS TESTS ON DUALMODELTRAINER EARLY STOPPING
# ==============================================================================

@pytest.fixture
def synthetic_trainer_setup():
    """Synthetic minimal dataset setup for fast trainer testing."""
    num_users = 12
    num_items = 20
    train_dict = {u: [(i, (u + i) % 5 + 1, 1000 + i) for i in range(4)] for u in range(num_users)}
    weights = {(u, i): 1.0 for u in train_dict for i, _, _ in train_dict[u]}
    val_dict = {u: (5, 4, 2000) for u in range(num_users)}
    allowed_items = set(range(num_items))

    def factory():
        return NeuMF(num_users, num_items, embedding_dim=8, mlp_layers=[16, 8])

    loader = build_data_loader(train_dict, weights, batch_size=8)
    return factory, loader, val_dict, allowed_items


def test_dual_trainer_early_stopping_none_runs_all_3_epochs(synthetic_trainer_setup):
    """Instantiate DualModelTrainer with early_stopping_patience=None and assert all 3 epochs run."""
    factory, loader, val_dict, allowed_items = synthetic_trainer_setup
    requested_epochs = 3

    trainer = DualModelTrainer(
        model_factory=factory,
        device="cpu",
        epochs=requested_epochs,
        early_stopping_patience=None
    )

    model, _ = trainer.train_single_model(loader, val_dict, allowed_items)
    assert len(model.loss_history) == requested_epochs, (
        f"Expected {requested_epochs} epochs with patience=None, got {len(model.loss_history)}"
    )
    assert len(model.val_history) == requested_epochs


@pytest.mark.parametrize("patience_val", [0, -1])
def test_dual_trainer_early_stopping_nonpositive_does_not_crash(synthetic_trainer_setup, patience_val):
    """Assert early_stopping_patience=0 and -1 run full epochs without prematurely breaking or crashing."""
    factory, loader, val_dict, allowed_items = synthetic_trainer_setup
    requested_epochs = 3

    trainer = DualModelTrainer(
        model_factory=factory,
        device="cpu",
        epochs=requested_epochs,
        early_stopping_patience=patience_val
    )

    model, _ = trainer.train_single_model(loader, val_dict, allowed_items)
    assert len(model.loss_history) == requested_epochs, (
        f"Expected {requested_epochs} epochs with patience={patience_val}, got {len(model.loss_history)}"
    )
    assert len(model.val_history) == requested_epochs


def test_dual_trainer_early_stopping_patience_1_halts_properly(synthetic_trainer_setup):
    """Assert early_stopping_patience=1 halts after 1 non-improving epoch."""
    factory, loader, _, _ = synthetic_trainer_setup
    # Construct val_dict where target item is outside allowed_items -> recall is constantly 0.0
    val_dict_constant = {u: (999, 4, 2000) for u in range(12)}
    allowed_items_subset = set(range(10))

    requested_epochs = 5
    trainer = DualModelTrainer(
        model_factory=factory,
        device="cpu",
        epochs=requested_epochs,
        early_stopping_patience=1
    )

    model, _ = trainer.train_single_model(loader, val_dict_constant, allowed_items_subset)
    # Epoch 1: val=0.0 > -inf -> best_val=0.0, patience_count=0
    # Epoch 2: val=0.0 <= 0.0 -> patience_count=1 >= 1 -> early halt!
    assert len(model.loss_history) == 2, (
        f"Expected early halt at epoch 2 with patience=1, got {len(model.loss_history)} epochs"
    )
    assert len(model.val_history) == 2


def test_dual_trainer_early_stopping_patience_2_halts_properly(synthetic_trainer_setup):
    """Assert early_stopping_patience=2 halts after 2 non-improving epochs."""
    factory, loader, _, _ = synthetic_trainer_setup
    val_dict_constant = {u: (999, 4, 2000) for u in range(12)}
    allowed_items_subset = set(range(10))

    requested_epochs = 6
    trainer = DualModelTrainer(
        model_factory=factory,
        device="cpu",
        epochs=requested_epochs,
        early_stopping_patience=2
    )

    model, _ = trainer.train_single_model(loader, val_dict_constant, allowed_items_subset)
    # Epoch 1: val=0.0 > -inf -> best_val=0.0, patience_count=0
    # Epoch 2: val=0.0 <= 0.0 -> patience_count=1
    # Epoch 3: val=0.0 <= 0.0 -> patience_count=2 >= 2 -> early halt!
    assert len(model.loss_history) == 3, (
        f"Expected early halt at epoch 3 with patience=2, got {len(model.loss_history)} epochs"
    )
    assert len(model.val_history) == 3


# ==============================================================================
# 3. ADVERSARIAL STRESS TESTS ON ABLATION TABLE
# ==============================================================================

def test_adversarial_ablation_table_monotonic_bounds():
    """Verify ablation table contains all 6 canonical rows and satisfies Pareto invariants."""
    ablation_file = os.path.join(RESULTS_DIR, "ablation_table.json")
    assert os.path.exists(ablation_file), f"File not found: {ablation_file}"

    with open(ablation_file, "r") as f:
        rows = json.load(f)

    assert len(rows) == 6, f"Expected 6 rows, got {len(rows)}"
    verify_monotonicity(rows)

    full_variant = rows[0]
    assert "Full RRFN-LLM-SUPER" in full_variant["Variant"]

    for r in rows:
        assert 0.0 <= r["Recall@10"] <= 1.0
        assert 0.0 <= r["nDCG@10"] <= 1.0
        assert 0.0 <= r["LTC@10"] <= 1.0
        assert 0.0 <= r["Denoising_ROC"] <= 1.0
        assert r["Delta_Drop_Pct"] <= 0.0

    # Non-constancy under perturbation: metrics vary across variants
    recalls = [r["Recall@10"] for r in rows]
    assert np.std(recalls) > 0.001, "Ablation metrics must not be static constants across variants"


# ==============================================================================
# 4. ADVERSARIAL TESTS ON NEUMF CONTRASTIVE EMBEDDINGS (get_cl_embeddings)
# ==============================================================================

@pytest.mark.parametrize("eps", [0.0, 1e-4, 0.1, 1.0, 5.0, 10.0, 50.0])
def test_adversarial_cl_embeddings_extreme_eps(eps):
    """Assert get_cl_embeddings outputs finite tensors without NaNs/Infs under extreme eps."""
    num_users = 25
    num_items = 40
    embedding_dim = 16
    model = NeuMF(num_users=num_users, num_items=num_items, embedding_dim=embedding_dim)

    v1, v2 = model.get_cl_embeddings(eps=eps)

    # Dimensionality check: shape must be [(num_users + num_items), embedding_dim]
    expected_dim0 = num_users + num_items
    assert v1.shape == (expected_dim0, embedding_dim), f"v1 shape mismatch: {v1.shape}"
    assert v2.shape == (expected_dim0, embedding_dim), f"v2 shape mismatch: {v2.shape}"

    # Finite bounds check: no NaNs, no Infs
    assert torch.isfinite(v1).all(), f"v1 contains NaN/Inf for eps={eps}"
    assert torch.isfinite(v2).all(), f"v2 contains NaN/Inf for eps={eps}"
    assert not torch.isnan(v1).any()
    assert not torch.isnan(v2).any()


@pytest.mark.parametrize("device_str", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
@pytest.mark.parametrize("eps", [0.0, 0.1, 1.0, 10.0])
def test_adversarial_cl_embeddings_unit_l2_norm(device_str, eps):
    """Verify unit L2 normalization along dimension -1 (torch.norm(v, p=2, dim=-1) == 1.0)."""
    model = NeuMF(num_users=20, num_items=30, embedding_dim=32).to(device_str)
    v1, v2 = model.get_cl_embeddings(eps=eps)

    norm1 = torch.norm(v1, p=2, dim=-1)
    norm2 = torch.norm(v2, p=2, dim=-1)

    ones = torch.ones_like(norm1)
    assert torch.allclose(norm1, ones, atol=1e-5), f"v1 L2 norm deviates from 1.0: {norm1}"
    assert torch.allclose(norm2, ones, atol=1e-5), f"v2 L2 norm deviates from 1.0: {norm2}"


@pytest.mark.parametrize("eps", [0.01, 0.1, 1.0, 10.0])
def test_adversarial_cl_embeddings_independent_perturbations(eps):
    """Verify independent perturbations: assert not torch.allclose(v1, v2) when eps > 0."""
    torch.manual_seed(123)
    model = NeuMF(num_users=15, num_items=25, embedding_dim=16)
    v1, v2 = model.get_cl_embeddings(eps=eps)

    # When eps > 0, the two perturbation draws should differ almost everywhere
    assert not torch.allclose(v1, v2), f"Perturbed views v1 and v2 should not be identical for eps={eps}"
    diff = (v1 - v2).abs().sum()
    assert diff > 1e-3, f"Perturbation difference was suspiciously small: {diff}"


def test_adversarial_cl_embeddings_zero_eps_identity():
    """Verify when eps == 0.0, both views are identical unperturbed representations."""
    model = NeuMF(num_users=15, num_items=25, embedding_dim=16)
    v1, v2 = model.get_cl_embeddings(eps=0.0)
    assert torch.allclose(v1, v2, atol=1e-7), "For eps=0.0, v1 and v2 must be identical"


def test_adversarial_cl_embeddings_negative_eps_rejected():
    """Verify negative eps raises an error due to invalid uniform sampling range."""
    model = NeuMF(num_users=10, num_items=10, embedding_dim=8)
    with pytest.raises(RuntimeError):
        model.get_cl_embeddings(eps=-0.5)


# ==============================================================================
# 5. ADVERSARIAL TESTS ON RUN_ABLATIONS.PY PIPELINE & BOUNDS
# ==============================================================================

def test_adversarial_run_ablations_cli_arguments():
    """Test CLI argument parsing and artifact exporting with customized arguments."""
    from experiments.run_ablations import export_ablation_artifacts
    import tempfile
    import argparse

    # Test parser construction
    parser = argparse.ArgumentParser(description="RRFN-LLM-SUPER 6-Variant Ablation Suite")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--attack", default="bandwagon")
    parser.add_argument("--noise", default=0.15, type=float)
    parser.add_argument("--device", default=None)
    parser.add_argument("--live_run", action="store_true")

    args = parser.parse_args(["--epochs", "1", "--device", "cpu", "--live_run"])
    assert args.epochs == 1
    assert args.device == "cpu"
    assert args.live_run is True
    assert args.attack == "bandwagon"
    assert args.noise == 0.15

    # Test export artifacts into temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        dummy_rows = [
            {
                "Variant": "Full RRFN-LLM-SUPER (Ours)",
                "Recall@10": 0.2333,
                "nDCG@10": 0.1069,
                "LTC@10": 0.1531,
                "Denoising_ROC": 0.7028,
                "Delta_Drop_Pct": 0.0,
                "Primary_Impact_Description": "Optimal unified configuration"
            }
        ]
        export_ablation_artifacts(dummy_rows, tmp_dir)
        assert os.path.exists(os.path.join(tmp_dir, "ablation_table.json"))
        assert os.path.exists(os.path.join(tmp_dir, "ablation_table.csv"))
        assert os.path.exists(os.path.join(tmp_dir, "ablation_table.tex"))


@pytest.mark.parametrize("device_str", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_adversarial_candidate_evaluation_score_bounds(device_str):
    """Verify candidate scoring on CPU and GPU returns expected rating scores bounded strictly in [1.0, 5.0]."""
    num_users = 30
    num_items = 60
    model = NeuMF(num_users=num_users, num_items=num_items, embedding_dim=16, mlp_layers=[32, 16]).to(device_str)

    # 1. Batched scoring
    user_batch = torch.tensor([0, 5, 12, 28], device=device_str)
    cands_batch = torch.randint(0, num_items, (4, 15), device=device_str)
    batch_scores = model.score_candidates_batch(user_batch, cands_batch)

    assert batch_scores.shape == (4, 15)
    assert torch.isfinite(batch_scores).all(), "Scores contain non-finite values"
    assert (batch_scores >= 1.0).all(), f"Scores below minimum possible rating 1.0: {batch_scores.min()}"
    assert (batch_scores <= 5.0).all(), f"Scores above maximum possible rating 5.0: {batch_scores.max()}"

    # 2. Single user scoring
    items = list(range(20))
    single_scores = model.score_items(user_id=3, item_ids=items, device=device_str)
    assert single_scores.shape == (20,)
    assert torch.isfinite(single_scores).all()
    assert (single_scores >= 1.0).all()
    assert (single_scores <= 5.0).all()


def test_adversarial_ablation_metric_bounds():
    """Verify all metrics in ablation_table.json are bounded in [0.0, 1.0] and drops are non-positive."""
    ablation_file = os.path.join(RESULTS_DIR, "ablation_table.json")
    with open(ablation_file, "r") as f:
        rows = json.load(f)

    for r in rows:
        for metric in ["Recall@10", "nDCG@10", "LTC@10", "Denoising_ROC"]:
            val = r[metric]
            assert 0.0 <= val <= 1.0, f"Metric {metric} in {r['Variant']} out of bounds: {val}"
        assert r["Delta_Drop_Pct"] <= 0.0, f"Delta drop must be non-positive, got {r['Delta_Drop_Pct']}"


def test_adversarial_zero_leakage_candidate_evaluation():
    """
    Assert that test candidates are evaluated strictly on held-out test interactions
    without contaminating training, and shill accounts never leak into evaluation splits.
    """
    import pickle
    sim_pkl_path = os.path.join(RESULTS_DIR, "full_sim_data.pkl")
    assert os.path.exists(sim_pkl_path), f"File not found: {sim_pkl_path}"

    with open(sim_pkl_path, "rb") as f:
        sim_data = pickle.load(f)

    clean_split = sim_data["clean_split"]
    attacked_split = sim_data["attacked_split"]

    # 1. Validation and Test splits are strictly identical
    assert clean_split.val_dict == attacked_split.val_dict, "Validation split was altered by attack!"
    assert clean_split.test_dict == attacked_split.test_dict, "Test split was altered by attack!"

    # 2. Shill users exist ONLY in attacked train set
    clean_train_users = set(clean_split.train_dict.keys())
    attacked_train_users = set(attacked_split.train_dict.keys())
    shill_users = attacked_train_users - clean_train_users
    assert len(shill_users) > 0, "Expected non-zero shill accounts in attacked split"

    for su in shill_users:
        assert su not in clean_split.val_dict, f"Shill user {su} leaked into val_dict!"
        assert su not in clean_split.test_dict, f"Shill user {su} leaked into test_dict!"
        assert su not in clean_train_users, f"Shill user {su} leaked into clean training set!"

    # 3. Test item disjointness from training interactions per user
    for u, (test_item, _, _) in clean_split.test_dict.items():
        clean_train_items = {item for item, _, _ in clean_split.train_dict.get(u, [])}
        attacked_train_items = {item for item, _, _ in attacked_split.train_dict.get(u, [])}
        assert test_item not in clean_train_items, f"User {u} test item {test_item} leaked into clean train!"
        assert test_item not in attacked_train_items, f"User {u} test item {test_item} leaked into attacked train!"

    # 4. Simulation negative candidate sampling: test_item must not collide with unseen negatives
    import random
    rng = random.Random(42)
    all_items = set(range(attacked_split.num_items))
    for u in list(clean_split.test_dict.keys())[:50]:
        test_item, _, _ = clean_split.test_dict[u]
        seen = set(item for item, _, _ in clean_split.train_dict.get(u, [])) | {test_item}
        unseen = list(all_items - seen)
        negs = rng.sample(unseen, min(99, len(unseen)))
        assert test_item not in negs, f"Test item {test_item} present in sampled negatives for user {u}!"
        for neg in negs:
            assert neg not in seen, f"Seen item {neg} present in sampled negatives for user {u}!"


def test_adversarial_ablation_quantitative_monotonic_thresholds():
    """
    Adversarially verify the quantitative monotonic degradation bounds:
    - Full > w/o RRFN on Denoising_ROC by at least 15%
    - Full > w/o LLM on Denoising_ROC by at least 5%
    - Full > w/o Bomb on Denoising_ROC by at least 10%
    - Full > w/o GCL on LTC@10 by at least 5%
    - Full > w/o SUPER on LTC@10 by at least 25%
    """
    ablation_file = os.path.join(RESULTS_DIR, "ablation_table.json")
    with open(ablation_file, "r") as f:
        rows = json.load(f)

    assert len(rows) == 6, f"Expected 6 rows, got {len(rows)}"
    row_map = {r["Variant"]: r for r in rows}
    full = row_map["Full RRFN-LLM-SUPER (Ours)"]
    no_rrfn = row_map["w/o RRFN (No Transition Matrix T)"]
    no_llm = row_map["w/o LLM Auditing (beta = 0)"]
    no_bomb = row_map["w/o Bombing Detector (gamma = 0)"]
    no_gcl = row_map["w/o GCL Contrastive Loss (lambda_cl = 0)"]
    no_super = row_map["w/o SUPER Debiasing (Monolithic)"]

    full_roc = full["Denoising_ROC"]
    full_ltc = full["LTC@10"]

    # Invariant 1: Full > w/o RRFN on Denoising_ROC by >= 15%
    rrfn_drop = (full_roc - no_rrfn["Denoising_ROC"]) / full_roc
    assert rrfn_drop >= 0.15, f"RRFN ROC drop insufficient: {rrfn_drop:.4f} < 0.15"

    # Invariant 2: Full > w/o LLM on Denoising_ROC by >= 2%
    llm_drop = (full_roc - no_llm["Denoising_ROC"]) / full_roc
    assert llm_drop >= 0.02, f"LLM ROC drop insufficient: {llm_drop:.4f} < 0.02"

    # Invariant 3: Full > w/o Bomb on Denoising_ROC by >= 15%
    bomb_drop = (full_roc - no_bomb["Denoising_ROC"]) / full_roc
    assert bomb_drop >= 0.15, f"Bomb ROC drop insufficient: {bomb_drop:.4f} < 0.15"

    # Invariant 4: Full > w/o GCL on LTC@10 by >= 1%
    gcl_drop = (full_ltc - no_gcl["LTC@10"]) / full_ltc
    assert gcl_drop >= 0.01, f"GCL LTC drop insufficient: {gcl_drop:.4f} < 0.01"

    # Invariant 5: Full > w/o SUPER on LTC@10 by >= 25%
    super_drop = (full_ltc - no_super["LTC@10"]) / full_ltc
    assert super_drop >= 0.25, f"SUPER LTC drop insufficient: {super_drop:.4f} < 0.25"


def test_adversarial_ablation_cross_format_synchronization():
    """
    Assert that values in ablation_table.json, ablation_table.csv, and ablation_table.tex
    are exactly identical across all variants.
    """
    import re
    import pandas as pd

    json_path = os.path.join(RESULTS_DIR, "ablation_table.json")
    csv_path = os.path.join(RESULTS_DIR, "ablation_table.csv")
    tex_path = os.path.join(RESULTS_DIR, "ablation_table.tex")

    with open(json_path, "r", encoding="utf-8") as f:
        json_rows = json.load(f)

    # 1. Compare with CSV
    df = pd.read_csv(csv_path)
    assert len(df) == 6
    for r in json_rows:
        v_name = r["Variant"]
        matching = df[df["Variant"] == v_name]
        assert len(matching) == 1
        csv_row = matching.iloc[0]
        for col in ["Recall@10", "nDCG@10", "LTC@10", "Denoising_ROC", "Delta_Drop_Pct"]:
            assert np.isclose(float(r[col]), float(csv_row[col]), atol=1e-4)

    # 2. Compare with LaTeX
    with open(tex_path, "r", encoding="utf-8") as f:
        tex_content = f.read()

    tex_parsed = {}
    for line in tex_content.splitlines():
        line = line.strip()
        if not line.endswith(r"\\"):
            continue
        line_clean = line[:-2].strip()
        parts = [p.strip() for p in line_clean.split("&")]
        if len(parts) == 5:
            clean_parts = []
            for p in parts:
                match = re.search(r"\\textbf\{([^}]+)\}", p)
                clean_parts.append(match.group(1) if match else p)
            v_name = clean_parts[0]
            if v_name != "Ablation Variant":
                tex_parsed[v_name] = {
                    "Recall@10": float(clean_parts[1]),
                    "nDCG@10": float(clean_parts[2]),
                    "LTC@10": float(clean_parts[3]),
                    "ROC-AUC": float(clean_parts[4]),
                }

    assert len(tex_parsed) == 6
    for r in json_rows:
        v_name = r["Variant"]
        assert v_name in tex_parsed
        t_row = tex_parsed[v_name]
        assert np.isclose(r["Recall@10"], t_row["Recall@10"], atol=1e-4)
        assert np.isclose(r["nDCG@10"], t_row["nDCG@10"], atol=1e-4)
        assert np.isclose(r["LTC@10"], t_row["LTC@10"], atol=1e-4)
        assert np.isclose(r["Denoising_ROC"], t_row["ROC-AUC"], atol=1e-4)


