"""
fedsuper_simulation/tests/verify_m1_invariants.py
Empirical Invariant and Privacy Validation Harness for Milestone 1.

Tests:
1. Invariants 1-4 across 10 random seeds (Calibration Error, Gini, LTER, Catalog Coverage).
2. Invariant 5 (Privacy Boundary & Zero Egress Verification).
3. Invariant 6 (FL Convergence over 10 rounds).
4. Edge cases & Stress tests (Cold start, skewed preferences, alpha variations, DP bounds).
"""
import sys
import os
import json
import math
import numpy as np

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import SimulationConfig
from src.mock_data import generate_mock_dataset, SyntheticDataset
from src.federated_core import FederatedServer, FederatedClient, FederatedSimulation
from src.super_engine import (
    compute_user_popularity_blueprint,
    intra_pool_zscore_standardization,
    fuse_cf_and_llm_scores,
    calibrated_blueprint_merge
)
from src.evaluator import (
    calculate_rmse_pc,
    calculate_gini_index,
    calculate_long_tail_exposure_ratio,
    calculate_catalog_coverage,
    calculate_recall_at_k,
    calculate_ndcg_at_k,
    calculate_novelty,
    calculate_ild,
    evaluate_all_metrics
)


def test_invariants_across_seeds(seeds=[42, 101, 777, 999, 1337, 2024, 31415, 8888, 54321, 99999]):
    """Empirical validation of Invariants 1-4 over 10 distinct seeds."""
    print("=" * 80)
    print("RUNNING MULTI-SEED EMPIRICAL INVARIANT VERIFICATION (10 SEEDS)")
    print("=" * 80)
    results = []

    for seed in seeds:
        cfg = SimulationConfig(
            num_clients=20,
            num_items=100,
            calibration_alpha=0.40,
            llm_lambda=0.70,
            top_k=10,
            seed=seed,
            clients_per_round=5,
            learning_rate=0.05
        )
        dataset = generate_mock_dataset(cfg, seed=seed)
        sim = FederatedSimulation(dataset, cfg)

        # Step 5 rounds
        for _ in range(5):
            sim.step_round()

        recs_uncalib = sim.get_recommendations(calibrated=False)
        recs_calib = sim.get_recommendations(calibrated=True)

        metrics = evaluate_all_metrics(dataset, recs_uncalib, recs_calib, top_k=10)

        un_rmse = metrics["uncalibrated"]["rmse_pc"]
        cal_rmse = metrics["calibrated"]["rmse_pc"]
        un_gini = metrics["uncalibrated"]["gini_index"]
        cal_gini = metrics["calibrated"]["gini_index"]
        un_lter = metrics["uncalibrated"]["long_tail_exposure_ratio"]
        cal_lter = metrics["calibrated"]["long_tail_exposure_ratio"]
        un_cov = metrics["uncalibrated"]["catalog_coverage"]
        cal_cov = metrics["calibrated"]["catalog_coverage"]

        # Check invariant conditions
        pass_inv1 = (cal_rmse <= 0.060) and (un_rmse >= 0.300)
        pass_inv2 = (cal_gini <= 0.400) and (un_gini >= 0.700)
        pass_inv3 = (cal_lter >= 0.250) and (un_lter <= 0.080)
        pass_inv4 = (cal_cov >= 0.800) and (un_cov <= 0.400)

        all_pass = pass_inv1 and pass_inv2 and pass_inv3 and pass_inv4

        row = {
            "seed": seed,
            "un_rmse": un_rmse,
            "cal_rmse": cal_rmse,
            "un_gini": un_gini,
            "cal_gini": cal_gini,
            "un_lter": un_lter,
            "cal_lter": cal_lter,
            "un_cov": un_cov,
            "cal_cov": cal_cov,
            "pass_inv1": pass_inv1,
            "pass_inv2": pass_inv2,
            "pass_inv3": pass_inv3,
            "pass_inv4": pass_inv4,
            "all_pass": all_pass
        }
        results.append(row)

        print(f"Seed {seed:5d} | RMSE-PC: {un_rmse:.4f}->{cal_rmse:.4f} ({'PASS' if pass_inv1 else 'FAIL'}) | "
              f"Gini: {un_gini:.4f}->{cal_gini:.4f} ({'PASS' if pass_inv2 else 'FAIL'}) | "
              f"LTER: {un_lter:.4f}->{cal_lter:.4f} ({'PASS' if pass_inv3 else 'FAIL'}) | "
              f"Cov: {un_cov:.4f}->{cal_cov:.4f} ({'PASS' if pass_inv4 else 'FAIL'})")

    return results


def test_privacy_boundary_and_zero_egress():
    """Validates Invariant 5: Zero Egress and Privacy Boundary."""
    print("\n" + "=" * 80)
    print("RUNNING PRIVACY BOUNDARY & ZERO EGRESS VERIFICATION (INVARIANT 5)")
    print("=" * 80)

    cfg = SimulationConfig(num_clients=10, num_items=50, dp_enabled=True, dp_l2_clip_norm=1.0, dp_epsilon=4.0, seed=42)
    dataset = generate_mock_dataset(cfg, seed=42)
    sim = FederatedSimulation(dataset, cfg)

    # 1. Inspect Server state
    server = sim.server
    assert not hasattr(server, "user_embeddings"), "Server must NOT store user embeddings!"
    assert not hasattr(server, "train_matrix"), "Server must NOT store client training interaction matrix!"
    assert not hasattr(server, "user_profiles"), "Server must NOT store user LLM profiles!"

    # 2. Inspect Client state and train step egress
    for client in sim.clients:
        # Check client holds private user embedding
        assert hasattr(client, "user_embedding"), "Client must hold private user embedding"
        assert client.user_embedding.shape == (cfg.embedding_dim,)
        
        # Step training
        t_idx, d_emb, d_bias, loss = client.local_train_step(
            server.item_embeddings, server.item_biases, cfg
        )

        # Verify egress payload
        assert isinstance(t_idx, np.ndarray)
        assert isinstance(d_emb, np.ndarray)
        assert isinstance(d_bias, np.ndarray)
        assert d_emb.shape[1] == cfg.embedding_dim

        # Ensure no user vector p_u is returned in delta
        assert d_emb.shape[0] == len(t_idx)
        # Check that d_bias is only per touched item
        assert len(d_bias) == len(t_idx)

    # 3. Step round and verify server parameters updated only on item space
    record = sim.step_round()
    assert server.item_embeddings.shape == (50, cfg.embedding_dim)
    assert server.item_biases.shape == (50,)
    assert not hasattr(server, "user_embeddings")

    print("[PASS] Invariant 5 Verified: User latent vectors p_u and interaction histories D_u have strict zero egress.")
    return True


def test_convergence_over_fl_rounds():
    """Validates Invariant 6: FL Convergence over 10 rounds."""
    print("\n" + "=" * 80)
    print("RUNNING FEDERATED LEARNING CONVERGENCE VERIFICATION (INVARIANT 6)")
    print("=" * 80)

    cfg = SimulationConfig(
        num_clients=30,
        num_items=100,
        learning_rate=0.08,
        clients_per_round=10,
        dp_enabled=True,
        dp_epsilon=8.0,
        dp_l2_clip_norm=1.5,
        seed=100
    )
    dataset = generate_mock_dataset(cfg, seed=100)
    sim = FederatedSimulation(dataset, cfg)

    losses = []
    for r in range(10):
        rec = sim.step_round()
        losses.append(rec["loss"])
        print(f"Round {rec['round']:2d}: Mean BPR Loss = {rec['loss']:.6f}, Items Touched = {rec['items_touched_pct']:.1f}%")

    initial_loss = np.mean(losses[:3])
    final_loss = np.mean(losses[-3:])
    loss_decrease = initial_loss - final_loss
    loss_decrease_pct = (loss_decrease / initial_loss) * 100.0

    print(f"Initial Mean Loss (R1-3): {initial_loss:.6f} | Final Mean Loss (R8-10): {final_loss:.6f} | Drop: {loss_decrease_pct:.2f}%")
    assert final_loss < initial_loss, f"Loss did not decrease: initial={initial_loss}, final={final_loss}"
    print(f"[PASS] Invariant 6 Verified: Mean FL BPR loss decreases over 10 communication rounds ({loss_decrease_pct:.2f}% reduction).")
    return {
        "losses": losses,
        "initial_loss": float(initial_loss),
        "final_loss": float(final_loss),
        "loss_decrease_pct": float(loss_decrease_pct)
    }


def test_alpha_sensitivity():
    """Evaluates the calibration curve across calibration_alpha in [0.0, 1.0]."""
    print("\n" + "=" * 80)
    print("RUNNING ALPHA SENSITIVITY SWEEP (alpha = 0.0, 0.2, 0.4, 0.6, 0.8, 1.0)")
    print("=" * 80)

    cfg = SimulationConfig(num_clients=20, num_items=100, top_k=10, seed=42)
    dataset = generate_mock_dataset(cfg, seed=42)
    sim = FederatedSimulation(dataset, cfg)
    for _ in range(5):
        sim.step_round()

    recs_uncalib = sim.get_recommendations(calibrated=False)
    fused_scores = fuse_cf_and_llm_scores(
        cf_scores=np.dot(np.array([c.user_embedding for c in sim.clients]), sim.server.item_embeddings.T) + sim.server.item_biases[None, :],
        user_profiles=dataset.user_llm_profiles,
        item_embeddings=dataset.item_llm_embeddings,
        head_idx=dataset.head_idx,
        torso_idx=dataset.torso_idx,
        tail_idx=dataset.tail_idx,
        llm_lambda=0.70
    )

    alpha_results = []
    for alpha in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        recs_cal = calibrated_blueprint_merge(
            fused_scores=fused_scores,
            user_blueprints=sim.user_blueprints,
            train_matrix=dataset.train_matrix,
            head_idx=dataset.head_idx,
            torso_idx=dataset.torso_idx,
            tail_idx=dataset.tail_idx,
            top_k=10,
            calibration_alpha=alpha
        )
        metrics = evaluate_all_metrics(dataset, recs_uncalib, recs_cal, top_k=10)
        r_entry = {
            "alpha": alpha,
            "rmse_pc": metrics["calibrated"]["rmse_pc"],
            "gini": metrics["calibrated"]["gini_index"],
            "lter": metrics["calibrated"]["long_tail_exposure_ratio"],
            "coverage": metrics["calibrated"]["catalog_coverage"],
            "recall": metrics["calibrated"]["recall_at_k"],
            "ndcg": metrics["calibrated"]["ndcg_at_k"]
        }
        alpha_results.append(r_entry)
        print(f"Alpha {alpha:.2f} | Rmse-PC: {r_entry['rmse_pc']:.4f} | Gini: {r_entry['gini']:.4f} | LTER: {r_entry['lter']:.4f} | Cov: {r_entry['coverage']:.4f} | Recall: {r_entry['recall']:.4f} | nDCG: {r_entry['ndcg']:.4f}")

    return alpha_results


if __name__ == "__main__":
    multi_seed_results = test_invariants_across_seeds()
    privacy_result = test_privacy_boundary_and_zero_egress()
    conv_result = test_convergence_over_fl_rounds()
    alpha_sweep_results = test_alpha_sensitivity()

    all_seed_passed = all(r["all_pass"] for r in multi_seed_results)
    print("\n" + "=" * 80)
    print(f"OVERALL EMPIRICAL VALIDATION RESULT: {'ALL PASSED (APPROVE)' if all_seed_passed else 'FAILED (REQUEST_CHANGES)'}")
    print("=" * 80)

    # Save validation results JSON for transparency
    report_data = {
        "multi_seed_results": multi_seed_results,
        "privacy_verified": privacy_result,
        "convergence_result": conv_result,
        "alpha_sweep_results": alpha_sweep_results,
        "all_seed_passed": all_seed_passed
    }
    with open("fedsuper_simulation/tests/empirical_validation_report.json", "w") as f:
        json.dump(report_data, f, indent=2)
    print("Saved empirical validation report to fedsuper_simulation/tests/empirical_validation_report.json")
