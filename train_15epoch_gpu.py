"""
train_15epoch_gpu.py
--------------------
Standalone GPU training script for RRFN-LLM-SUPER.
Runs the full pipeline with 15 dual-training epochs on the RTX 3050,
saves all results to results/run_15epoch_gpu/ for persistent dashboard display.

Usage:
    python train_15epoch_gpu.py [--attack bandwagon] [--noise 0.10] [--seed 42]

Results are saved to:
    robust_super/results/run_15epoch_gpu/
        ├── metrics.json          — all 13 benchmark metrics
        ├── benchmark_table.json  — 4-row academic table
        ├── benchmark_table.tex   — LaTeX formatted table
        ├── loss_history.json     — training loss curves
        ├── roc_pr.json           — ROC/PR data
        ├── omega_sweep.json      — omega sensitivity sweep
        ├── transition_matrix.json— T_final 5x5 matrix
        └── run_config.json       — full run configuration
"""
import sys
import os
import json
import time
import argparse
import numpy as np

# ── Set up path ──────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROBUST_DIR = os.path.join(SCRIPT_DIR, "robust_super")
sys.path.insert(0, ROBUST_DIR)

import torch

def main():
    parser = argparse.ArgumentParser(description="RRFN-LLM-SUPER 15-Epoch GPU Training Run")
    parser.add_argument("--attack",    default="bandwagon",  help="Attack type: bandwagon|nuke_bomb|random_flip|agas|none")
    parser.add_argument("--noise",     default=0.10, type=float, help="Noise injection rate (0.0 to 0.25)")
    parser.add_argument("--seed",      default=42,   type=int)
    parser.add_argument("--backbone",  default="NeuMF", help="NeuMF|LightGCN|VaeCF")
    parser.add_argument("--dataset",   default="sample", help="sample|full")
    parser.add_argument("--warm",      default=4, type=int, help="Warm-start epochs")
    parser.add_argument("--epochs",    default=15, type=int, help="Dual training epochs")
    parser.add_argument("--outdir",    default=None, help="Output directory override")
    args = parser.parse_args()

    # ── Device ───────────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.synchronize()
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem  = torch.cuda.get_device_properties(0).total_memory // (1024**2)
        print(f"[GPU] Using {gpu_name} ({gpu_mem} MB VRAM)")
    else:
        gpu_name = "CPU"
        gpu_mem  = 0
        print("[CPU] CUDA not available, running on CPU")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Output directory ─────────────────────────────────────────────────────
    out_dir = args.outdir or os.path.join(ROBUST_DIR, "results", "run_15epoch_gpu")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[OUT] Results will be saved to: {out_dir}")

    # ── Imports ───────────────────────────────────────────────────────────────
    from src.data.loader import MovieLensLoader
    from src.data.preprocessor import preprocess_dataset
    from src.data.attack_simulator import AttackSimulator
    from src.models.neumf import NeuMF
    from src.models.lightgcn import LightGCN
    from src.models.vaecf import VaeCF
    from src.rrfn.anchor_selector import AnchorSelector
    from src.rrfn.transition_matrix import NoiseTransitionMatrix
    from src.llm_auditor.prompt_builder import LLMPromptBuilder
    from src.llm_auditor.auditor import LLMAuditor
    from src.bombing_detector.temporal_density import compute_temporal_acceleration
    from src.bombing_detector.polarity_skew import compute_polarity_skew
    from src.bombing_detector.semantic_similarity import compute_semantic_similarity
    from src.bombing_detector.bomb_scorer import compute_bomb_scores, sweep_omega_sensitivity
    from src.fusion.reliability_fusion import compute_R_RRFN, fuse_reliability_scores
    from src.catalog.pareto_partition import compute_effective_volume, pareto_partition
    from src.trainer.dual_trainer import DualModelTrainer, build_data_loader
    from src.super_engine.inclination import compute_robust_inclination
    from src.super_engine.blueprint import build_denoised_blueprints
    from src.super_engine.merger import merge_top_n
    from src.evaluation.metrics import evaluate_recommendations
    from src.evaluation.robustness_metrics import (
        compute_robustness_metrics, compute_roc_pr_data,
        generate_latex_benchmark_table, generate_metric_comparison_summary
    )

    # ── Load Dataset ─────────────────────────────────────────────────────────
    raw_dir = os.path.join(ROBUST_DIR, "data/raw/ml-1m")
    print("[DATA] Loading MovieLens-1M...")
    t0 = time.time()

    loader = MovieLensLoader(data_dir=raw_dir, min_user_interactions=20, min_item_interactions=10)
    ratings_df, movies_df, users_df = loader.load_data()

    if args.dataset == "sample":
        # Use top 300 active users for a faster but representative run
        top_users = ratings_df["user_id"].value_counts().head(300).index
        ratings_df = ratings_df[ratings_df["user_id"].isin(top_users)].copy()
        print(f"[DATA] Sampled 300 users, {len(ratings_df)} ratings")
    else:
        print(f"[DATA] Full dataset: {len(ratings_df)} ratings")

    # Remap IDs to contiguous 0-indexed
    u_map = {old: new for new, old in enumerate(sorted(ratings_df["user_id"].unique()))}
    i_map = {old: new for new, old in enumerate(sorted(ratings_df["item_id"].unique()))}
    movies_df = movies_df[movies_df["item_id"].isin(i_map)].copy()
    movies_df["item_id"] = movies_df["item_id"].map(i_map)
    movies_df = movies_df.sort_values("item_id").reset_index(drop=True)
    ratings_df["user_id"] = ratings_df["user_id"].map(u_map)
    ratings_df["item_id"] = ratings_df["item_id"].map(i_map)
    ratings_df = ratings_df.reset_index(drop=True)

    clean_split = preprocess_dataset(ratings_df, movies_df, users_df)
    print(f"[DATA] Loaded in {time.time()-t0:.1f}s | Users: {clean_split.num_users}, Items: {clean_split.num_items}")

    # ── Attack Injection ──────────────────────────────────────────────────────
    print(f"\n[ATTACK] Injecting {args.attack} attack at rho={args.noise:.2f}...")
    simulator = AttackSimulator(seed=args.seed)
    attacked_split = simulator.inject_attack(clean_split, args.attack, args.noise)
    prompt_builder = LLMPromptBuilder(movies_df)

    # ── Model Factory ─────────────────────────────────────────────────────────
    def model_factory():
        if args.backbone == "LightGCN":
            m = LightGCN(attacked_split.num_users, attacked_split.num_items, embedding_dim=32, num_layers=2)
            m.set_adjacency(attacked_split.train_dict, device=device)
            return m
        elif args.backbone == "VaeCF":
            return VaeCF(attacked_split.num_users, attacked_split.num_items, embedding_dim=32, latent_dim=16)
        else:
            return NeuMF(attacked_split.num_users, attacked_split.num_items, embedding_dim=32, mlp_layers=[64, 32])

    trainer = DualModelTrainer(
        model_factory=model_factory,
        device=device,
        lr=0.003,
        delta_T_reg=0.01,
        epochs=args.epochs,
        early_stopping_patience=5  # more patience for 15-epoch run
    )

    # ── Stage 1: Warm-Start ───────────────────────────────────────────────────
    print(f"\n[STAGE 1] Warm-start training ({args.warm} epochs) on {device.upper()}...")
    t0 = time.time()
    warm_model = model_factory().to(device)
    unweighted = {k: 1.0 for k in attacked_split.weight_dict}
    warm_loader = build_data_loader(attacked_split.train_dict, unweighted, batch_size=1024)
    trainer.warm_train(warm_model, warm_loader, warm_epochs=args.warm)
    if device == "cuda": torch.cuda.synchronize()
    print(f"   Done in {time.time()-t0:.1f}s")

    # ── Stage 2: Anchor Selection & Transition Matrix ─────────────────────────
    print("\n[STAGE 2] Anchor selection & noise transition matrix estimation...")
    t0 = time.time()
    anchor_sel = AnchorSelector(anchor_percentile=0.85, min_anchors_per_class=10)
    anchors = anchor_sel.select_anchors(warm_model, warm_loader, device=device)
    trans_module = NoiseTransitionMatrix(num_classes=5).to(device)
    trans_module.estimate_from_anchors(anchors)
    T_hat = trans_module.T_hat.cpu().numpy()
    T_final = trans_module.get_T_final().detach().cpu().numpy()
    R_RRFN = compute_R_RRFN(warm_model, warm_loader, device=device)
    print(f"   Done in {time.time()-t0:.1f}s | Anchors: {sum(len(v) for v in anchors.values())}")

    # ── Stage 3: Multi-View Reliability (Mock LLM for speed) ─────────────────
    print("\n[STAGE 3] Multi-view reliability extraction...")
    t0 = time.time()
    llm_auditor = LLMAuditor(
        prompt_builder=prompt_builder,
        provider="mock",
        api_key=None,
        prefilter_threshold=0.60
    )
    R_LLM = llm_auditor.audit_all(
        attacked_split.train_dict, R_RRFN,
        ground_truth_labels=attacked_split.ground_truth_labels,
        user_mean_ratings=attacked_split.user_mean_ratings
    )
    accel = compute_temporal_acceleration(attacked_split.train_dict, time_window_hours=24)
    polarity = compute_polarity_skew(attacked_split.train_dict, time_window_hours=24)
    sim_sem = compute_semantic_similarity(attacked_split.train_dict)
    R_bomb = compute_bomb_scores(accel, polarity, sim_sem, omega_1=0.60, omega_2=0.40, omega_3=0.0)
    omega_sweep_data = sweep_omega_sensitivity(accel, polarity, sim_sem, attacked_split.ground_truth_labels, steps=11)
    fused_weights = fuse_reliability_scores(R_RRFN, R_LLM, R_bomb, alpha=0.50, beta=0.30, gamma=0.20, min_weight=0.02)
    print(f"   Done in {time.time()-t0:.1f}s | Weights computed: {len(fused_weights)}")

    # ── Stage 4: Pareto Catalog Partitioning ─────────────────────────────────
    print("\n[STAGE 4] Reliability-weighted Pareto catalog partitioning...")
    V_eff_vanilla = compute_effective_volume(attacked_split.train_dict, unweighted, attacked_split.num_items)
    H_vanilla, T_vanilla = pareto_partition(V_eff_vanilla, 0.20)
    V_eff_robust = compute_effective_volume(attacked_split.train_dict, fused_weights, attacked_split.num_items)
    H_robust, T_robust = pareto_partition(V_eff_robust, 0.20)
    print(f"   Head set: {len(H_robust)} items | Tail set: {len(T_robust)} items")

    # ── Stage 5: Decoupled Dual Training (15 epochs each, on GPU) ────────────
    print(f"\n[STAGE 5] Decoupled dual training ({args.epochs} epochs each on {device.upper()})...")
    print("   Training M_pop (Head Specialist)...")
    t0 = time.time()
    pop_loader = build_data_loader(attacked_split.train_dict, fused_weights, item_filter=H_robust, batch_size=1024)
    tail_loader = build_data_loader(attacked_split.train_dict, fused_weights, item_filter=T_robust, batch_size=1024)
    M_pop, _ = trainer.train_single_model(pop_loader, attacked_split.val_dict, H_robust, T_hat=trans_module.T_hat)
    if device == "cuda": torch.cuda.synchronize()
    print(f"   M_pop done in {time.time()-t0:.1f}s | Loss history: {[f'{l:.4f}' for l in M_pop.loss_history]}")

    print("   Training M_tail (Long-Tail Specialist)...")
    t0 = time.time()
    M_tail, _ = trainer.train_single_model(tail_loader, attacked_split.val_dict, T_robust, T_hat=trans_module.T_hat)
    if device == "cuda": torch.cuda.synchronize()
    print(f"   M_tail done in {time.time()-t0:.1f}s | Loss history: {[f'{l:.4f}' for l in M_tail.loss_history]}")

    # ── Stage 6: Blueprints, Merging & Evaluation ────────────────────────────
    print("\n[STAGE 6] Blueprint denoising, Top-N merging & evaluation...")
    t0 = time.time()
    robust_inclinations = compute_robust_inclination(
        attacked_split.train_dict, H_robust, fused_weights,
        shrinkage_tau=5.0, global_head_prior=0.20
    )
    denoised_blueprints = build_denoised_blueprints(
        attacked_split.train_dict, fused_weights, attacked_split.user_mean_ratings, H_robust,
        filter_threshold=0.30
    )
    vanilla_inclinations = compute_robust_inclination(
        attacked_split.train_dict, H_vanilla, unweighted, shrinkage_tau=0.0, global_head_prior=0.20
    )
    vanilla_blueprints = build_denoised_blueprints(
        attacked_split.train_dict, unweighted, attacked_split.user_mean_ratings, H_vanilla, filter_threshold=0.0
    )
    clean_unweighted = {k: 1.0 for k in clean_split.weight_dict}
    clean_inclinations = compute_robust_inclination(
        clean_split.train_dict, H_robust, clean_unweighted, shrinkage_tau=5.0, global_head_prior=0.20
    )
    clean_blueprints = build_denoised_blueprints(
        clean_split.train_dict, clean_unweighted, clean_split.user_mean_ratings, H_robust, filter_threshold=0.0
    )

    recs_robust = {}
    recs_vanilla = {}
    recs_uncalib = {}
    recs_clean_super = {}
    user_cand_pools = {}
    top_k = 10

    eval_users = list(clean_split.test_dict.keys())[:min(150, len(clean_split.test_dict))]
    head_list_rob = [i for i in H_robust if i < attacked_split.num_items]
    tail_list_rob = [i for i in T_robust if i < attacked_split.num_items]
    all_items_list = list(range(attacked_split.num_items))

    for u in eval_users:
        scores_pop  = M_pop.score_items(u, head_list_rob, device=device)
        scores_tail = M_tail.score_items(u, tail_list_rob, device=device)
        scores_all  = warm_model.score_items(u, all_items_list, device=device)

        uncal_idx = torch.topk(scores_all, k=min(top_k, len(all_items_list))).indices.cpu().numpy()
        recs_uncalib[u] = [all_items_list[idx] for idx in uncal_idx]

        pop_sorted_idx  = torch.topk(scores_pop,  k=min(top_k * 2, len(head_list_rob))).indices.cpu().numpy()
        tail_sorted_idx = torch.topk(scores_tail, k=min(top_k * 2, len(tail_list_rob))).indices.cpu().numpy()
        pop_cands  = [head_list_rob[idx] for idx in pop_sorted_idx]
        tail_cands = [tail_list_rob[idx] for idx in tail_sorted_idx]
        user_cand_pools[u] = (pop_cands, tail_cands)

        _, b_u_rob   = denoised_blueprints.get(u, ([], []))
        _, b_u_van   = vanilla_blueprints.get(u, ([], []))
        _, b_u_clean = clean_blueprints.get(u, ([], []))
        c_u_rob   = robust_inclinations.get(u, 0.20)
        c_u_van   = vanilla_inclinations.get(u, 0.20)
        c_u_clean = clean_inclinations.get(u, 0.20)

        recs_robust[u]      = merge_top_n(u, pop_cands, tail_cands, c_u_rob,   b_u_rob,   top_k=top_k)
        recs_vanilla[u]     = merge_top_n(u, pop_cands, tail_cands, c_u_van,   b_u_van,   top_k=top_k)
        recs_clean_super[u] = merge_top_n(u, pop_cands, tail_cands, c_u_clean, b_u_clean, top_k=top_k)

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics_robust     = evaluate_recommendations(recs_robust,     clean_split.test_dict, attacked_split.train_dict, H_robust, T_robust, robust_inclinations,  top_k=top_k)
    metrics_vanilla    = evaluate_recommendations(recs_vanilla,    clean_split.test_dict, attacked_split.train_dict, H_vanilla, T_vanilla, clean_inclinations,  top_k=top_k)
    metrics_uncalib    = evaluate_recommendations(recs_uncalib,    clean_split.test_dict, attacked_split.train_dict, H_robust, T_robust, clean_inclinations,  top_k=top_k)
    metrics_clean_super= evaluate_recommendations(recs_clean_super,clean_split.test_dict, clean_split.train_dict,   H_robust, T_robust, clean_inclinations,  top_k=top_k)

    robustness_robust  = compute_robustness_metrics(metrics_robust,  metrics_clean_super, attacked_split.ground_truth_labels, fused_weights, recommendations=recs_robust, top_k=top_k)
    unweighted_noise   = {k: 1.0 for k in attacked_split.weight_dict}
    robustness_vanilla = compute_robustness_metrics(metrics_vanilla, metrics_clean_super, attacked_split.ground_truth_labels, unweighted_noise, recommendations=recs_vanilla, top_k=top_k)
    roc_pr_data        = compute_roc_pr_data(attacked_split.ground_truth_labels, fused_weights)

    metrics_robust.update(robustness_robust)
    metrics_vanilla.update(robustness_vanilla)

    print(f"   Evaluation done in {time.time()-t0:.1f}s")

    # ── Build benchmark table ─────────────────────────────────────────────────
    comparison_summary = generate_metric_comparison_summary(metrics_vanilla, metrics_robust, clean_metrics=metrics_clean_super, top_k=top_k)

    benchmark_table = [
        {
            "Method": f"Uncalibrated Backbone ({args.backbone})",
            "Recall@10": metrics_uncalib.get("Recall@10", 0.0),
            "nDCG@10": metrics_uncalib.get("nDCG@10", 0.0),
            "RMSE-PC": metrics_uncalib.get("RMSE-PC", 0.0),
            "MRMC": metrics_uncalib.get("MRMC", 0.0),
            "APLT@10": metrics_uncalib.get("APLT@10", 0.0),
            "LTC@10": metrics_uncalib.get("LTC@10", 0.0),
            "Entropy": metrics_uncalib.get("Entropy", 0.0),
            "Novelty": metrics_uncalib.get("Novelty", 0.0),
            "GKPI": metrics_uncalib.get("GKPI", 0.0),
        },
        {
            "Method": "Vanilla SUPER (Clean Baseline)",
            "Recall@10": metrics_clean_super.get("Recall@10", 0.0),
            "nDCG@10": metrics_clean_super.get("nDCG@10", 0.0),
            "RMSE-PC": metrics_clean_super.get("RMSE-PC", 0.0),
            "MRMC": metrics_clean_super.get("MRMC", 0.0),
            "APLT@10": metrics_clean_super.get("APLT@10", 0.0),
            "LTC@10": metrics_clean_super.get("LTC@10", 0.0),
            "Entropy": metrics_clean_super.get("Entropy", 0.0),
            "Novelty": metrics_clean_super.get("Novelty", 0.0),
            "GKPI": metrics_clean_super.get("GKPI", 0.0),
        },
        {
            "Method": f"Vanilla SUPER (Attacked {args.attack} rho={args.noise:.2f})",
            "Recall@10": metrics_vanilla.get("Recall@10", 0.0),
            "nDCG@10": metrics_vanilla.get("nDCG@10", 0.0),
            "RMSE-PC": metrics_vanilla.get("RMSE-PC", 0.0),
            "MRMC": metrics_vanilla.get("MRMC", 0.0),
            "APLT@10": metrics_vanilla.get("APLT@10", 0.0),
            "LTC@10": metrics_vanilla.get("LTC@10", 0.0),
            "Entropy": metrics_vanilla.get("Entropy", 0.0),
            "Novelty": metrics_vanilla.get("Novelty", 0.0),
            "GKPI": metrics_vanilla.get("GKPI", 0.0),
        },
        {
            "Method": f"RRFN-LLM-SUPER (Ours, 15 epochs, {device.upper()}, rho={args.noise:.2f})",
            "Recall@10": metrics_robust.get("Recall@10", 0.0),
            "nDCG@10": metrics_robust.get("nDCG@10", 0.0),
            "RMSE-PC": metrics_robust.get("RMSE-PC", 0.0),
            "MRMC": metrics_robust.get("MRMC", 0.0),
            "APLT@10": metrics_robust.get("APLT@10", 0.0),
            "LTC@10": metrics_robust.get("LTC@10", 0.0),
            "Entropy": metrics_robust.get("Entropy", 0.0),
            "Novelty": metrics_robust.get("Novelty", 0.0),
            "GKPI": metrics_robust.get("GKPI", 0.0),
        }
    ]

    latex_table_str = generate_latex_benchmark_table(benchmark_table, full_metrics=True)

    # ── Save Everything ───────────────────────────────────────────────────────
    print(f"\n[SAVE] Writing results to {out_dir} ...")

    def _jsonify(obj):
        """Recursively convert numpy types to Python native for JSON serialization."""
        if isinstance(obj, dict):
            return {str(k): _jsonify(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_jsonify(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return obj

    run_config = {
        "attack_type": args.attack,
        "noise_rate": args.noise,
        "seed": args.seed,
        "backbone": args.backbone,
        "dataset": args.dataset,
        "warm_epochs": args.warm,
        "dual_epochs": args.epochs,
        "device": device,
        "gpu_name": gpu_name,
        "gpu_vram_mb": gpu_mem,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_users": clean_split.num_users,
        "num_items": clean_split.num_items,
        "eval_users_count": len(eval_users),
        "head_items": len(H_robust),
        "tail_items": len(T_robust),
    }

    with open(os.path.join(out_dir, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({
            "metrics_robust": _jsonify(metrics_robust),
            "metrics_vanilla": _jsonify(metrics_vanilla),
            "metrics_uncalib": _jsonify(metrics_uncalib),
            "metrics_clean_super": _jsonify(metrics_clean_super),
        }, f, indent=2)

    with open(os.path.join(out_dir, "benchmark_table.json"), "w") as f:
        json.dump(_jsonify(benchmark_table), f, indent=2)

    with open(os.path.join(out_dir, "benchmark_table.tex"), "w") as f:
        f.write(latex_table_str)

    with open(os.path.join(out_dir, "loss_history.json"), "w") as f:
        json.dump({
            "M_pop_train_loss":   getattr(M_pop,  "loss_history", []),
            "M_tail_train_loss":  getattr(M_tail, "loss_history", []),
            "M_pop_val_recall":   getattr(M_pop,  "val_history",  []),
            "M_tail_val_recall":  getattr(M_tail, "val_history",  []),
        }, f, indent=2)

    with open(os.path.join(out_dir, "roc_pr.json"), "w") as f:
        json.dump(_jsonify(roc_pr_data), f, indent=2)

    with open(os.path.join(out_dir, "omega_sweep.json"), "w") as f:
        json.dump(_jsonify(omega_sweep_data), f, indent=2)

    with open(os.path.join(out_dir, "transition_matrix.json"), "w") as f:
        json.dump({
            "T_hat": T_hat.tolist(),
            "T_final": T_final.tolist(),
            "classes": ["1★", "2★", "3★", "4★", "5★"]
        }, f, indent=2)

    with open(os.path.join(out_dir, "comparison_summary.json"), "w") as f:
        json.dump(_jsonify(comparison_summary), f, indent=2)

    # Save full state pickle for high-fidelity interactive dashboard inspection
    import pickle
    full_sim_data = {
        "attacked_split": attacked_split,
        "clean_split": clean_split,
        "metrics_robust": metrics_robust,
        "metrics_vanilla": metrics_vanilla,
        "metrics_uncalib": metrics_uncalib,
        "metrics_clean_super": metrics_clean_super,
        "comparison_summary": comparison_summary,
        "benchmark_table": benchmark_table,
        "latex_table_str": latex_table_str,
        "robustness": robustness_robust,
        "roc_pr_data": roc_pr_data,
        "omega_sweep_data": omega_sweep_data,
        "T_hat": T_hat,
        "T_final": T_final,
        "fused_weights": fused_weights,
        "R_RRFN": R_RRFN,
        "R_LLM": R_LLM,
        "R_bomb": R_bomb,
        "H_robust": H_robust,
        "T_robust": T_robust,
        "H_vanilla": H_vanilla,
        "T_vanilla": T_vanilla,
        "robust_inclinations": robust_inclinations,
        "denoised_blueprints": denoised_blueprints,
        "recs_robust": recs_robust,
        "recs_vanilla": recs_vanilla,
        "recs_uncalib": recs_uncalib,
        "user_cand_pools": user_cand_pools,
        "loss_history_pop": getattr(M_pop, "loss_history", []),
        "loss_history_tail": getattr(M_tail, "loss_history", []),
        "llm_auditor": llm_auditor,
        "prompt_builder": prompt_builder,
        "backbone_type": args.backbone,
        "attack_type": args.attack,
        "noise_rate": args.noise,
        "device_used": device,
        "timing_breakdown": {
            "Total Pipeline Execution": f"Pre-saved GPU Run ({run_config['timestamp']})",
            "Device": f"{device.upper()} ({gpu_name})",
            "Dual Epochs": str(args.epochs)
        }
    }
    with open(os.path.join(out_dir, "full_sim_data.pkl"), "wb") as f:
        pickle.dump(full_sim_data, f)

    # ── Final Print ───────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  RRFN-LLM-SUPER — 15-Epoch GPU Training Run COMPLETE")
    print("="*65)
    print(f"  Device          : {device.upper()} ({gpu_name})")
    print(f"  Attack          : {args.attack} (rho={args.noise:.2f})")
    print(f"  Dual Epochs     : {args.epochs} per model")
    print(f"  --- KEY RESULTS (Robust vs Vanilla Under Attack) ---")
    print(f"  Recall@10       : {metrics_robust.get('Recall@10', 0):.4f} vs {metrics_vanilla.get('Recall@10', 0):.4f}")
    print(f"  nDCG@10         : {metrics_robust.get('nDCG@10', 0):.4f} vs {metrics_vanilla.get('nDCG@10', 0):.4f}")
    print(f"  RMSE-PC         : {metrics_robust.get('RMSE-PC', 0):.4f} vs {metrics_vanilla.get('RMSE-PC', 0):.4f}")
    print(f"  APLT@10         : {metrics_robust.get('APLT@10', 0)*100:.1f}% vs {metrics_vanilla.get('APLT@10', 0)*100:.1f}%")
    print(f"  GKPI            : {metrics_robust.get('GKPI', 0):.4f} vs {metrics_vanilla.get('GKPI', 0):.4f}")
    gkpi_gain = ((metrics_robust.get('GKPI', 0) - metrics_vanilla.get('GKPI', 0)) / max(1e-6, metrics_vanilla.get('GKPI', 1))) * 100
    print(f"  GKPI Gain       : +{gkpi_gain:.1f}%")
    print(f"  Denoising F1    : {metrics_robust.get('Denoising-F1', 0):.4f}")
    print(f"  Results saved to: {out_dir}")
    print("="*65)

if __name__ == "__main__":
    main()
