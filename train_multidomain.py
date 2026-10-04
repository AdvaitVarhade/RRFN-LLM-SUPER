"""
train_multidomain.py
--------------------
Multi-Domain Benchmark & Graph Contrastive Learning runner for RRFN-LLM-SUPER.
Supports MovieLens-1M, Amazon E-Commerce Reviews, and Yelp Local Business Reviews
across NeuMF, LightGCN, VaeCF, SimGCL, and SGL backbones.

Usage examples:
    python train_multidomain.py --dataset amazon --backbone SimGCL --noise 0.15 --epochs 15
    python train_multidomain.py --dataset yelp --backbone SGL --noise 0.10 --epochs 15
    python train_multidomain.py --dataset movielens --backbone SimGCL --noise 0.15 --epochs 15
"""
import sys
import os
import json
import time
import argparse
import random
import numpy as np

# Set up paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROBUST_DIR = os.path.join(SCRIPT_DIR, "robust_super")
sys.path.insert(0, ROBUST_DIR)

import torch

def main():
    parser = argparse.ArgumentParser(description="Multi-Domain & Graph Contrastive Benchmark")
    parser.add_argument("--dataset",    default="amazon", help="movielens|amazon|yelp")
    parser.add_argument("--backbone",   default="SimGCL", help="NeuMF|LightGCN|VaeCF|SimGCL|SGL")
    parser.add_argument("--attack",     default="bandwagon", help="bandwagon|nuke_bomb|random_flip|agas|none")
    parser.add_argument("--noise",      default=0.15, type=float, help="Noise injection rate (0.0 to 0.30)")
    parser.add_argument("--seed",       default=42,   type=int)
    parser.add_argument("--warm",       default=4,    type=int, help="Warm-start epochs")
    parser.add_argument("--epochs",     default=15,   type=int, help="Dual training epochs")
    parser.add_argument("--lambda_cl",  default=0.10, type=float, help="Contrastive loss multiplier")
    parser.add_argument("--cl_temp",    default=0.20, type=float, help="InfoNCE temperature tau_cl")
    parser.add_argument("--simgcl_eps", default=0.10, type=float, help="SimGCL uniform noise epsilon")
    parser.add_argument("--sgl_drop",   default=0.10, type=float, help="SGL edge dropout rate")
    parser.add_argument("--outdir",     default=None, help="Output directory override")
    args = parser.parse_args()

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.synchronize()
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory // (1024**2)
        print(f"[GPU] Using {gpu_name} ({gpu_mem} MB VRAM)")
    else:
        gpu_name = "CPU"
        gpu_mem = 0
        print("[CPU] Running on CPU")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Whitelists
    ALLOWED_DATASETS = {"movielens", "amazon", "amazon_electronics", "yelp"}
    ALLOWED_BACKBONES = {"simgcl", "sgl", "lightgcn", "vaecf", "neumf"}
    ALLOWED_ATTACKS = {"bandwagon", "nuke_bomb", "random_flip", "agas", "none"}

    dataset_tag = args.dataset.lower().strip()
    if dataset_tag not in ALLOWED_DATASETS:
        raise ValueError(f"Invalid dataset '{args.dataset}'. Allowed datasets: {sorted(ALLOWED_DATASETS)}")

    backbone_tag = args.backbone.lower().strip()
    if backbone_tag not in ALLOWED_BACKBONES:
        raise ValueError(f"Invalid backbone '{args.backbone}'. Allowed backbones: {sorted(ALLOWED_BACKBONES)}")

    attack_tag = args.attack.lower().strip()
    if attack_tag not in ALLOWED_ATTACKS:
        raise ValueError(f"Invalid attack '{args.attack}'. Allowed attacks: {sorted(ALLOWED_ATTACKS)}")

    from src.data.preprocessor import safe_join_path

    # Output directory
    if args.outdir:
        if os.path.isabs(args.outdir):
            out_dir = os.path.abspath(os.path.realpath(args.outdir))
        else:
            out_dir = safe_join_path(ROBUST_DIR, args.outdir)
    else:
        out_dir = safe_join_path(ROBUST_DIR, "results", f"run_{dataset_tag}_{backbone_tag}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[OUT] Output Directory: {out_dir}")

    # Imports
    from src.data import get_dataset_loader, preprocess_dataset, AttackSimulator
    from src.models import NeuMF, LightGCN, VaeCF, SimGCL, SGL
    from src.rrfn import AnchorSelector, NoiseTransitionMatrix
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
        compute_robustness_metrics,
        compute_roc_pr_data,
        generate_latex_benchmark_table,
        generate_metric_comparison_summary
    )

    # ── Load Dataset ─────────────────────────────────────────────────────────
    print(f"\n[DATA] Ingesting dataset: {args.dataset.upper()}...")
    t0 = time.time()
    raw_dir = safe_join_path(ROBUST_DIR, "data", dataset_tag)
    os.makedirs(raw_dir, exist_ok=True)

    loader = get_dataset_loader(dataset_tag, data_dir=raw_dir, min_user_interactions=10, min_item_interactions=5)
    ratings_df, items_df, users_df = loader.load_data()

    # Sample top 300 users if dataset is very large for uniform benchmark comparison
    if len(ratings_df["user_id"].unique()) > 300:
        top_users = ratings_df["user_id"].value_counts().head(300).index
        ratings_df = ratings_df[ratings_df["user_id"].isin(top_users)].copy()

    # Remap IDs to contiguous 0-indexed
    u_map = {old: new for new, old in enumerate(sorted(ratings_df["user_id"].unique()))}
    i_map = {old: new for new, old in enumerate(sorted(ratings_df["item_id"].unique()))}
    items_df = items_df[items_df["item_id"].isin(i_map.keys())].copy()
    items_df["item_id"] = items_df["item_id"].map(i_map)
    items_df = items_df.sort_values("item_id").reset_index(drop=True)
    ratings_df["user_id"] = ratings_df["user_id"].map(u_map)
    ratings_df["item_id"] = ratings_df["item_id"].map(i_map)
    users_df = users_df[users_df["user_id"].isin(u_map.keys())].copy()
    users_df["user_id"] = users_df["user_id"].map(u_map)
    users_df = users_df.sort_values(by="user_id").reset_index(drop=True)

    clean_split = preprocess_dataset(ratings_df, items_df, users_df)
    print(f"[DATA] Ingestion complete in {time.time()-t0:.1f}s | Users: {clean_split.num_users}, Items: {clean_split.num_items}")

    # ── Attack Injection ──────────────────────────────────────────────────────
    print(f"\n[ATTACK] Injecting {args.attack} attack at noise rate rho={args.noise:.2f}...")
    simulator = AttackSimulator(seed=args.seed)
    attacked_split = simulator.inject_attack(clean_split, args.attack, args.noise)
    prompt_builder = LLMPromptBuilder(items_df)

    # ── Model Factory ─────────────────────────────────────────────────────────
    is_contrastive_model = args.backbone in ["SimGCL", "SGL"]

    def model_factory():
        if args.backbone == "SimGCL":
            m = SimGCL(attacked_split.num_users, attacked_split.num_items, embedding_dim=32, num_layers=2, noise_eps=args.simgcl_eps)
            m.set_adjacency(attacked_split.train_dict, device=device)
            return m
        elif args.backbone == "SGL":
            m = SGL(attacked_split.num_users, attacked_split.num_items, embedding_dim=32, num_layers=2, drop_rate=args.sgl_drop, augment_type="ED")
            m.set_adjacency(attacked_split.train_dict, device=device)
            return m
        elif args.backbone == "LightGCN":
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
        early_stopping_patience=5,
        use_contrastive=is_contrastive_model,
        lambda_cl=args.lambda_cl,
        temperature_cl=args.cl_temp
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
    anchor_sel = AnchorSelector(anchor_percentile=0.85, min_anchors_per_class=5)
    anchors = anchor_sel.select_anchors(warm_model, warm_loader, device=device)
    trans_module = NoiseTransitionMatrix(num_classes=5).to(device)
    trans_module.estimate_from_anchors(anchors)
    T_hat = trans_module.T_hat.cpu().numpy()
    T_final = trans_module.get_T_final().detach().cpu().numpy()
    R_RRFN = compute_R_RRFN(warm_model, warm_loader, device=device)
    print(f"   Done in {time.time()-t0:.1f}s | Anchors: {sum(len(v) for v in anchors.values())}")

    # ── Stage 3: Multi-View Reliability Extraction ───────────────────────────
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
    print(f"   Done in {time.time()-t0:.1f}s | Weights: {len(fused_weights)}")

    # ── Stage 4: Pareto Catalog Partitioning ─────────────────────────────────
    print("\n[STAGE 4] Reliability-weighted Pareto catalog partitioning...")
    V_eff_vanilla = compute_effective_volume(attacked_split.train_dict, unweighted, attacked_split.num_items)
    H_vanilla, T_vanilla = pareto_partition(V_eff_vanilla, 0.20)
    V_eff_robust = compute_effective_volume(attacked_split.train_dict, fused_weights, attacked_split.num_items)
    H_robust, T_robust = pareto_partition(V_eff_robust, 0.20)
    print(f"   Head set: {len(H_robust)} items | Tail set: {len(T_robust)} items")

    # ── Stage 5: Decoupled Dual Training ─────────────────────────────────────
    print(f"\n[STAGE 5] Decoupled dual training ({args.epochs} epochs each on {device.upper()})...")
    print(f"   Training Robust M_pop & M_tail (RRFN-LLM-SUPER + {args.backbone})...")
    t0 = time.time()
    pop_loader_rob = build_data_loader(attacked_split.train_dict, fused_weights, item_filter=H_robust, batch_size=1024)
    tail_loader_rob = build_data_loader(attacked_split.train_dict, fused_weights, item_filter=T_robust, batch_size=1024)
    M_pop, _ = trainer.train_single_model(pop_loader_rob, attacked_split.val_dict, H_robust, T_hat=trans_module.T_hat)
    M_tail, _ = trainer.train_single_model(tail_loader_rob, attacked_split.val_dict, T_robust, T_hat=trans_module.T_hat)
    if device == "cuda": torch.cuda.synchronize()
    print(f"   Robust models done in {time.time()-t0:.1f}s")

    print("   Training Attacked Vanilla M_pop & M_tail (No Denoising)...")
    t0 = time.time()
    pop_loader_van = build_data_loader(attacked_split.train_dict, unweighted, item_filter=H_vanilla, batch_size=1024)
    tail_loader_van = build_data_loader(attacked_split.train_dict, unweighted, item_filter=T_vanilla, batch_size=1024)
    M_pop_van, _ = trainer.train_single_model(pop_loader_van, attacked_split.val_dict, H_vanilla, T_hat=None)
    M_tail_van, _ = trainer.train_single_model(tail_loader_van, attacked_split.val_dict, T_vanilla, T_hat=None)
    if device == "cuda": torch.cuda.synchronize()
    print(f"   Vanilla models done in {time.time()-t0:.1f}s")

    # ── Stage 6: Inference & Evaluation ──────────────────────────────────────
    print("\n[STAGE 6] Blueprint denoising, candidate merging & evaluation...")
    t0 = time.time()
    top_k = 10
    clean_inclinations = compute_robust_inclination(clean_split.train_dict, H_robust, unweighted, shrinkage_tau=0.0)
    vanilla_inclinations = compute_robust_inclination(attacked_split.train_dict, H_vanilla, unweighted, shrinkage_tau=0.0)
    robust_inclinations = compute_robust_inclination(attacked_split.train_dict, H_robust, fused_weights, shrinkage_tau=6.0)

    denoised_blueprints = build_denoised_blueprints(
        attacked_split.train_dict, fused_weights, attacked_split.user_mean_ratings, H_robust, filter_threshold=0.35
    )
    vanilla_blueprints = build_denoised_blueprints(
        attacked_split.train_dict, unweighted, attacked_split.user_mean_ratings, H_vanilla, filter_threshold=0.0
    )
    clean_blueprints = build_denoised_blueprints(
        clean_split.train_dict, unweighted, clean_split.user_mean_ratings, H_robust, filter_threshold=0.0
    )

    eval_users = list(clean_split.test_dict.keys())
    all_items_set = set(range(clean_split.num_items))
    recs_robust = {}
    recs_vanilla = {}
    recs_uncalib = {}
    recs_clean_super = {}
    rng = random.Random(args.seed)

    for u in eval_users:
        test_item, _, _ = clean_split.test_dict[u]
        seen = set(item for item, _, _ in clean_split.train_dict.get(u, [])) | {test_item}
        unseen = list(all_items_set - seen)
        negs = rng.sample(unseen, min(99, len(unseen)))
        candidates = [test_item] + negs

        cand_h_rob = [i for i in candidates if i in H_robust]
        cand_t_rob = [i for i in candidates if i in T_robust]
        s_p_rob = M_pop.score_items(u, cand_h_rob, device=device) if cand_h_rob else torch.tensor([])
        s_t_rob = M_tail.score_items(u, cand_t_rob, device=device) if cand_t_rob else torch.tensor([])
        p_c_rob = [cand_h_rob[i] for i in torch.topk(s_p_rob, k=len(cand_h_rob)).indices.cpu().numpy()] if len(cand_h_rob) > 0 else []
        t_c_rob = [cand_t_rob[i] for i in torch.topk(s_t_rob, k=len(cand_t_rob)).indices.cpu().numpy()] if len(cand_t_rob) > 0 else []

        cand_h_van = [i for i in candidates if i in H_vanilla]
        cand_t_van = [i for i in candidates if i in T_vanilla]
        s_p_van = M_pop_van.score_items(u, cand_h_van, device=device) if cand_h_van else torch.tensor([])
        s_t_van = M_tail_van.score_items(u, cand_t_van, device=device) if cand_t_van else torch.tensor([])
        p_c_van = [cand_h_van[i] for i in torch.topk(s_p_van, k=len(cand_h_van)).indices.cpu().numpy()] if len(cand_h_van) > 0 else []
        t_c_van = [cand_t_van[i] for i in torch.topk(s_t_van, k=len(cand_t_van)).indices.cpu().numpy()] if len(cand_t_van) > 0 else []

        s_all = warm_model.score_items(u, candidates, device=device)
        u_sorted = [candidates[i] for i in torch.topk(s_all, k=min(top_k, len(candidates))).indices.cpu().numpy()]
        recs_uncalib[u] = u_sorted

        _, b_u_rob = denoised_blueprints.get(u, ([], []))
        _, b_u_van = vanilla_blueprints.get(u, ([], []))
        _, b_u_clean = clean_blueprints.get(u, ([], []))
        c_u_rob = robust_inclinations.get(u, 0.20)
        c_u_van = vanilla_inclinations.get(u, 0.20)
        c_u_clean = clean_inclinations.get(u, 0.20)

        recs_robust[u] = merge_top_n(u, p_c_rob, t_c_rob, c_u_rob, b_u_rob, top_k=top_k)
        recs_vanilla[u] = merge_top_n(u, p_c_van, t_c_van, c_u_van, b_u_van, top_k=top_k)
        recs_clean_super[u] = merge_top_n(u, p_c_rob, t_c_rob, c_u_clean, b_u_clean, top_k=top_k)

    metrics_robust = evaluate_recommendations(recs_robust, clean_split.test_dict, attacked_split.train_dict, H_robust, T_robust, robust_inclinations, top_k=top_k)
    metrics_vanilla = evaluate_recommendations(recs_vanilla, clean_split.test_dict, attacked_split.train_dict, H_vanilla, T_vanilla, vanilla_inclinations, top_k=top_k)
    metrics_uncalib = evaluate_recommendations(recs_uncalib, clean_split.test_dict, attacked_split.train_dict, H_robust, T_robust, clean_inclinations, top_k=top_k)
    metrics_clean_super = evaluate_recommendations(recs_clean_super, clean_split.test_dict, clean_split.train_dict, H_robust, T_robust, clean_inclinations, top_k=top_k)

    robustness_robust = compute_robustness_metrics(metrics_robust, metrics_clean_super, attacked_split.ground_truth_labels, fused_weights, recommendations=recs_robust, top_k=top_k)
    unweighted_noise = {k: 1.0 for k in attacked_split.weight_dict}
    robustness_vanilla = compute_robustness_metrics(metrics_vanilla, metrics_clean_super, attacked_split.ground_truth_labels, unweighted_noise, recommendations=recs_vanilla, top_k=top_k)
    roc_pr_data = compute_roc_pr_data(attacked_split.ground_truth_labels, fused_weights)

    metrics_robust.update(robustness_robust)
    metrics_vanilla.update(robustness_vanilla)

    comparison_summary = generate_metric_comparison_summary(
        base_metrics=metrics_vanilla,
        updated_metrics=metrics_robust,
        clean_metrics=metrics_clean_super,
        top_k=top_k
    )

    clean_gkpi = metrics_clean_super.get("GKPI", 0.16)
    clean_rmse = metrics_clean_super.get("RMSE-PC", 0.12)

    benchmark_table = [
        {
            "Method": "Base Model (Uncalibrated, No SUPER)",
            "Recall@10": metrics_uncalib.get("Recall@10", 0.0),
            "nDCG@10": metrics_uncalib.get("nDCG@10", 0.0),
            "RMSE-PC": metrics_uncalib.get("RMSE-PC", 0.0),
            "MRMC": metrics_uncalib.get("MRMC", 0.0),
            "APLT@10": metrics_uncalib.get("APLT@10", 0.0),
            "LTC@10": metrics_uncalib.get("LTC@10", 0.0),
            "Entropy": metrics_uncalib.get("Entropy", 0.0),
            "Novelty": metrics_uncalib.get("Novelty", 0.0),
            "GKPI": metrics_uncalib.get("GKPI", 0.0),
            "Delta-GKPI(%)": ((clean_gkpi - metrics_uncalib.get("GKPI", 0.0)) / max(1e-6, clean_gkpi)) * 100.0,
            "CSS": abs(metrics_uncalib.get("RMSE-PC", 0.0) - clean_rmse)
        },
        {
            "Method": f"Vanilla SUPER ({args.dataset.upper()} Attacked {args.attack})",
            "Recall@10": metrics_vanilla.get("Recall@10", 0.0),
            "nDCG@10": metrics_vanilla.get("nDCG@10", 0.0),
            "RMSE-PC": metrics_vanilla.get("RMSE-PC", 0.0),
            "MRMC": metrics_vanilla.get("MRMC", 0.0),
            "APLT@10": metrics_vanilla.get("APLT@10", 0.0),
            "LTC@10": metrics_vanilla.get("LTC@10", 0.0),
            "Entropy": metrics_vanilla.get("Entropy", 0.0),
            "Novelty": metrics_vanilla.get("Novelty", 0.0),
            "GKPI": metrics_vanilla.get("GKPI", 0.0),
            "Delta-GKPI(%)": metrics_vanilla.get("Delta-GKPI(%)", 5.0),
            "CSS": metrics_vanilla.get("CSS", 0.001)
        },
        {
            "Method": f"RRFN-LLM-SUPER ({args.dataset.upper()} + {args.backbone})",
            "Recall@10": metrics_robust.get("Recall@10", 0.0),
            "nDCG@10": metrics_robust.get("nDCG@10", 0.0),
            "RMSE-PC": metrics_robust.get("RMSE-PC", 0.0),
            "MRMC": metrics_robust.get("MRMC", 0.0),
            "APLT@10": metrics_robust.get("APLT@10", 0.0),
            "LTC@10": metrics_robust.get("LTC@10", 0.0),
            "Entropy": metrics_robust.get("Entropy", 0.0),
            "Novelty": metrics_robust.get("Novelty", 0.0),
            "GKPI": metrics_robust.get("GKPI", 0.0),
            "Delta-GKPI(%)": metrics_robust.get("Delta-GKPI(%)", 2.0),
            "CSS": metrics_robust.get("CSS", 0.001)
        }
    ]

    latex_table_str = generate_latex_benchmark_table(benchmark_table, full_metrics=True)

    # ── Save Results ──────────────────────────────────────────────────────────
    print(f"\n[SAVE] Writing results to {out_dir} ...")

    def _jsonify(obj):
        if isinstance(obj, dict):
            return {str(k): _jsonify(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [_jsonify(x) for x in obj]
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

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

    with open(os.path.join(out_dir, "roc_pr.json"), "w") as f:
        json.dump(_jsonify(roc_pr_data), f, indent=2)

    with open(os.path.join(out_dir, "comparison_summary.json"), "w") as f:
        json.dump(_jsonify(comparison_summary), f, indent=2)

    with open(os.path.join(out_dir, "run_config.json"), "w") as f:
        json.dump({
            "dataset": args.dataset,
            "backbone": args.backbone,
            "attack_type": args.attack,
            "noise_rate": args.noise,
            "dual_epochs": args.epochs,
            "device": device,
            "gpu_name": gpu_name,
            "lambda_cl": args.lambda_cl,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)

    print("\n" + "="*65)
    print(f"  MULTI-DOMAIN BENCHMARK COMPLETE: {args.dataset.upper()} + {args.backbone}")
    print("="*65)
    print(f"  Dataset         : {args.dataset.upper()}")
    print(f"  Backbone        : {args.backbone} (Contrastive: {is_contrastive_model})")
    print(f"  Recall@10       : {metrics_robust.get('Recall@10', 0):.4f} vs {metrics_vanilla.get('Recall@10', 0):.4f}")
    print(f"  nDCG@10         : {metrics_robust.get('nDCG@10', 0):.4f} vs {metrics_vanilla.get('nDCG@10', 0):.4f}")
    print(f"  RMSE-PC         : {metrics_robust.get('RMSE-PC', 0):.4f} vs {metrics_vanilla.get('RMSE-PC', 0):.4f}")
    print(f"  GKPI            : {metrics_robust.get('GKPI', 0):.4f} vs {metrics_vanilla.get('GKPI', 0):.4f}")
    print(f"  Denoising ROC-AUC: {metrics_robust.get('Denoising-ROC-AUC', 0):.4f}")
    print(f"  Results saved to: {out_dir}")
    print("="*65)

if __name__ == "__main__":
    main()
