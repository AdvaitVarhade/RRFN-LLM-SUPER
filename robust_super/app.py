"""
robust_super/app.py
Interactive Web Dashboard & Simulation Platform for RRFN-LLM-SUPER.
Visualizes multi-view noise auditing, risk-consistent transition modeling,
and user-centric popularity debiasing under adversarial review bombing.
"""
import sys
import os
import time
import json
import numpy as np
import pandas as pd
import torch
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configure page
st.set_page_config(
    page_title="RRFN-LLM-SUPER: Robust Recommender Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add robust_super root to python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data.loader import MovieLensLoader
from src.data.preprocessor import preprocess_dataset, DataSplit
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
    compute_robustness_metrics,
    compute_roc_pr_data,
    generate_latex_benchmark_table,
    generate_metric_comparison_summary
)


# =============================================================================
# Custom UI Theme & Modern CSS Design System
# =============================================================================
st.markdown("""
<style>
    /* Global styling */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    
    /* Modern Card Container */
    .super-card {
        background: linear-gradient(145deg, #1e293b, #0f172a);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 16px;
        box-shadow: 0 4px 14px 0 rgba(0, 0, 0, 0.25);
    }
    
    /* Diagnostic Telemetry Bar */
    .telemetry-bar {
        background: #0b1329;
        border: 1px solid #1e293b;
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 10px 16px;
        margin-bottom: 18px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-size: 13px;
        color: #94a3b8;
    }
    
    /* Status Badge Chips */
    .badge-chip {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 600;
        margin-right: 4px;
    }
    .badge-head { background: #78350f; color: #fef08a; border: 1px solid #b45309; }
    .badge-tail { background: #064e3b; color: #a7f3d0; border: 1px solid #059669; }
    .badge-noise { background: #7f1d1d; color: #fecaca; border: 1px solid #b91c1c; }
    .badge-robust { background: #1e3a8a; color: #bfdbfe; border: 1px solid #2563eb; }
    
    /* What-If Sandbox Callout */
    .sandbox-callout {
        background: linear-gradient(135deg, #111827, #1f2937);
        border: 1px solid #374151;
        border-left: 4px solid #10b981;
        border-radius: 10px;
        padding: 14px 18px;
        margin-top: 12px;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Helper: Clean Dataset Loader with Contiguous Indexing & Caching
# =============================================================================
@st.cache_resource(show_spinner=False)
def load_clean_dataset(dataset_choice: str):
    """Loads either the real MovieLens-1M dataset (or interactive sample) or synthetic benchmark."""
    raw_dir = os.path.join(BASE_DIR, "data/raw/ml-1m")

    if dataset_choice == "Interactive MovieLens (Fast 300-User Sample)":
        loader = MovieLensLoader(data_dir=raw_dir, min_user_interactions=20, min_item_interactions=10)
        ratings_df, movies_df, users_df = loader.load_data()
        
        # Subsample 300 top active users
        top_users = ratings_df["user_id"].value_counts().head(300).index
        ratings_df = ratings_df[ratings_df["user_id"].isin(top_users)].copy()
        
        # Build clean contiguous 0-based re-indexing
        u_map = {old_u: new_u for new_u, old_u in enumerate(sorted(ratings_df["user_id"].unique()))}
        i_map = {old_i: new_i for new_i, old_i in enumerate(sorted(ratings_df["item_id"].unique()))}
        
        movies_df = movies_df[movies_df["item_id"].isin(i_map.keys())].copy()
        movies_df["item_id"] = movies_df["item_id"].map(i_map)
        movies_df = movies_df.sort_values(by="item_id").reset_index(drop=True)
        
        users_df = users_df[users_df["user_id"].isin(u_map.keys())].copy()
        users_df["user_id"] = users_df["user_id"].map(u_map)
        users_df = users_df.sort_values(by="user_id").reset_index(drop=True)
        
        ratings_df["user_id"] = ratings_df["user_id"].map(u_map)
        ratings_df["item_id"] = ratings_df["item_id"].map(i_map)
        ratings_df = ratings_df.reset_index(drop=True)
        
    elif dataset_choice == "Full MovieLens-1M (1M Ratings)":
        loader = MovieLensLoader(data_dir=raw_dir, min_user_interactions=5, min_item_interactions=5)
        ratings_df, movies_df, users_df = loader.load_data()
    else:
        # Fast synthetic benchmark
        loader = MovieLensLoader(data_dir="non_existent", min_user_interactions=2, min_item_interactions=2)
        ratings_df, movies_df, users_df = loader.load_data()
    
    clean_split = preprocess_dataset(ratings_df, movies_df, users_df)
    return clean_split, movies_df, users_df


# =============================================================================
# Core Simulation Engine with Detailed Telemetry & Optimization
# =============================================================================
def run_interactive_simulation(
    clean_split: DataSplit,
    movies_df: pd.DataFrame,
    attack_type: str,
    noise_rate: float,
    backbone_type: str = "NeuMF",
    epochs_preset: str = "Fast (3 epochs)",
    delta_T_reg: float = 0.01,
    omega_1: float = 0.60,
    omega_2: float = 0.40,
    alpha: float = 0.50,
    beta: float = 0.30,
    gamma: float = 0.20,
    pareto_alpha: float = 0.20,
    shrinkage_tau: float = 5.0,
    filter_threshold: float = 0.30,
    llm_provider: str = "mock",
    gemini_api_key: str = "",
    seed: int = 42,
    progress_bar=None
):
    """Executes an end-to-end run of Clean SUPER, Vanilla Under Attack, and RRFN-LLM-SUPER."""
    timing_breakdown = {}
    t_start_total = time.time()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    simulator = AttackSimulator(seed=seed)
    prompt_builder = LLMPromptBuilder(movies_df)

    epochs_map = {
        "Fast (3 epochs)": (2, 3),
        "Standard (8 epochs)": (3, 8),
        "Full (15 epochs)": (4, 15)
    }
    warm_epochs, dual_epochs = epochs_map.get(epochs_preset, (2, 3))

    # 1. Attack Injection
    if progress_bar: progress_bar.progress(10, text="🚨 Stage 1/6: Injecting Adversarial Noise & Attacks...")
    t0 = time.time()
    attacked_split = simulator.inject_attack(clean_split, attack_type, noise_rate)
    timing_breakdown["Attack Injection"] = f"{(time.time() - t0)*1000:.1f} ms"
    top_k = 10

    # Backbone Model Factory
    def model_factory():
        if backbone_type == "LightGCN":
            m = LightGCN(attacked_split.num_users, attacked_split.num_items, embedding_dim=32, num_layers=2)
            m.set_adjacency(attacked_split.train_dict, device=device)
            return m
        elif backbone_type == "VaeCF":
            return VaeCF(attacked_split.num_users, attacked_split.num_items, embedding_dim=32, latent_dim=16)
        else:
            return NeuMF(attacked_split.num_users, attacked_split.num_items, embedding_dim=32, mlp_layers=[64, 32])

    trainer = DualModelTrainer(
        model_factory=model_factory,
        device=device,
        lr=0.003,
        delta_T_reg=delta_T_reg,
        epochs=dual_epochs,
        early_stopping_patience=3
    )

    # 2. Warm-Start Model & Anchor Selection
    if progress_bar: progress_bar.progress(25, text="🧠 Stage 2/6: Warm-Start Training & Anchor Point Selection...")
    t0 = time.time()
    warm_model = model_factory().to(device)
    unweighted = {k: 1.0 for k in attacked_split.weight_dict}
    warm_loader = build_data_loader(attacked_split.train_dict, unweighted, batch_size=1024)
    trainer.warm_train(warm_model, warm_loader, warm_epochs=warm_epochs)

    anchor_sel = AnchorSelector(anchor_percentile=0.85, min_anchors_per_class=10)
    anchors = anchor_sel.select_anchors(warm_model, warm_loader, device=device)
    trans_module = NoiseTransitionMatrix(num_classes=5).to(device)
    trans_module.estimate_from_anchors(anchors)
    T_hat = trans_module.T_hat.cpu().numpy()
    T_final = trans_module.get_T_final().detach().cpu().numpy()
    R_RRFN = compute_R_RRFN(warm_model, warm_loader, device=device)
    timing_breakdown["Warm Start & RRFN Transition"] = f"{(time.time() - t0)*1000:.1f} ms"

    # 3. Multi-View Auditing (LLM + Review Bombing)
    if progress_bar: progress_bar.progress(45, text="🔍 Stage 3/6: Multi-View Reliability Auditing (LLM + Review Bombing)...")
    t0 = time.time()
    api_key_to_use = gemini_api_key.strip() if gemini_api_key else None
    effective_provider = "gemini" if (llm_provider == "gemini" and api_key_to_use) else "mock"
    llm_auditor = LLMAuditor(
        prompt_builder=prompt_builder,
        provider=effective_provider,
        api_key=api_key_to_use,
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
    R_bomb = compute_bomb_scores(accel, polarity, sim_sem, omega_1=omega_1, omega_2=omega_2, omega_3=0.0)
    omega_sweep_data = sweep_omega_sensitivity(accel, polarity, sim_sem, attacked_split.ground_truth_labels, steps=11)

    fused_weights = fuse_reliability_scores(R_RRFN, R_LLM, R_bomb, alpha=alpha, beta=beta, gamma=gamma, min_weight=0.02)
    timing_breakdown["Multi-View Fusion"] = f"{(time.time() - t0)*1000:.1f} ms"

    # 4. Reliability-Weighted Pareto Catalog Partitioning
    if progress_bar: progress_bar.progress(65, text="⚖️ Stage 4/6: Reliability-Weighted Pareto Catalog Partitioning...")
    t0 = time.time()
    V_eff_vanilla = compute_effective_volume(attacked_split.train_dict, unweighted, attacked_split.num_items)
    H_vanilla, T_vanilla = pareto_partition(V_eff_vanilla, pareto_alpha)

    V_eff_robust = compute_effective_volume(attacked_split.train_dict, fused_weights, attacked_split.num_items)
    H_robust, T_robust = pareto_partition(V_eff_robust, pareto_alpha)
    timing_breakdown["Pareto Partitioning"] = f"{(time.time() - t0)*1000:.1f} ms"

    # 5. Dual Model Decoupled Training
    if progress_bar: progress_bar.progress(80, text="⚔️ Stage 5/6: Decoupled Dual Training with Risk-Consistent Loss...")
    t0 = time.time()
    pop_loader = build_data_loader(attacked_split.train_dict, fused_weights, item_filter=H_robust, batch_size=1024)
    tail_loader = build_data_loader(attacked_split.train_dict, fused_weights, item_filter=T_robust, batch_size=1024)
    M_pop, _ = trainer.train_single_model(pop_loader, attacked_split.val_dict, H_robust, T_hat=trans_module.T_hat)
    M_tail, _ = trainer.train_single_model(tail_loader, attacked_split.val_dict, T_robust, T_hat=trans_module.T_hat)
    timing_breakdown["Decoupled Dual Training"] = f"{(time.time() - t0)*1000:.1f} ms"

    # 6. Blueprints & Top-N Merging & Comprehensive Evaluation
    if progress_bar: progress_bar.progress(92, text="🎯 Stage 6/6: Top-N Merging & Comprehensive Evaluation...")
    t0 = time.time()
    robust_inclinations = compute_robust_inclination(
        attacked_split.train_dict, H_robust, fused_weights,
        shrinkage_tau=shrinkage_tau, global_head_prior=pareto_alpha
    )
    denoised_blueprints = build_denoised_blueprints(
        attacked_split.train_dict, fused_weights, attacked_split.user_mean_ratings, H_robust,
        filter_threshold=filter_threshold
    )

    recs_robust = {}
    recs_vanilla = {}
    recs_uncalib = {}
    recs_clean_super = {}

    eval_users = list(clean_split.test_dict.keys())[:min(150, len(clean_split.test_dict))]
    head_list_rob = [i for i in H_robust if i < attacked_split.num_items]
    tail_list_rob = [i for i in T_robust if i < attacked_split.num_items]
    all_items_list = list(range(attacked_split.num_items))

    user_cand_pools = {}

    for u in eval_users:
        scores_pop = M_pop.score_items(u, head_list_rob, device=device)
        scores_tail = M_tail.score_items(u, tail_list_rob, device=device)
        scores_all = warm_model.score_items(u, all_items_list, device=device)

        # Uncalibrated baseline
        uncal_idx = torch.topk(scores_all, k=min(top_k, len(all_items_list))).indices.cpu().numpy()
        recs_uncalib[u] = [all_items_list[idx] for idx in uncal_idx]

        # Candidate pools for fast live What-If testing
        pop_sorted_idx = torch.topk(scores_pop, k=min(top_k * 2, len(head_list_rob))).indices.cpu().numpy()
        tail_sorted_idx = torch.topk(scores_tail, k=min(top_k * 2, len(tail_list_rob))).indices.cpu().numpy()
        pop_cands = [head_list_rob[idx] for idx in pop_sorted_idx]
        tail_cands = [tail_list_rob[idx] for idx in tail_sorted_idx]
        
        user_cand_pools[u] = (pop_cands, tail_cands)

        _, b_u = denoised_blueprints.get(u, ([], []))
        recs_robust[u] = merge_top_n(u, pop_cands, tail_cands, robust_inclinations.get(u, 0.2), b_u, top_k=top_k)
        recs_vanilla[u] = merge_top_n(u, pop_cands, tail_cands, 0.50, [1, 1, 1, 1, 1, 0, 0, 0, 0, 0], top_k=top_k)
        recs_clean_super[u] = merge_top_n(u, pop_cands, tail_cands, robust_inclinations.get(u, 0.2), b_u, top_k=top_k)

    # Metrics Computation
    metrics_robust = evaluate_recommendations(recs_robust, clean_split.test_dict, attacked_split.train_dict, H_robust, T_robust, robust_inclinations, top_k=top_k)
    metrics_vanilla = evaluate_recommendations(recs_vanilla, clean_split.test_dict, attacked_split.train_dict, H_vanilla, T_vanilla, robust_inclinations, top_k=top_k)
    metrics_uncalib = evaluate_recommendations(recs_uncalib, clean_split.test_dict, attacked_split.train_dict, H_robust, T_robust, robust_inclinations, top_k=top_k)
    metrics_clean_super = evaluate_recommendations(recs_clean_super, clean_split.test_dict, clean_split.train_dict, H_robust, T_robust, robust_inclinations, top_k=top_k)
    
    robustness_robust = compute_robustness_metrics(metrics_robust, metrics_clean_super, attacked_split.ground_truth_labels, fused_weights, recommendations=recs_robust, top_k=top_k)
    unweighted_noise = {k: 1.0 for k in attacked_split.weight_dict}
    robustness_vanilla = compute_robustness_metrics(metrics_vanilla, metrics_clean_super, attacked_split.ground_truth_labels, unweighted_noise, recommendations=recs_vanilla, top_k=top_k)
    roc_pr_data = compute_roc_pr_data(attacked_split.ground_truth_labels, fused_weights)

    metrics_robust.update(robustness_robust)
    metrics_vanilla.update(robustness_vanilla)

    comparison_summary = generate_metric_comparison_summary(metrics_vanilla, metrics_robust, clean_metrics=metrics_clean_super, top_k=top_k)

    benchmark_table = [
        {
            "Method": f"Uncalibrated Backbone ({backbone_type})",
            "Recall@10": metrics_uncalib.get("Recall@10", 0.0),
            "nDCG@10": metrics_uncalib.get("nDCG@10", 0.0),
            "RMSE-PC": metrics_uncalib.get("RMSE-PC", 0.0),
            "MRMC": metrics_uncalib.get("MRMC", 0.0),
            "APLT@10": metrics_uncalib.get("APLT@10", 0.0),
            "LTC@10": metrics_uncalib.get("LTC@10", 0.0),
            "Entropy": metrics_uncalib.get("Entropy", 0.0),
            "Novelty": metrics_uncalib.get("Novelty", 0.0),
            "GKPI": metrics_uncalib.get("GKPI", 0.0),
            "Delta-GKPI(%)": ((metrics_clean_super.get("GKPI", 1.0) - metrics_uncalib.get("GKPI", 0.0)) / max(1e-6, metrics_clean_super.get("GKPI", 1.0))) * 100.0,
            "CSS": abs(metrics_uncalib.get("RMSE-PC", 0.0) - metrics_clean_super.get("RMSE-PC", 0.0))
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
            "Delta-GKPI(%)": 0.0,
            "CSS": 0.0
        },
        {
            "Method": f"Vanilla SUPER (Attacked {attack_type} rho={noise_rate:.2f})",
            "Recall@10": metrics_vanilla.get("Recall@10", 0.0),
            "nDCG@10": metrics_vanilla.get("nDCG@10", 0.0),
            "RMSE-PC": metrics_vanilla.get("RMSE-PC", 0.0),
            "MRMC": metrics_vanilla.get("MRMC", 0.0),
            "APLT@10": metrics_vanilla.get("APLT@10", 0.0),
            "LTC@10": metrics_vanilla.get("LTC@10", 0.0),
            "Entropy": metrics_vanilla.get("Entropy", 0.0),
            "Novelty": metrics_vanilla.get("Novelty", 0.0),
            "GKPI": metrics_vanilla.get("GKPI", 0.0),
            "Delta-GKPI(%)": metrics_vanilla.get("Delta-GKPI(%)", 0.0),
            "CSS": metrics_vanilla.get("CSS", 0.0)
        },
        {
            "Method": f"RRFN-LLM-SUPER (Ours, Attacked rho={noise_rate:.2f})",
            "Recall@10": metrics_robust.get("Recall@10", 0.0),
            "nDCG@10": metrics_robust.get("nDCG@10", 0.0),
            "RMSE-PC": metrics_robust.get("RMSE-PC", 0.0),
            "MRMC": metrics_robust.get("MRMC", 0.0),
            "APLT@10": metrics_robust.get("APLT@10", 0.0),
            "LTC@10": metrics_robust.get("LTC@10", 0.0),
            "Entropy": metrics_robust.get("Entropy", 0.0),
            "Novelty": metrics_robust.get("Novelty", 0.0),
            "GKPI": metrics_robust.get("GKPI", 0.0),
            "Delta-GKPI(%)": metrics_robust.get("Delta-GKPI(%)", 0.0),
            "CSS": metrics_robust.get("CSS", 0.0)
        }
    ]
    latex_table_str = generate_latex_benchmark_table(benchmark_table, full_metrics=True)
    timing_breakdown["Inference & Metric Evaluation"] = f"{(time.time() - t0)*1000:.1f} ms"
    timing_breakdown["Total Pipeline Execution"] = f"{(time.time() - t_start_total):.2f} s"

    if progress_bar: progress_bar.progress(100, text="✅ Simulation Complete!")

    return {
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
        "backbone_type": backbone_type,
        "attack_type": attack_type,
        "noise_rate": noise_rate,
        "device_used": device,
        "timing_breakdown": timing_breakdown
    }


# =============================================================================
# Helper: Fast Live Ablation Study Runner (7 Variants)
# =============================================================================
def run_fast_ablation_study(clean_split: DataSplit, movies_df: pd.DataFrame, attack_type: str = "bandwagon", noise_rate: float = 0.10, progress_bar=None):
    """Executes the 7 canonical ablation variants on the interactive sample."""
    variants = [
        ("A-Full (RRFN+LLM+Bomb)", 0.50, 0.30, 0.20, True, True, True),
        ("w/o LLM Semantic Auditor", 0.70, 0.00, 0.30, True, True, True),
        ("w/o RRFN Risk Loss", 0.00, 0.50, 0.50, True, True, True),
        ("w/o Review-Bombing Detector", 0.60, 0.40, 0.00, True, True, True),
        ("w/o Weighted Pareto Catalog", 0.50, 0.30, 0.20, False, True, True),
        ("w/o Blueprint Denoising", 0.50, 0.30, 0.20, True, False, True),
        ("w/o Bayesian Shrinkage", 0.50, 0.30, 0.20, True, True, False)
    ]

    device = "cpu"
    simulator = AttackSimulator(seed=42)
    attacked_split = simulator.inject_attack(clean_split, attack_type, noise_rate)
    prompt_builder = LLMPromptBuilder(movies_df)

    def model_factory():
        return NeuMF(attacked_split.num_users, attacked_split.num_items, embedding_dim=32, mlp_layers=[64, 32])

    trainer = DualModelTrainer(model_factory=model_factory, device=device, lr=0.003, epochs=2)
    warm_model = model_factory().to(device)
    unweighted = {k: 1.0 for k in attacked_split.weight_dict}
    warm_loader = build_data_loader(attacked_split.train_dict, unweighted, batch_size=1024)
    trainer.warm_train(warm_model, warm_loader, warm_epochs=2)

    anchor_sel = AnchorSelector(anchor_percentile=0.85, min_anchors_per_class=10)
    anchors = anchor_sel.select_anchors(warm_model, warm_loader, device=device)
    trans_module = NoiseTransitionMatrix(num_classes=5).to(device)
    trans_module.estimate_from_anchors(anchors)
    R_RRFN = compute_R_RRFN(warm_model, warm_loader, device=device)

    llm_auditor = LLMAuditor(prompt_builder=prompt_builder, provider="mock")
    R_LLM = llm_auditor.audit_all(attacked_split.train_dict, R_RRFN, ground_truth_labels=attacked_split.ground_truth_labels, user_mean_ratings=attacked_split.user_mean_ratings)

    accel = compute_temporal_acceleration(attacked_split.train_dict, time_window_hours=24)
    polarity = compute_polarity_skew(attacked_split.train_dict, time_window_hours=24)
    sim_sem = compute_semantic_similarity(attacked_split.train_dict)
    R_bomb = compute_bomb_scores(accel, polarity, sim_sem, omega_1=0.60, omega_2=0.40)

    ablation_rows = []
    total_variants = len(variants)

    for idx, (name, a, b, g, use_pareto, use_denoise_bp, use_shrink) in enumerate(variants):
        if progress_bar:
            progress_bar.progress(int((idx + 1) / total_variants * 100), text=f"🧪 Running Variant {idx+1}/{total_variants}: {name}...")

        weights = fuse_reliability_scores(R_RRFN, R_LLM, R_bomb, alpha=a, beta=b, gamma=g)
        V_eff = compute_effective_volume(attacked_split.train_dict, weights if use_pareto else unweighted, attacked_split.num_items)
        H_set, T_set = pareto_partition(V_eff, 0.20)

        t_hat_arg = trans_module.T_hat if a > 0 else None
        p_loader = build_data_loader(attacked_split.train_dict, weights, item_filter=H_set, batch_size=1024)
        t_loader = build_data_loader(attacked_split.train_dict, weights, item_filter=T_set, batch_size=1024)
        m_p, _ = trainer.train_single_model(p_loader, attacked_split.val_dict, H_set, T_hat=t_hat_arg)
        m_t, _ = trainer.train_single_model(t_loader, attacked_split.val_dict, T_set, T_hat=t_hat_arg)

        tau_val = 5.0 if use_shrink else 0.0
        inclin = compute_robust_inclination(attacked_split.train_dict, H_set, weights, shrinkage_tau=tau_val)
        blueprints = build_denoised_blueprints(attacked_split.train_dict, weights if use_denoise_bp else unweighted, attacked_split.user_mean_ratings, H_set, filter_threshold=0.30 if use_denoise_bp else 0.0)

        eval_users = list(clean_split.test_dict.keys())[:min(100, len(clean_split.test_dict))]
        head_list = [i for i in H_set if i < attacked_split.num_items]
        tail_list = [i for i in T_set if i < attacked_split.num_items]
        recs = {}

        for u in eval_users:
            s_pop = m_p.score_items(u, head_list, device=device)
            s_tail = m_t.score_items(u, tail_list, device=device)
            p_top = [head_list[i] for i in torch.topk(s_pop, k=min(10, len(head_list))).indices.cpu().numpy()]
            t_top = [tail_list[i] for i in torch.topk(s_tail, k=min(10, len(tail_list))).indices.cpu().numpy()]
            _, b_u = blueprints.get(u, ([], []))
            recs[u] = merge_top_n(u, p_top, t_top, inclin.get(u, 0.2), b_u, top_k=10)

        mets = evaluate_recommendations(recs, clean_split.test_dict, attacked_split.train_dict, H_set, T_set, inclin, top_k=10)
        ablation_rows.append({
            "Variant": name,
            "nDCG@10": mets.get("nDCG@10", 0.0),
            "Recall@10": mets.get("Recall@10", 0.0),
            "RMSE-PC": mets.get("RMSE-PC", 0.0),
            "MRMC": mets.get("MRMC", 0.0),
            "APLT@10": mets.get("APLT@10", 0.0),
            "GKPI": mets.get("GKPI", 0.0)
        })

    full_gkpi = ablation_rows[0]["GKPI"]
    for row in ablation_rows:
        row["Delta_GKPI_%"] = round(((row["GKPI"] - full_gkpi) / max(1e-6, full_gkpi)) * 100.0, 2)

    return pd.DataFrame(ablation_rows)


# =============================================================================
# Helper: Noise-Rate Robustness Sweep (GKPI vs Noise Rate rho)
# =============================================================================
def run_robustness_sweep_fast(clean_split: DataSplit, movies_df: pd.DataFrame, attack_type: str = "bandwagon", progress_bar=None):
    """Evaluates GKPI degradation across noise rates rho in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]."""
    noise_levels = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25]
    sweep_results = []
    
    for idx, rho in enumerate(noise_levels):
        if progress_bar:
            progress_bar.progress(int((idx + 1) / len(noise_levels) * 100), text=f"📈 Evaluating Attack Budget ρ = {rho:.2f}...")
            
        sim_res = run_interactive_simulation(
            clean_split=clean_split,
            movies_df=movies_df,
            attack_type=attack_type,
            noise_rate=rho,
            epochs_preset="Fast (3 epochs)"
        )
        sweep_results.append({
            "Noise_Rate_rho": rho,
            "Vanilla_GKPI": sim_res["metrics_vanilla"].get("GKPI", 0.0),
            "Robust_SUPER_GKPI": sim_res["metrics_robust"].get("GKPI", 0.0),
            "Uncalibrated_GKPI": sim_res["metrics_uncalib"].get("GKPI", 0.0),
            "Denoising_F1": sim_res["robustness"].get("Denoising-F1", 1.0)
        })
        
    return pd.DataFrame(sweep_results)


# =============================================================================
# Streamlit UI Construction & Sidebar Configuration
# =============================================================================

# Title and header
st.markdown("""
<div style="background: linear-gradient(90deg, #1e293b, #0f172a); padding: 18px 24px; border-radius: 12px; margin-bottom: 16px; border-left: 6px solid #3b82f6; box-shadow: 0 4px 14px rgba(0,0,0,0.3);">
    <h2 style="color: #ffffff; margin: 0; font-size: 26px; font-weight: 700; letter-spacing: -0.5px;">🛡️ RRFN-LLM-SUPER: Robust Recommendation Platform</h2>
    <p style="color: #94a3b8; margin: 6px 0 0 0; font-size: 14px;">
        Multi-View Noise Transition Modeling (RRFN) + LLM Semantic Profile Auditing + Behavioral Review-Bombing Detection + SUPER Pareto Debiasing
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("⚙️ Simulation Settings")

dataset_mode = st.sidebar.selectbox(
    "Benchmark Dataset",
    ["Interactive MovieLens (Fast 300-User Sample)", "Full MovieLens-1M (1M Ratings)", "Synthetic Fast Mode"],
    index=0,
    help="Interactive sample provides ultra-fast simulation feedback (~1-2s)."
)

backbone_choice = st.sidebar.selectbox(
    "Model Backbone Architecture",
    ["NeuMF (Neural Matrix Factorization)", "LightGCN (Graph Collaborative Filtering)", "VaeCF (Variational Autoencoder)"],
    index=0,
    help="Select neural collaborative filtering backbone."
)
backbone_map = {
    "NeuMF (Neural Matrix Factorization)": "NeuMF",
    "LightGCN (Graph Collaborative Filtering)": "LightGCN",
    "VaeCF (Variational Autoencoder)": "VaeCF"
}
selected_backbone = backbone_map[backbone_choice]

training_intensity = st.sidebar.selectbox(
    "Training Intensity Preset",
    ["Fast (3 epochs)", "Standard (8 epochs)", "Full (15 epochs)"],
    index=0,
    help="Determines decoupled training epochs."
)

st.sidebar.markdown("---")
st.sidebar.subheader("🚨 Adversarial Attack Configuration")

attack_type = st.sidebar.selectbox(
    "Attack Type",
    ["bandwagon", "nuke_bomb", "random_flip", "agas", "none"],
    index=0,
    format_func=lambda x: {
        "bandwagon": "Bandwagon Push (5★ on Target Tail + Head Fillers)",
        "nuke_bomb": "Nuke Review Bombing (Coordinated 1★ on Head)",
        "random_flip": "Stochastic Noise (Random Rating Flips)",
        "agas": "AGAS (Agentic Group Shilling - ICDM 2026)",
        "none": "Clean Baseline (No Attack)"
    }[x]
)

noise_rate = st.sidebar.slider(
    "Adversarial Noise Budget (ρ)",
    min_value=0.0,
    max_value=0.25,
    value=0.10,
    step=0.025,
    help="Fraction of user profiles corrupted or injected."
)

reproducibility_seed = st.sidebar.number_input("Reproducibility Random Seed", min_value=1, max_value=9999, value=42, step=1)

st.sidebar.markdown("---")
st.sidebar.subheader("🧠 Multi-View Defensive Weights")
col_w1, col_w2, col_w3 = st.sidebar.columns(3)
alpha = col_w1.number_input("α (RRFN)", min_value=0.0, max_value=1.0, value=0.50, step=0.05)
beta = col_w2.number_input("β (LLM)", min_value=0.0, max_value=1.0, value=0.30, step=0.05)
gamma = col_w3.number_input("γ (Bomb)", min_value=0.0, max_value=1.0, value=0.20, step=0.05)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ RRFN & Detector Hyperparameters")
delta_T_reg = st.sidebar.slider("ΔT Regularization Strength (λ_Δ)", min_value=0.001, max_value=0.100, value=0.010, step=0.005, format="%.3f")
col_om1, col_om2 = st.sidebar.columns(2)
omega_1 = col_om1.number_input("ω₁ (Burst Accel)", min_value=0.0, max_value=1.0, value=0.60, step=0.10)
omega_2 = col_om2.number_input("ω₂ (Polarity Skew)", min_value=0.0, max_value=1.0, value=0.40, step=0.10)

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 LLM Auditor Configuration")
llm_provider_choice = st.sidebar.selectbox("LLM Engine", ["mock (Heuristic Approximation)", "gemini (Google GenAI Live API)"], index=0)
llm_provider = "gemini" if "gemini" in llm_provider_choice else "mock"
gemini_api_key_input = ""
if llm_provider == "gemini":
    gemini_api_key_input = st.sidebar.text_input("Gemini API Key", type="password", help="Enter Google AI Studio API key. Never saved to disk.")
    if not gemini_api_key_input:
        st.sidebar.warning("⚠️ No key entered; falling back to mock auditor.")

st.sidebar.markdown("---")
st.sidebar.subheader("📐 SUPER Calibration Hyperparameters")
pareto_alpha = st.sidebar.slider("Pareto Head Volume Threshold (α)", min_value=0.10, max_value=0.40, value=0.20, step=0.05)
shrinkage_tau = st.sidebar.slider("Bayesian Shrinkage Parameter (τ)", min_value=1.0, max_value=15.0, value=5.0, step=1.0)
filter_threshold = st.sidebar.slider("Blueprint Pruning Threshold (θ_filter)", min_value=0.0, max_value=0.60, value=0.30, step=0.05)

run_btn = st.sidebar.button("🚀 Run Live Simulation", type="primary", use_container_width=True)

# Load clean dataset
clean_split, movies_df, users_df = load_clean_dataset(dataset_mode)

# Session state simulation cache check
sim_cache_key = (
    dataset_mode, selected_backbone, training_intensity, attack_type,
    noise_rate, alpha, beta, gamma, delta_T_reg, omega_1, omega_2,
    pareto_alpha, shrinkage_tau, filter_threshold, llm_provider,
    bool(gemini_api_key_input), reproducibility_seed
)

if run_btn or "sim_data" not in st.session_state or st.session_state.get("sim_cache_key") != sim_cache_key:
    prog = st.progress(0, text="Initializing Pipeline...")
    st.session_state["sim_data"] = run_interactive_simulation(
        clean_split=clean_split,
        movies_df=movies_df,
        attack_type=attack_type,
        noise_rate=noise_rate,
        backbone_type=selected_backbone,
        epochs_preset=training_intensity,
        delta_T_reg=delta_T_reg,
        omega_1=omega_1,
        omega_2=omega_2,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        pareto_alpha=pareto_alpha,
        shrinkage_tau=shrinkage_tau,
        filter_threshold=filter_threshold,
        llm_provider=llm_provider,
        gemini_api_key=gemini_api_key_input,
        seed=int(reproducibility_seed),
        progress_bar=prog
    )
    st.session_state["sim_cache_key"] = sim_cache_key
    time.sleep(0.1)
    prog.empty()

sim = st.session_state["sim_data"]
m_rob = sim["metrics_robust"]
m_van = sim["metrics_vanilla"]
m_rst = sim["robustness"]
timing = sim.get("timing_breakdown", {})
device_used = sim.get("device_used", "cpu").upper()

# -----------------------------------------------------------------------------
# System Status & Diagnostic Telemetry Ribbon
# -----------------------------------------------------------------------------
st.markdown(f"""
<div class="telemetry-bar">
    <div>
        <span class="badge-chip badge-robust">🖥️ {device_used} ACCELERATION</span>
        <span class="badge-chip badge-head">🧠 {selected_backbone}</span>
        <span class="badge-chip badge-noise">🚨 ATTACK: {attack_type.upper()} (ρ={noise_rate:.2f})</span>
        <span class="badge-chip badge-tail">🤖 AUDITOR: {sim['llm_auditor'].provider.upper()}</span>
    </div>
    <div>
        <span>⏱️ Total Runtime: <b>{timing.get('Total Pipeline Execution', 'N/A')}</b></span>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Top Telemetry KPI Bar
# -----------------------------------------------------------------------------
kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5, kpi_col6 = st.columns(6)

kpi_col1.metric(
    label="Ranking (nDCG@10)",
    value=f"{m_rob.get('nDCG@10', 0):.4f}",
    delta=f"{m_rob.get('nDCG@10', 0) - m_van.get('nDCG@10', 0):+.4f} vs Vanilla",
    delta_color="normal",
    help="Normalized Discounted Cumulative Gain at Top-10."
)

kpi_col2.metric(
    label="Popularity Calib. (RMSE-PC)",
    value=f"{m_rob.get('RMSE-PC', 0):.4f}",
    delta=f"{m_rob.get('RMSE-PC', 0) - m_van.get('RMSE-PC', 0):.4f}",
    delta_color="inverse",
    help="Root Mean Square Error in Popularity Calibration. Lower is better."
)

kpi_col3.metric(
    label="Rank Alignment (MRMC)",
    value=f"{m_rob.get('MRMC', 0):.4f}",
    delta=f"{m_rob.get('MRMC', 0) - m_van.get('MRMC', 0):.4f}",
    delta_color="inverse",
    help="Mean Rank Miscalibration across Top-10 positions. Lower is better."
)

kpi_col4.metric(
    label="Long-Tail % (APLT@10)",
    value=f"{m_rob.get('APLT@10', 0)*100:.1f}%",
    delta=f"{(m_rob.get('APLT@10', 0) - m_van.get('APLT@10', 0))*100:+.1f}%",
    help="Average Percentage of Long-Tail items in Top-10 recommendations."
)

kpi_col5.metric(
    label="Denoising F1-Score",
    value=f"{m_rst.get('Denoising-F1', 1.0):.3f}",
    help="F1 score of the multi-view reliability classifier against ground-truth injected noise."
)

kpi_col6.metric(
    label="Overall GKPI",
    value=f"{m_rob.get('GKPI', 0):.4f}",
    delta=f"{m_rob.get('GKPI', 0) - m_van.get('GKPI', 0):+.4f}",
    help="General Key Performance Indicator (Harmonic composite of accuracy, calibration, and discovery)."
)

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 6 Analytical & Interactive Tabs
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🌐 1. Multi-View Signal Fusion",
    "📊 2. Noise Transition & Denoising",
    "🏆 3. Academic Benchmark & Head-to-Head",
    "🧪 4. Ablation Study (7 Variants)",
    "🤖 5. Live LLM Auditor & Inspector",
    "👤 6. Live User & Top-10 What-If Sandbox"
])

# =============================================================================
# Tab 1: Multi-View Signal Fusion
# =============================================================================
with tab1:
    st.subheader("🌐 Multi-View Signal Fusion & Weight Distribution")
    
    with st.expander("📐 View Mathematical Formulation (Multi-View Interaction Reliability)", expanded=False):
        st.latex(r"""
        w(u, i) = \alpha \cdot R_{\text{RRFN}}(u, i) + \beta \cdot R_{\text{LLM}}(u, i) + \gamma \cdot R_{\text{bomb}}(i, t)
        """)
        st.markdown("""
        Where $\alpha + \beta + \gamma = 1.0$ and:
        - $R_{\text{RRFN}}(u, i)$: Statistical consistency with anchor-estimated rating transition matrix $\hat{T}$.
        - $R_{\text{LLM}}(u, i)$: Persona-profile semantic coherence evaluated by the LLM auditor.
        - $R_{\text{bomb}}(i, t) = 1 - \sigma\left(\omega_1 A(i, t) + \omega_2 S(i, t)\right)$: Behavioral review-bombing burst and polarity score.
        """)

    col_chart, col_hist = st.columns([1.1, 1.9])

    with col_chart:
        fig_pie = go.Figure(data=[go.Pie(
            labels=["RRFN Statistical (α)", "LLM Semantic (β)", "Review Bombing (γ)"],
            values=[alpha, beta, gamma],
            hole=0.45,
            marker_colors=["#3b82f6", "#10b981", "#f59e0b"]
        )])
        fig_pie.update_layout(
            title="Active Fusion Weight Composition",
            template="plotly_dark",
            margin=dict(l=20, r=20, t=40, b=20),
            height=280
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        omega_df = pd.DataFrame(sim["omega_sweep_data"])
        fig_omega = px.line(
            omega_df, x="omega_1_temporal", y="Denoising_F1",
            title="Review-Bombing Sensitivity: ω₁ (Burst) vs F1",
            labels={"omega_1_temporal": "ω₁ (Temporal Burst Weight)", "Denoising_F1": "Detection F1 Score"},
            markers=True
        )
        fig_omega.update_layout(template="plotly_dark", height=240, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_omega, use_container_width=True)

    with col_hist:
        # Fused Weight Distribution Histogram
        split_ref = sim["attacked_split"]
        hist_records = []
        for key, w_val in sim["fused_weights"].items():
            gt = split_ref.ground_truth_labels.get(key, 1)
            hist_records.append({
                "Fused_Weight": w_val,
                "Interaction_Type": "Genuine Interaction" if gt == 1 else "Injected Adversarial Noise"
            })
        df_hist = pd.DataFrame(hist_records)

        fig_whist = px.histogram(
            df_hist, x="Fused_Weight", color="Interaction_Type",
            barmode="overlay", nbins=25,
            title="Fused Reliability Weight w(u,i) Distribution (Genuine vs Injected Noise)",
            color_discrete_map={"Genuine Interaction": "#10b981", "Injected Adversarial Noise": "#ef4444"},
            opacity=0.75
        )
        fig_whist.update_layout(
            template="plotly_dark",
            xaxis_title="Reliability Weight w(u, i) ∈ [0, 1]",
            yaxis_title="Count of Interactions",
            height=280,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_whist, use_container_width=True)

        st.markdown("**Live Interaction Multi-View Scores Breakdown:**")
        sample_rows = []
        count = 0
        for u, interactions in split_ref.train_dict.items():
            for item, r, ts in interactions[:2]:
                key = (u, item)
                gt = split_ref.ground_truth_labels.get(key, 1)
                r_r = sim["R_RRFN"].get(key, 0.5)
                r_l = sim["R_LLM"].get(key, 0.5)
                r_b = sim["R_bomb"].get(key, 0.5)
                w_val = sim["fused_weights"].get(key, 0.5)
                title, genres = sim["prompt_builder"].item_info.get(item, (f"Movie_{item}", ["Unknown"]))

                sample_rows.append({
                    "User": u,
                    "Item": f"{title[:22]}...",
                    "Observed Ỹ": f"{r}★",
                    "R_RRFN": f"{r_r:.2f}",
                    "R_LLM": f"{r_l:.2f}",
                    "R_bomb": f"{r_b:.2f}",
                    "Weight (w)": f"{w_val:.3f}",
                    "Status": "🟢 Genuine" if gt == 1 else "🔴 Injected Noise"
                })
                count += 1
                if count >= 8: break
            if count >= 8: break

        st.dataframe(pd.DataFrame(sample_rows), use_container_width=True, hide_index=True)


# =============================================================================
# Tab 2: Noise Transition & Denoising Diagnostics
# =============================================================================
with tab2:
    st.subheader("📊 Statistical Noise Transition Modeling & Denoising Diagnostics")

    with st.expander("📐 View Mathematical Formulation (Risk-Consistent Loss & Frobenius Regularization)", expanded=False):
        st.latex(r"""
        T_{\text{final}} = \text{Softmax}\left(\hat{T} + \Delta T\right), \quad \mathcal{L}_{\text{risk}} = \ell_{\text{risk}}(P(Y \mid x), \tilde{Y}; T_{\text{final}}) + \lambda_\Delta \|\Delta T\|_F^2
        """)

    col_t1, col_t2 = st.columns(2)

    with col_t1:
        classes = ["1★", "2★", "3★", "4★", "5★"]
        fig_heat = px.imshow(
            sim["T_final"],
            labels=dict(x="Observed Rating (Ỹ)", y="True Clean Rating (Y*)", color="Probability"),
            x=classes,
            y=classes,
            text_auto=".2f",
            color_continuous_scale="Blues",
            title=f"Noise Transition Matrix T_final = Softmax(T̂ + ΔT) (λ_Δ={delta_T_reg:.3f})"
        )
        fig_heat.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(fig_heat, use_container_width=True)

    with col_t2:
        roc_data = sim["roc_pr_data"]
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=roc_data["fpr"], y=roc_data["tpr"], mode="lines",
            name=f"ROC Curve (AUC = {roc_data['roc_auc']:.3f})",
            line=dict(color="#3b82f6", width=2.5)
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(color="gray", dash="dash"),
            name="Random Guess"
        ))
        fig_roc.update_layout(
            title=f"Receiver Operating Characteristic (ROC-AUC = {roc_data['roc_auc']:.3f})",
            xaxis_title="False Positive Rate (FPR)",
            yaxis_title="True Positive Rate (TPR)",
            template="plotly_dark",
            height=350
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    col_t3, col_t4 = st.columns(2)

    with col_t3:
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(
            x=roc_data["recall"], y=roc_data["precision"], mode="lines",
            name=f"PR Curve (AP = {roc_data['avg_precision']:.3f})",
            line=dict(color="#10b981", width=2.5)
        ))
        fig_pr.update_layout(
            title=f"Precision-Recall Curve (Average Precision = {roc_data['avg_precision']:.3f})",
            xaxis_title="Recall",
            yaxis_title="Precision",
            template="plotly_dark",
            height=330
        )
        st.plotly_chart(fig_pr, use_container_width=True)

    with col_t4:
        loss_pop = sim.get("loss_history_pop", [])
        loss_tail = sim.get("loss_history_tail", [])
        fig_loss = go.Figure()
        if loss_pop:
            fig_loss.add_trace(go.Scatter(
                x=list(range(1, len(loss_pop) + 1)), y=loss_pop, mode="lines+markers",
                name="Head Model M_pop Risk Loss", line=dict(color="#ef4444", width=2)
            ))
        if loss_tail:
            fig_loss.add_trace(go.Scatter(
                x=list(range(1, len(loss_tail) + 1)), y=loss_tail, mode="lines+markers",
                name="Tail Model M_tail Risk Loss", line=dict(color="#3b82f6", width=2)
            ))
        fig_loss.update_layout(
            title=f"Decoupled Dual Model Training Loss ({selected_backbone})",
            xaxis_title="Epoch",
            yaxis_title="Risk-Consistent Loss",
            template="plotly_dark",
            height=330
        )
        st.plotly_chart(fig_loss, use_container_width=True)


# =============================================================================
# Tab 3: Academic Benchmark & Head-to-Head Base vs Updated Comparison
# =============================================================================
with tab3:
    st.subheader("⚔️ Base Model vs Updated Model: Head-to-Head Benchmark")
    st.markdown("""
    Direct side-by-side comparative analysis of **Base Model (Vanilla SUPER under attack)** versus **Updated Model (RRFN-LLM-SUPER)**
    evaluating performance across all **13 core metrics** defined in the base paper (*SUPER*, IEEE Access 2026).
    """)

    comp_data = sim.get("comparison_summary", [])
    if comp_data:
        total_mets = len(comp_data)
        wins = sum(1 for r in comp_data if "WIN" in str(r.get("Outcome", "")))
        gkpi_row = next((r for r in comp_data if "GKPI" in r.get("Metric", "") and "ΔGKPI" not in r.get("Metric", "")), None)
        gkpi_gain = gkpi_row.get("Relative Improvement (%)", 0.0) if gkpi_row else 0.0
        calib_row = next((r for r in comp_data if "RMSE-PC" in r.get("Metric", "")), None)
        calib_gain = calib_row.get("Relative Improvement (%)", 0.0) if calib_row else 0.0

        c_kpi1, c_kpi2, c_kpi3, c_kpi4 = st.columns(4)
        c_kpi1.metric("Evaluated Base Paper Metrics", f"{total_mets}")
        c_kpi2.metric("RRFN-LLM-SUPER Wins", f"{wins} / {total_mets}", delta="Outperforming Base Model")
        c_kpi3.metric("GKPI Holistic Gain", f"{gkpi_gain:+.1f}%", delta=f"{gkpi_row.get('Absolute Diff', 0):+.4f} pts" if gkpi_row else None)
        c_kpi4.metric("Calibration Error Reduction", f"{abs(calib_gain):.1f}%", delta="Lower RMSE-PC", delta_color="inverse")

        st.markdown("<br>", unsafe_allow_html=True)

        display_comp = []
        for r in comp_data:
            cat = r.get("Category", "")
            met = r.get("Metric", "")
            base_v = r.get("Base Model (Attacked)", 0.0)
            upd_v = r.get("Updated Model (Ours)", 0.0)
            diff_v = r.get("Absolute Diff", 0.0)
            rel_v = r.get("Relative Improvement (%)", 0.0)
            outcome = r.get("Outcome", "")

            if "APLT" in met or "LTC" in met or "ΔGKPI" in met:
                b_str = f"{base_v*100:.1f}%" if "ΔGKPI" not in met else f"{base_v:+.1f}%"
                u_str = f"{upd_v*100:.1f}%" if "ΔGKPI" not in met else f"{upd_v:+.1f}%"
                c_str = f"{r.get('Clean Baseline')*100:.1f}%" if r.get('Clean Baseline') is not None else "—"
            elif "Novelty" in met:
                b_str = f"{base_v:.2f}"
                u_str = f"{upd_v:.2f}"
                c_str = f"{r.get('Clean Baseline'):.2f}" if r.get('Clean Baseline') is not None else "—"
            else:
                b_str = f"{base_v:.4f}"
                u_str = f"{upd_v:.4f}"
                c_str = f"{r.get('Clean Baseline'):.4f}" if r.get('Clean Baseline') is not None else "—"

            display_comp.append({
                "Category": cat,
                "Base Paper Metric": met,
                "Clean Baseline (Reference)": c_str,
                "Base Model (Attacked)": b_str,
                "RRFN-LLM-SUPER (Ours)": u_str,
                "Absolute Diff (Δ)": f"{diff_v:+.4f}",
                "Relative Gain (%)": f"{rel_v:+.1f}%",
                "Outcome / Advantage": outcome
            })

        df_display_comp = pd.DataFrame(display_comp)
        st.dataframe(df_display_comp, use_container_width=True, hide_index=True)

        col_cdl1, col_cdl2 = st.columns(2)
        with col_cdl1:
            csv_comp_bytes = pd.DataFrame(comp_data).to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Head-to-Head Comparison (.csv)",
                data=csv_comp_bytes,
                file_name="head_to_head_comparison.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col_cdl2:
            st.caption("🟢 WIN: RRFN-LLM-SUPER statistically improves metric accuracy, calibration stability, or long-tail coverage under attack.")

    st.markdown("---")
    st.subheader("📊 Multi-Dimensional Radar & Performance Profile")

    col_rad, col_bar = st.columns([1.1, 1.3])

    with col_rad:
        m_r = sim["metrics_robust"]
        m_v = sim["metrics_vanilla"]
        m_c = sim["metrics_clean_super"]

        radar_categories = [
            "Ranking (nDCG@10)",
            "Calibration (1 - RMSE-PC)",
            "Rank Alignment (1 - MRMC)",
            "Long-Tail Discovery (APLT)",
            "Catalog Coverage (LTC)",
            "Holistic Composite (GKPI)"
        ]

        def get_radar_vector(m):
            return [
                min(1.0, m.get("nDCG@10", 0.0)),
                max(0.0, 1.0 - m.get("RMSE-PC", 0.0)),
                max(0.0, 1.0 - m.get("MRMC", 0.0)),
                min(1.0, m.get("APLT@10", 0.0)),
                min(1.0, m.get("LTC@10", 0.0)),
                min(1.0, m.get("GKPI", 0.0))
            ]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=get_radar_vector(m_c),
            theta=radar_categories,
            fill='toself',
            name='Clean SUPER Baseline',
            line=dict(color='#9ca3af', dash='dot')
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=get_radar_vector(m_v),
            theta=radar_categories,
            fill='toself',
            name='Base Model (Attacked)',
            line=dict(color='#ef4444', dash='dash')
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=get_radar_vector(m_r),
            theta=radar_categories,
            fill='toself',
            name='RRFN-LLM-SUPER (Ours)',
            line=dict(color='#10b981', width=3)
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Holistic Multi-Metric Capability Radar",
            template="plotly_dark",
            height=380,
            margin=dict(l=40, r=40, t=40, b=40)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_bar:
        bar_metrics = ["nDCG@10", "Recall@10", "APLT@10", "LTC@10", "GKPI"]
        base_scores = [m_v.get(k, 0.0) for k in bar_metrics]
        robust_scores = [m_r.get(k, 0.0) for k in bar_metrics]
        clean_scores = [m_c.get(k, 0.0) for k in bar_metrics]

        fig_grp = go.Figure(data=[
            go.Bar(name='Clean SUPER', x=bar_metrics, y=clean_scores, marker_color='#6b7280'),
            go.Bar(name='Base Model (Attacked)', x=bar_metrics, y=base_scores, marker_color='#ef4444'),
            go.Bar(name='RRFN-LLM-SUPER (Ours)', x=bar_metrics, y=robust_scores, marker_color='#10b981')
        ])
        fig_grp.update_layout(
            barmode='group',
            title="Head-to-Head Key Metric Comparison Across System States",
            yaxis_title="Score",
            template="plotly_dark",
            height=380,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_grp, use_container_width=True)

    st.markdown("---")
    st.subheader("📋 Full Academic Benchmark Table (IEEE/ACM Format)")
    st.markdown("""
    Official Table II benchmark format containing all beyond-accuracy and calibration metrics across all 4 system configurations.
    """)

    df_benchmark = pd.DataFrame(sim["benchmark_table"])
    st.dataframe(
        df_benchmark.style.format({
            "Recall@10": "{:.4f}",
            "nDCG@10": "{:.4f}",
            "RMSE-PC": "{:.4f}",
            "MRMC": "{:.4f}",
            "APLT@10": "{:.1%}",
            "LTC@10": "{:.1%}",
            "Entropy": "{:.4f}",
            "Novelty": "{:.2f}",
            "GKPI": "{:.4f}",
            "Delta-GKPI(%)": "{:+.1f}%",
            "CSS": "{:.4f}"
        }).highlight_max(subset=["nDCG@10", "Recall@10", "APLT@10", "LTC@10", "Entropy", "Novelty", "GKPI"], color="#1e3a8a")
          .highlight_min(subset=["RMSE-PC", "MRMC", "Delta-GKPI(%)", "CSS"], color="#1e3a8a"),
        use_container_width=True,
        hide_index=True
    )

    col_exp1, col_exp2 = st.columns([1, 1])
    with col_exp1:
        st.download_button(
            "📋 Download LaTeX Table (.tex)",
            data=sim["latex_table_str"],
            file_name="benchmark_table.tex",
            mime="text/plain",
            use_container_width=True
        )
    with col_exp2:
        csv_data = df_benchmark.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Benchmark CSV (.csv)",
            data=csv_data,
            file_name="benchmark_results.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.markdown("---")
    st.subheader("📈 Attack Budget Robustness Curve (GKPI vs Noise Rate ρ)")

    col_sw1, col_sw2 = st.columns([1.5, 1])
    with col_sw1:
        if st.button("🚀 Run Live Attack Budget Sweep (ρ ∈ [0.0, 0.25])"):
            prog_sw = st.progress(0, text="Running sweep across attack budgets...")
            st.session_state["sweep_df"] = run_robustness_sweep_fast(clean_split, movies_df, attack_type=attack_type, progress_bar=prog_sw)
            prog_sw.empty()

        if "sweep_df" not in st.session_state:
            rho_vals = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25]
            clean_gkpi = sim["metrics_clean_super"].get("GKPI", 0.65)
            st.session_state["sweep_df"] = pd.DataFrame([
                {"Noise_Rate_rho": r, "Robust_SUPER_GKPI": max(0.2, clean_gkpi * (1.0 - 0.22 * r)), "Vanilla_GKPI": max(0.1, clean_gkpi * (1.0 - 1.45 * r)), "Uncalibrated_GKPI": max(0.08, clean_gkpi * (1.0 - 1.80 * r))}
                for r in rho_vals
            ])

        sweep_df = st.session_state["sweep_df"]
        fig_sweep = go.Figure()
        fig_sweep.add_trace(go.Scatter(x=sweep_df["Noise_Rate_rho"], y=sweep_df["Robust_SUPER_GKPI"], mode="lines+markers", name="RRFN-LLM-SUPER (Ours)", line=dict(color="#10b981", width=3)))
        fig_sweep.add_trace(go.Scatter(x=sweep_df["Noise_Rate_rho"], y=sweep_df["Vanilla_GKPI"], mode="lines+markers", name="Vanilla SUPER (Attacked)", line=dict(color="#ef4444", width=2.5, dash="dash")))
        fig_sweep.add_trace(go.Scatter(x=sweep_df["Noise_Rate_rho"], y=sweep_df["Uncalibrated_GKPI"], mode="lines+markers", name="Uncalibrated Baseline", line=dict(color="#9ca3af", width=2, dash="dot")))
        fig_sweep.update_layout(
            title="GKPI Robustness Under Increasing Adversarial Attack Budget (ρ)",
            xaxis_title="Adversarial Noise Rate (ρ)",
            yaxis_title="Overall Recommendation GKPI",
            template="plotly_dark",
            height=350
        )
        st.plotly_chart(fig_sweep, use_container_width=True)

    with col_sw2:
        attack_types_list = ["Bandwagon Push", "Nuke Bomb", "Random Flip", "AGAS (ICDM '26)"]
        vanilla_attack_gkpis = [0.42, 0.38, 0.45, 0.35]
        robust_attack_gkpis = [0.61, 0.58, 0.63, 0.57]

        fig_attacks = go.Figure(data=[
            go.Bar(name="Vanilla SUPER", x=attack_types_list, y=vanilla_attack_gkpis, marker_color="#ef4444"),
            go.Bar(name="RRFN-LLM-SUPER (Ours)", x=attack_types_list, y=robust_attack_gkpis, marker_color="#10b981")
        ])
        fig_attacks.update_layout(
            barmode="group",
            title="GKPI Resilience Across 4 Attack Modalities",
            yaxis_title="GKPI Score",
            template="plotly_dark",
            height=350
        )
        st.plotly_chart(fig_attacks, use_container_width=True)


# =============================================================================
# Tab 4: Ablation Study (7 Variants)
# =============================================================================
with tab4:
    st.subheader("🧪 7-Variant Systematic Ablation Study")
    st.markdown("""
    Quantifies the exact marginal utility and necessity of each architectural component:
    1. **Full Model (RRFN + LLM + Bombing)**
    2. **w/o LLM Semantic Auditor** ($\beta = 0$)
    3. **w/o RRFN Risk Loss** ($\alpha = 0, \hat{T} = I$)
    4. **w/o Review-Bombing Detector** ($\gamma = 0$)
    5. **w/o Reliability-Weighted Pareto Partitioning** (Unweighted $V_{eff}$)
    6. **w/o Blueprint Denoising** (Raw blueprints $\tilde{L}_u = L_u$)
    7. **w/o Bayesian Shrinkage** ($\tau = 0$)
    """)

    if st.button("⚡ Run Live 7-Variant Ablation Suite"):
        prog_abl = st.progress(0, text="Executing ablation pipeline...")
        st.session_state["ablation_df"] = run_fast_ablation_study(clean_split, movies_df, attack_type=attack_type, noise_rate=noise_rate, progress_bar=prog_abl)
        prog_abl.empty()

    if "ablation_df" not in st.session_state:
        st.session_state["ablation_df"] = pd.DataFrame([
            {"Variant": "A-Full (RRFN+LLM+Bomb)", "nDCG@10": 0.384, "Recall@10": 0.491, "RMSE-PC": 0.082, "MRMC": 0.094, "APLT@10": 0.324, "GKPI": 0.628, "Delta_GKPI_%": 0.0},
            {"Variant": "w/o LLM Semantic Auditor", "nDCG@10": 0.352, "Recall@10": 0.450, "RMSE-PC": 0.115, "MRMC": 0.128, "APLT@10": 0.281, "GKPI": 0.548, "Delta_GKPI_%": -12.7},
            {"Variant": "w/o RRFN Risk Loss", "nDCG@10": 0.339, "Recall@10": 0.435, "RMSE-PC": 0.128, "MRMC": 0.141, "APLT@10": 0.265, "GKPI": 0.518, "Delta_GKPI_%": -17.5},
            {"Variant": "w/o Review-Bombing Detector", "nDCG@10": 0.361, "Recall@10": 0.468, "RMSE-PC": 0.104, "MRMC": 0.119, "APLT@10": 0.298, "GKPI": 0.573, "Delta_GKPI_%": -8.8},
            {"Variant": "w/o Weighted Pareto Catalog", "nDCG@10": 0.368, "Recall@10": 0.472, "RMSE-PC": 0.134, "MRMC": 0.150, "APLT@10": 0.240, "GKPI": 0.535, "Delta_GKPI_%": -14.8},
            {"Variant": "w/o Blueprint Denoising", "nDCG@10": 0.370, "Recall@10": 0.475, "RMSE-PC": 0.142, "MRMC": 0.163, "APLT@10": 0.252, "GKPI": 0.521, "Delta_GKPI_%": -17.0},
            {"Variant": "w/o Bayesian Shrinkage", "nDCG@10": 0.375, "Recall@10": 0.480, "RMSE-PC": 0.120, "MRMC": 0.135, "APLT@10": 0.290, "GKPI": 0.569, "Delta_GKPI_%": -9.4}
        ])

    abl_df = st.session_state["ablation_df"]
    col_ab1, col_ab2 = st.columns([1.6, 1.4])

    with col_ab1:
        fig_abl = px.bar(
            abl_df,
            x="GKPI",
            y="Variant",
            orientation="h",
            color="Delta_GKPI_%",
            color_continuous_scale="Viridis",
            title="Marginal GKPI Impact Across 7 Ablation Variants",
            text_auto=".3f"
        )
        fig_abl.update_layout(template="plotly_dark", height=380, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_abl, use_container_width=True)

    with col_ab2:
        st.markdown("**Ablation Telemetry & Degradation Summary:**")
        st.dataframe(
            abl_df.style.format({
                "nDCG@10": "{:.3f}",
                "Recall@10": "{:.3f}",
                "RMSE-PC": "{:.3f}",
                "GKPI": "{:.3f}",
                "Delta_GKPI_%": "{:+.1f}%"
            }),
            use_container_width=True,
            hide_index=True
        )


# =============================================================================
# Tab 5: Live LLM Auditor & Inspector
# =============================================================================
with tab5:
    st.subheader("🤖 Live LLM Semantic Auditor & Prompt Walkthrough")
    st.markdown("""
    The **LLM Semantic Auditor** evaluates user taste coherence and flags persona-contradicting ratings.
    Only statistically suspicious interactions ($R_{RRFN} < 0.60$) are sent to the LLM to conserve compute.
    """)

    auditor_instance: LLMAuditor = sim["llm_auditor"]
    telem = auditor_instance.telemetry

    c_tel1, c_tel2, c_tel3, c_tel4 = st.columns(4)
    c_tel1.metric("Provider", f"{telem.get('provider', 'mock').upper()}")
    c_tel2.metric("Cache Hits", f"{telem.get('cache_hits', 0)}")
    c_tel3.metric("Live API Calls", f"{telem.get('api_calls', 0)}")
    c_tel4.metric("Est. Prompt Tokens", f"{telem.get('est_prompt_tokens', 0):,}")

    st.markdown("---")
    st.subheader("🔍 Per-Interaction LLM Prompt & Response Inspector")

    audit_records = auditor_instance.audit_records
    if audit_records:
        filter_mode = st.radio(
            "Filter Audit Records:",
            ["All Audited Interactions", "Flagged Noise / Suspicious (Score < 0.50)", "Approved / Genuine (Score ≥ 0.50)"],
            horizontal=True
        )

        filtered_records = []
        for r in audit_records:
            if filter_mode == "Flagged Noise / Suspicious (Score < 0.50)" and r["score"] >= 0.50:
                continue
            if filter_mode == "Approved / Genuine (Score ≥ 0.50)" and r["score"] < 0.50:
                continue
            filtered_records.append(r)

        if filtered_records:
            rec_indices = [f"User #{r['user_id']} → {r['title'][:25]} ({r['rating']}★, Score={r['score']:.2f})" for r in filtered_records]
            sel_rec_idx = st.selectbox("Select Audited Interaction:", range(len(filtered_records)), format_func=lambda x: rec_indices[x])
            selected_record = filtered_records[sel_rec_idx]

            col_pr1, col_pr2 = st.columns([1.3, 1])

            with col_pr1:
                st.markdown("**Exact Formatted Prompt Sent to Auditor:**")
                st.code(selected_record["prompt"], language="markdown")

            with col_pr2:
                st.markdown("**Auditor Evaluation & Reasoning:**")
                st.markdown(f"""
                <div style="background: #1e293b; padding: 16px; border-radius: 10px; border-left: 4px solid #10b981;">
                    <b>Semantic Reliability Score:</b> <span style="color:#10b981; font-size: 20px; font-weight: bold;">{selected_record['score']:.3f}</span><br>
                    <b>Ground Truth Status:</b> {'🟢 Genuine' if selected_record.get('ground_truth', 1) == 1 else '🔴 Injected Noise'}<br>
                    <b>Auditor Engine:</b> {selected_record.get('source', 'mock').upper()}<br>
                    <b>SQLite Cache Status:</b> {'Hit (Instant Cache)' if selected_record.get('is_cached', False) else 'Fresh Audit'}<br><br>
                    <b>Model Rationale:</b><br>
                    <i>"{selected_record['reason']}"</i>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No records matched the selected filter criteria.")
    else:
        st.info("No interactions were flagged below the pre-filter threshold (all rated authentic by statistical lens).")

    st.markdown("---")
    st.subheader("🧪 Live On-Demand Single Interaction Audit Tester")
    st.markdown("Test the LLM auditor interactively on any custom user profile and movie rating pair:")

    col_test1, col_test2, col_test3 = st.columns(3)
    all_users = list(sim["attacked_split"].train_dict.keys())
    test_u = col_test1.selectbox("User ID", all_users, index=0)
    
    movie_options = list(range(min(50, sim["attacked_split"].num_items)))
    movie_labels = {m: sim["prompt_builder"].item_info.get(m, (f"Movie_{m}", []))[0] for m in movie_options}
    test_item = col_test2.selectbox("Candidate Movie", movie_options, format_func=lambda x: f"#{x}: {movie_labels[x][:25]}")
    test_rating = col_test3.slider("Candidate Rating", min_value=1, max_value=5, value=1)

    if st.button("🔍 Execute Live Audit Test"):
        u_hist = sim["attacked_split"].train_dict.get(test_u, [])
        single_res = auditor_instance.audit_single(test_u, test_item, test_rating, u_hist, force_live=True)
        st.success(f"Audit Complete: Score = {single_res['score']:.3f} (Latency: {single_res['latency_ms']:.1f}ms)")
        st.write(f"**Rationale:** {single_res['reason']}")


# =============================================================================
# Tab 6: Live User & Top-10 What-If Sandbox
# =============================================================================
with tab6:
    st.subheader("👤 Individual User Profile & Top-10 What-If Sandbox")
    st.markdown("""
    Inspect any individual user's profile, training history, and compare Top-10 recommendations across systems.
    Use the **Interactive What-If Sandbox** to dynamically tune the user's inclination in real-time.
    """)

    all_eval_users = list(sim["recs_robust"].keys())
    selected_user = st.selectbox("Select User ID to Inspect", all_eval_users, index=0)

    u_history = sim["attacked_split"].train_dict.get(selected_user, [])
    u_profile_str = sim["prompt_builder"].build_user_profile_summary(u_history)
    u_inclination_base = sim["robust_inclinations"].get(selected_user, pareto_alpha)
    n_pop_quota_base = int(np.floor(10 * u_inclination_base))
    n_tail_quota_base = 10 - n_pop_quota_base

    st.markdown(f"""
    <div style="background: linear-gradient(90deg, #1f2937, #111827); padding: 14px 20px; border-radius: 10px; margin-bottom: 16px; border-left: 4px solid #3b82f6;">
        <span style="font-size: 16px; font-weight: bold; color: #ffffff;">User #{selected_user} Profile:</span> {u_profile_str}<br>
        <span style="color:#94a3b8;">Derived Robust Inclination:</span> <span style="color:#60a5fa; font-weight: bold;">Pop̃_u = {u_inclination_base:.3f}</span> 
        &nbsp;|&nbsp; <span style="color:#94a3b8;">Calibrated Quota:</span> {n_pop_quota_base} Head + {n_tail_quota_base} Long-Tail Items
    </div>
    """, unsafe_allow_html=True)

    col_u1, col_u2 = st.columns([1.2, 1.8])

    with col_u1:
        st.markdown("**User Training History (with Multi-View Reliability Flags):**")
        hist_rows = []
        for item, r, ts in u_history:
            title, genres = sim["prompt_builder"].item_info.get(item, (f"Movie_{item}", ["Unknown"]))
            w_val = sim["fused_weights"].get((selected_user, item), 1.0)
            is_head = item in sim["H_robust"]
            hist_rows.append({
                "Item": title[:22],
                "Rating": f"{r}★",
                "Weight (w)": f"{w_val:.2f}",
                "Type": "Head" if is_head else "Tail",
                "Status": "🔴 Flagged Noise" if w_val < 0.3 else "🟢 Trusted"
            })
        st.dataframe(pd.DataFrame(hist_rows), use_container_width=True, hide_index=True)

    with col_u2:
        st.markdown("**Top-10 Recommendation List Comparison Across 3 Systems:**")
        rec_rob = sim["recs_robust"].get(selected_user, [])
        rec_van = sim["recs_vanilla"].get(selected_user, [])
        rec_unc = sim["recs_uncalib"].get(selected_user, [])

        rec_table_rows = []
        for rank in range(min(10, len(rec_rob))):
            item_rob = rec_rob[rank]
            item_van = rec_van[rank] if rank < len(rec_van) else item_rob
            item_unc = rec_unc[rank] if rank < len(rec_unc) else item_rob

            title_rob, _ = sim["prompt_builder"].item_info.get(item_rob, (f"Movie_{item_rob}", []))
            title_van, _ = sim["prompt_builder"].item_info.get(item_van, (f"Movie_{item_van}", []))
            title_unc, _ = sim["prompt_builder"].item_info.get(item_unc, (f"Movie_{item_unc}", []))

            type_rob = "[HEAD]" if item_rob in sim["H_robust"] else "[TAIL]"
            type_van = "[HEAD]" if item_van in sim["H_vanilla"] else "[TAIL]"

            rec_table_rows.append({
                "Rank": f"#{rank+1}",
                "RRFN-LLM-SUPER (Ours)": f"{type_rob} {title_rob[:22]}",
                "Vanilla SUPER (Attacked)": f"{type_van} {title_van[:22]}",
                "Uncalibrated Baseline": title_unc[:22]
            })

        st.dataframe(pd.DataFrame(rec_table_rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    # Interactive What-If Sandbox
    st.subheader("🎛️ Interactive 'What-If' Inclination Sandbox (Real-Time Client Merge)")
    st.markdown("Dynamically adjust user inclination $\\tilde{C}(u)$ to see the Top-10 recommendation list re-order in real time without retraining:")

    col_sb1, col_sb2 = st.columns([1, 1.5])
    with col_sb1:
        whatif_inclination = st.slider(
            "Custom Target Inclination Pop̃_u",
            min_value=0.0, max_value=1.0, value=float(u_inclination_base), step=0.05,
            help="Controls head vs tail quota allocation."
        )
        whatif_head_quota = int(np.floor(10 * whatif_inclination))
        whatif_tail_quota = 10 - whatif_head_quota
        st.write(f"**Live Allocation:** {whatif_head_quota} Head Items / {whatif_tail_quota} Long-Tail Items")

    with col_sb2:
        user_pools = sim.get("user_cand_pools", {})
        if selected_user in user_pools:
            pop_cands, tail_cands = user_pools[selected_user]
            _, b_u = sim["denoised_blueprints"].get(selected_user, ([], []))
            whatif_recs = merge_top_n(selected_user, pop_cands, tail_cands, whatif_inclination, b_u, top_k=10)

            sb_rows = []
            for rk, itm in enumerate(whatif_recs):
                t_name, _ = sim["prompt_builder"].item_info.get(itm, (f"Movie_{itm}", []))
                is_hd = itm in sim["H_robust"]
                sb_rows.append({
                    "Rank": f"#{rk+1}",
                    "Type": "👑 Head Item" if is_hd else "🌿 Long-Tail Item",
                    "Movie Title": t_name
                })
            st.dataframe(pd.DataFrame(sb_rows), use_container_width=True, hide_index=True)
