"""
robust_super/experiments/run_comprehensive_comparison.py
Executes a rigorous head-to-head comparison between:
1. Base Model (Vanilla SUPER under noise / review bombing)
2. Updated Model (RRFN-LLM-SUPER with multi-view auditing and risk-consistent loss)
across all metrics from the base paper (IEEE Access 2026).
"""
import os
import sys
import yaml
import torch
import numpy as np
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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
from src.bombing_detector.bomb_scorer import compute_bomb_scores
from src.fusion.reliability_fusion import compute_R_RRFN, fuse_reliability_scores
from src.catalog.pareto_partition import compute_effective_volume, pareto_partition
from src.trainer.dual_trainer import DualModelTrainer, build_data_loader
from src.super_engine.inclination import compute_robust_inclination
from src.super_engine.blueprint import build_denoised_blueprints
from src.super_engine.merger import merge_top_n
from src.evaluation.metrics import evaluate_recommendations
from src.evaluation.robustness_metrics import (
    compute_robustness_metrics,
    generate_latex_benchmark_table,
    generate_metric_comparison_summary
)
from src.evaluation.reporter import ExperimentReporter


def run_full_benchmark_comparison(
    attacks=["bandwagon", "nuke_bomb", "random_flip", "agas"],
    noise_rates=[0.05, 0.10, 0.20],
    subsample_users=300,
    top_k=10
):
    config_path = os.path.join(os.path.dirname(__file__), "../configs/default_config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() and cfg["training"]["device"] == "cuda" else "cpu"
    print(f"[*] Running Comprehensive Base vs Updated Benchmark on device: {device}")

    # Load Clean Data
    loader = MovieLensLoader(
        data_dir=cfg["data"]["raw_dir"],
        min_user_interactions=20,
        min_item_interactions=10
    )
    ratings_df, movies_df, users_df = loader.load_data()

    if subsample_users and subsample_users < len(users_df):
        print(f"[*] Subsampling {subsample_users} most active users for evaluation...")
        top_u = ratings_df["user_id"].value_counts().head(subsample_users).index
        ratings_df = ratings_df[ratings_df["user_id"].isin(top_u)].copy()
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

    clean_split = preprocess_dataset(ratings_df, movies_df, users_df)
    simulator = AttackSimulator(seed=42)
    prompt_builder = LLMPromptBuilder(movies_df)
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    # 1. Clean Baseline Evaluation
    print("\n[*] Evaluating Clean SUPER Baseline (No Noise)...")
    def clean_model_factory():
        return NeuMF(clean_split.num_users, clean_split.num_items, 32, [64, 32])

    clean_trainer = DualModelTrainer(clean_model_factory, device=device, lr=0.003, epochs=3)
    clean_warm = clean_model_factory().to(device)
    clean_unweighted = {k: 1.0 for k in clean_split.weight_dict}
    clean_warm_loader = build_data_loader(clean_split.train_dict, clean_unweighted, batch_size=1024)
    clean_trainer.warm_train(clean_warm, clean_warm_loader, warm_epochs=2)

    V_clean = compute_effective_volume(clean_split.train_dict, clean_unweighted, clean_split.num_items)
    H_clean, T_clean = pareto_partition(V_clean, 0.20)
    p_loader = build_data_loader(clean_split.train_dict, clean_unweighted, item_filter=H_clean, batch_size=1024)
    t_loader = build_data_loader(clean_split.train_dict, clean_unweighted, item_filter=T_clean, batch_size=1024)
    M_pop_clean, _ = clean_trainer.train_single_model(p_loader, clean_split.val_dict, H_clean)
    M_tail_clean, _ = clean_trainer.train_single_model(t_loader, clean_split.val_dict, T_clean)

    inclin_clean = compute_robust_inclination(clean_split.train_dict, H_clean, clean_unweighted)
    bp_clean = build_denoised_blueprints(clean_split.train_dict, clean_unweighted, clean_split.user_mean_ratings, H_clean)

    recs_clean = {}
    eval_users = list(clean_split.test_dict.keys())[:min(150, len(clean_split.test_dict))]
    h_list_c = [i for i in H_clean if i < clean_split.num_items]
    t_list_c = [i for i in T_clean if i < clean_split.num_items]

    for u in eval_users:
        sp = M_pop_clean.score_items(u, h_list_c, device=device)
        st = M_tail_clean.score_items(u, t_list_c, device=device)
        pc = [h_list_c[idx] for idx in torch.topk(sp, k=min(top_k, len(h_list_c))).indices.cpu().numpy()]
        tc = [t_list_c[idx] for idx in torch.topk(st, k=min(top_k, len(t_list_c))).indices.cpu().numpy()]
        _, bu = bp_clean.get(u, ([], []))
        recs_clean[u] = merge_top_n(u, pc, tc, inclin_clean.get(u, 0.2), bu, top_k=top_k)

    metrics_clean = evaluate_recommendations(recs_clean, clean_split.test_dict, clean_split.train_dict, H_clean, T_clean, inclin_clean, top_k=top_k)
    print(f"[+] Clean SUPER Baseline: GKPI = {metrics_clean.get('GKPI', 0.0):.4f}, nDCG@10 = {metrics_clean.get('nDCG@10', 0.0):.4f}, RMSE-PC = {metrics_clean.get('RMSE-PC', 0.0):.4f}")

    all_comparison_records = []
    benchmark_table_records = []

    # 2. Iterate Attacks and Noise Budgets
    for attack_type in attacks:
        for noise_rate in noise_rates:
            print(f"\n{'='*75}\n[>] Attack: {attack_type.upper()} | Noise Rate rho = {noise_rate:.2f}\n{'='*75}")

            attacked_split = simulator.inject_attack(clean_split, attack_type, noise_rate)
            unweighted = {k: 1.0 for k in attacked_split.weight_dict}

            def model_factory():
                return NeuMF(attacked_split.num_users, attacked_split.num_items, 32, [64, 32])

            trainer = DualModelTrainer(model_factory, device=device, lr=0.003, epochs=3)
            warm_model = model_factory().to(device)
            warm_loader = build_data_loader(attacked_split.train_dict, unweighted, batch_size=1024)
            trainer.warm_train(warm_model, warm_loader, warm_epochs=2)

            # --- Base Model (Vanilla SUPER under attack) ---
            V_eff_base = compute_effective_volume(attacked_split.train_dict, unweighted, attacked_split.num_items)
            H_base, T_base = pareto_partition(V_eff_base, 0.20)
            p_loader_base = build_data_loader(attacked_split.train_dict, unweighted, item_filter=H_base, batch_size=1024)
            t_loader_base = build_data_loader(attacked_split.train_dict, unweighted, item_filter=T_base, batch_size=1024)
            M_pop_base, _ = trainer.train_single_model(p_loader_base, attacked_split.val_dict, H_base)
            M_tail_base, _ = trainer.train_single_model(t_loader_base, attacked_split.val_dict, T_base)
            inclin_base = compute_robust_inclination(attacked_split.train_dict, H_base, unweighted, shrinkage_tau=0.0)

            # --- Updated Model (RRFN-LLM-SUPER under attack) ---
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

            fused_weights = fuse_reliability_scores(R_RRFN, R_LLM, R_bomb, alpha=0.50, beta=0.30, gamma=0.20)
            V_eff_rob = compute_effective_volume(attacked_split.train_dict, fused_weights, attacked_split.num_items)
            H_rob, T_rob = pareto_partition(V_eff_rob, 0.20)

            p_loader_rob = build_data_loader(attacked_split.train_dict, fused_weights, item_filter=H_rob, batch_size=1024)
            t_loader_rob = build_data_loader(attacked_split.train_dict, fused_weights, item_filter=T_rob, batch_size=1024)
            M_pop_rob, _ = trainer.train_single_model(p_loader_rob, attacked_split.val_dict, H_rob, T_hat=trans_module.T_hat)
            M_tail_rob, _ = trainer.train_single_model(t_loader_rob, attacked_split.val_dict, T_rob, T_hat=trans_module.T_hat)

            inclin_rob = compute_robust_inclination(attacked_split.train_dict, H_rob, fused_weights, shrinkage_tau=5.0)
            bp_rob = build_denoised_blueprints(attacked_split.train_dict, fused_weights, attacked_split.user_mean_ratings, H_rob, filter_threshold=0.30)

            # Recommendations
            recs_base = {}
            recs_rob = {}
            h_list_rob = [i for i in H_rob if i < attacked_split.num_items]
            t_list_rob = [i for i in T_rob if i < attacked_split.num_items]
            h_list_base = [i for i in H_base if i < attacked_split.num_items]
            t_list_base = [i for i in T_base if i < attacked_split.num_items]

            for u in eval_users:
                # Base model recommendations
                sp_b = M_pop_base.score_items(u, h_list_base, device=device)
                st_b = M_tail_base.score_items(u, t_list_base, device=device)
                pc_b = [h_list_base[idx] for idx in torch.topk(sp_b, k=min(top_k, len(h_list_base))).indices.cpu().numpy()]
                tc_b = [t_list_base[idx] for idx in torch.topk(st_b, k=min(top_k, len(t_list_base))).indices.cpu().numpy()]
                recs_base[u] = merge_top_n(u, pc_b, tc_b, 0.50, [1, 1, 1, 1, 1, 0, 0, 0, 0, 0], top_k=top_k)

                # Updated model recommendations
                sp_r = M_pop_rob.score_items(u, h_list_rob, device=device)
                st_r = M_tail_rob.score_items(u, t_list_rob, device=device)
                pc_r = [h_list_rob[idx] for idx in torch.topk(sp_r, k=min(top_k, len(h_list_rob))).indices.cpu().numpy()]
                tc_r = [t_list_rob[idx] for idx in torch.topk(st_r, k=min(top_k, len(t_list_rob))).indices.cpu().numpy()]
                _, bu_r = bp_rob.get(u, ([], []))
                recs_rob[u] = merge_top_n(u, pc_r, tc_r, inclin_rob.get(u, 0.2), bu_r, top_k=top_k)

            # Evaluate Metrics on all Base Paper dimensions
            m_base = evaluate_recommendations(recs_base, clean_split.test_dict, attacked_split.train_dict, H_base, T_base, inclin_clean, top_k=top_k)
            m_rob = evaluate_recommendations(recs_rob, clean_split.test_dict, attacked_split.train_dict, H_rob, T_rob, inclin_rob, top_k=top_k)

            r_base = compute_robustness_metrics(m_base, metrics_clean, attacked_split.ground_truth_labels, unweighted, recommendations=recs_base, top_k=top_k)
            r_rob = compute_robustness_metrics(m_rob, metrics_clean, attacked_split.ground_truth_labels, fused_weights, recommendations=recs_rob, top_k=top_k)

            m_base.update(r_base)
            m_rob.update(r_rob)

            # Head to head comparison summary
            summary = generate_metric_comparison_summary(m_base, m_rob, metrics_clean, top_k=top_k)
            for item in summary:
                item["Attack"] = attack_type
                item["NoiseRate"] = noise_rate
                all_comparison_records.append(item)

            # Append to benchmark table records
            row_base = {"Method": f"Base SUPER (Attacked {attack_type} rho={noise_rate:.2f})", "Attack": attack_type, "NoiseRate": noise_rate}
            row_base.update(m_base)
            benchmark_table_records.append(row_base)

            row_rob = {"Method": f"RRFN-LLM-SUPER (Ours, rho={noise_rate:.2f})", "Attack": attack_type, "NoiseRate": noise_rate}
            row_rob.update(m_rob)
            benchmark_table_records.append(row_rob)

            print(f"[*] Base Model:    GKPI = {m_base.get('GKPI', 0):.4f} | Delta-GKPI = {m_base.get('Delta-GKPI(%)', 0):+.1f}% | RMSE-PC = {m_base.get('RMSE-PC', 0):.4f}")
            print(f"[*] Updated Model: GKPI = {m_rob.get('GKPI', 0):.4f} | Delta-GKPI = {m_rob.get('Delta-GKPI(%)', 0):+.1f}% | RMSE-PC = {m_rob.get('RMSE-PC', 0):.4f} | F1 = {m_rob.get('Denoising-F1', 0):.3f}")

    # Export All Results
    df_comparison = pd.DataFrame(all_comparison_records)
    csv_comp_path = os.path.join(results_dir, "base_vs_updated_comparison.csv")
    df_comparison.to_csv(csv_comp_path, index=False)
    print(f"\n[+] Saved complete head-to-head comparison to: {csv_comp_path}")

    df_benchmark = pd.DataFrame(benchmark_table_records)
    csv_bench_path = os.path.join(results_dir, "complete_metrics_benchmark.csv")
    df_benchmark.to_csv(csv_bench_path, index=False)
    print(f"[+] Saved complete benchmark table to: {csv_bench_path}")

    # Generate LaTeX Tables
    latex_table_str = generate_latex_benchmark_table(benchmark_table_records, full_metrics=True)
    tex_path = os.path.join(results_dir, "complete_metrics_benchmark.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex_table_str)
    print(f"[+] Saved publishable LaTeX table to: {tex_path}")

    return df_comparison, df_benchmark


if __name__ == "__main__":
    run_full_benchmark_comparison(subsample_users=300)