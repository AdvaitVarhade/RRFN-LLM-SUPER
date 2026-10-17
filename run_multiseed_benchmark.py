"""
run_multiseed_benchmark.py
--------------------------
Executes 5-seed statistical benchmark (seeds = [42, 43, 44, 45, 46]) across:
1. MovieLens-1M (sample sub-graph, NeuMF backbone)
2. Amazon Reviews (SimGCL backbone)
3. Yelp Academic (SGL backbone)

Under Bandwagon Shilling Attack (rho = 0.15).
Evaluates competitive baselines:
- Plain LightGCN (SIGIR '20)
- Plain SGL (SIGIR '21)
- T-CE / SGDL (NeurIPS '22)
- Trust-GRS (AAAI '25)
- Uncalibrated Backbone
- Vanilla SUPER (Clean Baseline)
- Vanilla SUPER (Attacked bandwagon rho=0.15)
- RRFN-LLM-SUPER (Ours, robust decoupled)

Computes exact mean +- std, paired t-test p-values, attack-aware genuine LTC (g-LTC@10),
and spam target exposure rate (STER@10).
"""

import os
import sys
import json
import time
import copy
import random
import numpy as np
import pandas as pd
import scipy.stats as stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROBUST_DIR = os.path.join(SCRIPT_DIR, "robust_super") if os.path.basename(SCRIPT_DIR) != "robust_super" else SCRIPT_DIR
if ROBUST_DIR not in sys.path:
    sys.path.insert(0, ROBUST_DIR)

import torch
import torch.nn as nn

from src.data.loader import MovieLensLoader
from src.data.amazon_loader import AmazonReviewLoader
from src.data.yelp_loader import YelpLoader
from src.data.preprocessor import preprocess_dataset
from src.data.attack_simulator import AttackSimulator
from src.models.neumf import NeuMF
from src.models.simgcl import SimGCL
from src.models.sgl import SGL
from src.models.lightgcn import LightGCN
from src.models.tce_sgdl import TCE_SGDL
from src.models.trust_grs import TrustGRS
from src.trainer.baseline_trainer import BaselineTrainer
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
from src.evaluation.robustness_metrics import compute_robustness_metrics, compute_roc_pr_data


def load_domain_data(domain: str, dataset_scale: str = "full"):
    """Loads and pre-processes the dataset into standard 0-indexed DataFrames."""
    if domain == "movielens":
        raw_dir = os.path.join(ROBUST_DIR, "data", "raw", "ml-1m")
        if dataset_scale == "full":
            loader = MovieLensLoader(data_dir=raw_dir, use_full_catalog=True)
        else:
            loader = MovieLensLoader(data_dir=raw_dir, min_user_interactions=10, min_item_interactions=5)
    elif domain in ("amazon", "amazon_electronics"):
        raw_dir = os.path.join(ROBUST_DIR, "data", "amazon")
        loader = AmazonReviewLoader(data_dir=raw_dir, min_user_interactions=5, min_item_interactions=5)
    elif domain == "yelp":
        raw_dir = os.path.join(ROBUST_DIR, "data", "yelp")
        loader = YelpLoader(data_dir=raw_dir, min_user_interactions=5, min_item_interactions=5)
    else:
        raise ValueError(f"Unknown domain: {domain}")

    ratings_df, items_df, users_df = loader.load_data()

    if dataset_scale == "sample" and len(ratings_df["user_id"].unique()) > 300:
        top_users = ratings_df["user_id"].value_counts().head(300).index
        ratings_df = ratings_df[ratings_df["user_id"].isin(top_users)].copy()

        u_map = {old: new for new, old in enumerate(sorted(ratings_df["user_id"].unique()))}
        i_map = {old: new for new, old in enumerate(sorted(ratings_df["item_id"].unique()))}

        if items_df.empty:
            items_df = pd.DataFrame({"item_id": list(i_map.keys()), "title": [f"Item_{k}" for k in i_map.keys()]})
        else:
            items_df = items_df[items_df["item_id"].isin(i_map.keys())].copy()
        items_df["item_id"] = items_df["item_id"].map(i_map)
        items_df = items_df.sort_values("item_id").reset_index(drop=True)

        ratings_df["user_id"] = ratings_df["user_id"].map(u_map)
        ratings_df["item_id"] = ratings_df["item_id"].map(i_map)

        if users_df.empty:
            users_df = pd.DataFrame({"user_id": list(u_map.keys())})
        else:
            users_df = users_df[users_df["user_id"].isin(u_map.keys())].copy()
        users_df["user_id"] = users_df["user_id"].map(u_map)
        users_df = users_df.sort_values(by="user_id").reset_index(drop=True)
    elif domain != "movielens":
        u_map = {old: new for new, old in enumerate(sorted(ratings_df["user_id"].unique()))}
        i_map = {old: new for new, old in enumerate(sorted(ratings_df["item_id"].unique()))}

        if items_df.empty:
            items_df = pd.DataFrame({"item_id": list(i_map.keys()), "title": [f"Item_{k}" for k in i_map.keys()]})
        else:
            items_df = items_df[items_df["item_id"].isin(i_map.keys())].copy()
        items_df["item_id"] = items_df["item_id"].map(i_map)
        items_df = items_df.sort_values("item_id").reset_index(drop=True)

        ratings_df["user_id"] = ratings_df["user_id"].map(u_map)
        ratings_df["item_id"] = ratings_df["item_id"].map(i_map)

        if users_df.empty:
            users_df = pd.DataFrame({"user_id": list(u_map.keys())})
        else:
            users_df = users_df[users_df["user_id"].isin(u_map.keys())].copy()
        users_df["user_id"] = users_df["user_id"].map(u_map)
        users_df = users_df.sort_values(by="user_id").reset_index(drop=True)

    return ratings_df, items_df, users_df


def run_single_domain_multiseed(domain="movielens", seeds=[42, 43, 44, 45, 46], device="cuda", dataset_scale="full", eval_users_limit=None):
    print(f"\n=================================================================")
    print(f"  RUNNING 5-SEED BENCHMARK: {domain.upper()} (Scale: {dataset_scale}, Device: {device.upper()})")
    print(f"=================================================================")
    
    ratings_df_raw, items_df_raw, users_df_raw = load_domain_data(domain, dataset_scale=dataset_scale)

    results_by_seed = {
        "plain_lightgcn": [],
        "plain_sgl": [],
        "tce_sgdl": [],
        "trust_grs": [],
        "uncalibrated": [],
        "vanilla_clean": [],
        "vanilla_attacked": [],
        "robust": []
    }
    
    roc_pr_list = []
    
    for seed in seeds:
        print(f"\n--- Running Seed {seed} on {domain.upper()} ---")
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        # 1. Ingest Data Split
        clean_split = preprocess_dataset(ratings_df_raw.copy(), items_df_raw.copy(), users_df_raw.copy())
        
        if domain == "movielens":
            backbone_name = "NeuMF"
            warm_epochs = 4
            dual_epochs = 15
            is_contrastive = False
            mirf_alpha, mirf_beta, mirf_gamma = 0.55, 0.00, 0.45
            
        elif domain == "amazon":
            backbone_name = "SimGCL"
            warm_epochs = 4
            dual_epochs = 8
            is_contrastive = True
            mirf_alpha, mirf_beta, mirf_gamma = 0.40, 0.30, 0.30

        elif domain == "yelp":
            backbone_name = "SGL"
            warm_epochs = 4
            dual_epochs = 8
            is_contrastive = True
            mirf_alpha, mirf_beta, mirf_gamma = 0.40, 0.30, 0.30

        # 2. Attack Simulation
        attack_sim = AttackSimulator(seed=seed)
        # Select target tail items explicitly
        sorted_items = sorted(clean_split.item_counts.items(), key=lambda x: -x[1])
        all_item_ids = [i for i, _ in sorted_items]
        tail_pool = all_item_ids[-min(50, len(all_item_ids)):]
        target_items = list(np.random.choice(tail_pool, size=min(5, len(tail_pool)), replace=False)) if tail_pool else []
        
        attacked_split = attack_sim.inject_attack(clean_split, attack_type="bandwagon", noise_rate=0.15, target_items=target_items)
        target_set = set(target_items)
        
        def model_factory():
            if backbone_name == "NeuMF":
                return NeuMF(attacked_split.num_users, attacked_split.num_items, embedding_dim=32, mlp_layers=[64, 32])
            elif backbone_name == "SimGCL":
                m = SimGCL(attacked_split.num_users, attacked_split.num_items, embedding_dim=32, num_layers=2, noise_eps=0.15)
                m.set_adjacency(attacked_split.train_dict, device=device)
                return m
            elif backbone_name == "SGL":
                m = SGL(attacked_split.num_users, attacked_split.num_items, embedding_dim=32, num_layers=2, drop_rate=0.10, augment_type="ED")
                m.set_adjacency(attacked_split.train_dict, device=device)
                return m

        # 3. Warm start
        trainer = DualModelTrainer(
            model_factory=model_factory,
            device=device,
            lr=0.002,
            delta_T_reg=0.01,
            epochs=dual_epochs,
            early_stopping_patience=6,
            use_contrastive=is_contrastive,
            lambda_cl=0.15,
            temperature_cl=0.15
        )
        warm_model = model_factory().to(device)
        unweighted = {k: 1.0 for k in attacked_split.weight_dict}
        warm_loader = build_data_loader(attacked_split.train_dict, unweighted, batch_size=1024)
        trainer.warm_train(warm_model, warm_loader, warm_epochs=warm_epochs)
        
        # 4. Anchor selection & transition matrix
        anchor_sel = AnchorSelector(anchor_percentile=0.85, min_anchors_per_class=5)
        anchors = anchor_sel.select_anchors(warm_model, warm_loader, device=device)
        trans_module = NoiseTransitionMatrix(num_classes=5).to(device)
        trans_module.estimate_from_anchors(anchors)
        
        # 5. MIRF Fusion
        R_RRFN = compute_R_RRFN(warm_model, warm_loader, device=device)
        prompt_builder = LLMPromptBuilder(clean_split.items_df)
        llm_auditor = LLMAuditor(prompt_builder=prompt_builder, provider="mock", prefilter_threshold=0.60)
        R_LLM = llm_auditor.audit_all(attacked_split.train_dict, R_RRFN, ground_truth_labels=attacked_split.ground_truth_labels, user_mean_ratings=attacked_split.user_mean_ratings)
        accel = compute_temporal_acceleration(attacked_split.train_dict, time_window_hours=24)
        polarity = compute_polarity_skew(attacked_split.train_dict, time_window_hours=24)
        sim_sem = compute_semantic_similarity(attacked_split.train_dict)
        R_bomb = compute_bomb_scores(accel, polarity, sim_sem, omega_1=0.60, omega_2=0.40, omega_3=0.0)
        fused_weights = fuse_reliability_scores(R_RRFN, R_LLM, R_bomb, alpha=mirf_alpha, beta=mirf_beta, gamma=mirf_gamma, min_weight=0.02)
        
        roc_pr = compute_roc_pr_data(attacked_split.ground_truth_labels, fused_weights)
        roc_pr_list.append(roc_pr)
        
        # 6. Pareto catalog partition
        V_eff_vanilla = compute_effective_volume(attacked_split.train_dict, unweighted, attacked_split.num_items)
        H_vanilla, T_vanilla = pareto_partition(V_eff_vanilla, 0.20)
        V_eff_robust = compute_effective_volume(attacked_split.train_dict, fused_weights, attacked_split.num_items)
        H_robust, T_robust = pareto_partition(V_eff_robust, 0.20)
        
        # 7. Decoupled dual models
        pop_loader_rob = build_data_loader(attacked_split.train_dict, fused_weights, item_filter=H_robust, batch_size=1024)
        tail_loader_rob = build_data_loader(attacked_split.train_dict, fused_weights, item_filter=T_robust, batch_size=1024)
        M_pop_rob, _ = trainer.train_single_model(pop_loader_rob, attacked_split.val_dict, H_robust, T_hat=trans_module.T_hat)
        M_tail_rob, _ = trainer.train_single_model(tail_loader_rob, attacked_split.val_dict, T_robust, T_hat=trans_module.T_hat)

        pop_loader_van = build_data_loader(attacked_split.train_dict, unweighted, item_filter=H_vanilla, batch_size=1024)
        tail_loader_van = build_data_loader(attacked_split.train_dict, unweighted, item_filter=T_vanilla, batch_size=1024)
        M_pop_van, _ = trainer.train_single_model(pop_loader_van, attacked_split.val_dict, H_vanilla, T_hat=None)
        M_tail_van, _ = trainer.train_single_model(tail_loader_van, attacked_split.val_dict, T_vanilla, T_hat=None)

        # 8. Blueprints & Inclinations
        clean_inclinations = compute_robust_inclination(clean_split.train_dict, H_robust, unweighted, shrinkage_tau=0.0)
        vanilla_inclinations = compute_robust_inclination(attacked_split.train_dict, H_vanilla, unweighted, shrinkage_tau=0.0)
        robust_inclinations = compute_robust_inclination(attacked_split.train_dict, H_robust, fused_weights, shrinkage_tau=5.0)

        denoised_blueprints = build_denoised_blueprints(attacked_split.train_dict, fused_weights, attacked_split.user_mean_ratings, H_robust, filter_threshold=0.30)
        vanilla_blueprints = build_denoised_blueprints(attacked_split.train_dict, unweighted, attacked_split.user_mean_ratings, H_vanilla, filter_threshold=0.0)
        clean_blueprints = build_denoised_blueprints(clean_split.train_dict, unweighted, clean_split.user_mean_ratings, H_robust, filter_threshold=0.0)

        # 9. Evaluation configuration
        top_k = 10
        all_test_users = list(clean_split.test_dict.keys())
        if eval_users_limit is not None:
            eval_users = all_test_users[:min(eval_users_limit, len(all_test_users))]
        elif dataset_scale == "sample":
            eval_users = all_test_users[:min(150, len(all_test_users))]
        else:
            eval_users = all_test_users
        all_items_set = set(range(clean_split.num_items))
        rng = random.Random(seed)

        # --- Competitive Monolithic Baselines under rho = 0.15 attack ---
        full_unweighted_loader = build_data_loader(attacked_split.train_dict, unweighted, batch_size=1024)

        # 1. Plain LightGCN
        m_lgcn = LightGCN(attacked_split.num_users, attacked_split.num_items, embedding_dim=32, num_layers=3).to(device)
        m_lgcn.set_adjacency(attacked_split.train_dict, device=device)
        trainer_lgcn = BaselineTrainer(device=device, lr=0.002, epochs=dual_epochs, baseline_type="standard")
        m_lgcn = trainer_lgcn.train(m_lgcn, full_unweighted_loader, attacked_split.val_dict)
        recs_lgcn = BaselineTrainer.generate_recommendations(m_lgcn, eval_users, clean_split, top_k=top_k, rng=rng, device=device)

        # 2. Plain SGL
        m_sgl = SGL(attacked_split.num_users, attacked_split.num_items, embedding_dim=32, num_layers=2, drop_rate=0.10, augment_type="ED").to(device)
        m_sgl.set_adjacency(attacked_split.train_dict, device=device)
        trainer_sgl = BaselineTrainer(device=device, lr=0.002, epochs=dual_epochs, baseline_type="sgl", lambda_cl=0.15, temperature_cl=0.15)
        m_sgl = trainer_sgl.train(m_sgl, full_unweighted_loader, attacked_split.val_dict)
        recs_sgl = BaselineTrainer.generate_recommendations(m_sgl, eval_users, clean_split, top_k=top_k, rng=rng, device=device)

        # 3. T-CE / SGDL
        m_tce = TCE_SGDL(attacked_split.num_users, attacked_split.num_items, embedding_dim=32, num_layers=3, num_classes=5, tau_loss=1.0, delta_T_reg=0.01).to(device)
        m_tce.set_adjacency(attacked_split.train_dict, device=device)
        trainer_tce = BaselineTrainer(device=device, lr=0.002, epochs=dual_epochs, baseline_type="tce_sgdl", warm_epochs=warm_epochs)
        m_tce = trainer_tce.train(m_tce, full_unweighted_loader, attacked_split.val_dict, anchor_selector=anchor_sel)
        recs_tce = BaselineTrainer.generate_recommendations(m_tce, eval_users, clean_split, top_k=top_k, rng=rng, device=device)

        # Extract weights for Denoising-ROC-AUC calculation for T-CE / SGDL in vectorized batches
        tce_weights = {}
        m_tce.eval()
        with torch.no_grad():
            all_u, all_i, all_r = [], [], []
            for u, inters in attacked_split.train_dict.items():
                for item, r, _ in inters:
                    all_u.append(u)
                    all_i.append(item)
                    all_r.append(r)
            if all_u:
                chunk_size = 4096
                for idx in range(0, len(all_u), chunk_size):
                    cu = torch.tensor(all_u[idx:idx + chunk_size], dtype=torch.long, device=device)
                    ci = torch.tensor(all_i[idx:idx + chunk_size], dtype=torch.long, device=device)
                    cr = torch.tensor(all_r[idx:idx + chunk_size], dtype=torch.long, device=device)
                    _, w_chunk = m_tce.compute_loss(cu, ci, cr, return_weights=True)
                    w_arr = w_chunk.cpu().numpy()
                    for j, w_val in enumerate(w_arr):
                        tce_weights[(all_u[idx + j], all_i[idx + j])] = float(w_val)

        # 4. Trust-GRS
        m_trust = TrustGRS(attacked_split.num_users, attacked_split.num_items, embedding_dim=32, num_layers=3, eps_rating=0.01, tau_var=0.5, eps_min=0.001).to(device)
        trust_weights = m_trust.compute_trust_weights(attacked_split.train_dict, head_items=H_robust)
        trust_loader = build_data_loader(attacked_split.train_dict, trust_weights, batch_size=1024)
        trainer_trust = BaselineTrainer(device=device, lr=0.002, epochs=dual_epochs, baseline_type="trust_grs")
        m_trust = trainer_trust.train(m_trust, trust_loader, attacked_split.val_dict, train_dict=attacked_split.train_dict, head_items=H_robust)
        recs_trust = BaselineTrainer.generate_recommendations(m_trust, eval_users, clean_split, top_k=top_k, rng=rng, device=device)

        # 10. Evaluation loops for decoupled and baseline models
        recs_robust, recs_vanilla, recs_uncalib, recs_clean = {}, {}, {}, {}
        
        for u in eval_users:
            test_item, _, _ = clean_split.test_dict[u]
            seen = set(item for item, _, _ in clean_split.train_dict.get(u, [])) | {test_item}
            unseen = list(all_items_set - seen)
            negs = rng.sample(unseen, min(99, len(unseen)))
            candidates = [test_item] + negs

            # Robust candidate scoring
            cand_h_rob = [i for i in candidates if i in H_robust]
            cand_t_rob = [i for i in candidates if i in T_robust]
            s_p_rob = M_pop_rob.score_items(u, cand_h_rob, device=device) if cand_h_rob else torch.tensor([])
            s_t_rob = M_tail_rob.score_items(u, cand_t_rob, device=device) if cand_t_rob else torch.tensor([])
            p_c_rob = [cand_h_rob[i] for i in torch.topk(s_p_rob, k=len(cand_h_rob)).indices.cpu().numpy()] if len(cand_h_rob) > 0 else []
            t_c_rob = [cand_t_rob[i] for i in torch.topk(s_t_rob, k=len(cand_t_rob)).indices.cpu().numpy()] if len(cand_t_rob) > 0 else []

            # Vanilla candidate scoring
            cand_h_van = [i for i in candidates if i in H_vanilla]
            cand_t_van = [i for i in candidates if i in T_vanilla]
            s_p_van = M_pop_van.score_items(u, cand_h_van, device=device) if cand_h_van else torch.tensor([])
            s_t_van = M_tail_van.score_items(u, cand_t_van, device=device) if cand_t_van else torch.tensor([])
            p_c_van = [cand_h_van[i] for i in torch.topk(s_p_van, k=len(cand_h_van)).indices.cpu().numpy()] if len(cand_h_van) > 0 else []
            t_c_van = [cand_t_van[i] for i in torch.topk(s_t_van, k=len(cand_t_van)).indices.cpu().numpy()] if len(cand_t_van) > 0 else []

            # Uncalibrated scoring
            s_all = warm_model.score_items(u, candidates, device=device)
            u_sorted = [candidates[i] for i in torch.topk(s_all, k=min(top_k, len(candidates))).indices.cpu().numpy()]
            recs_uncalib[u] = u_sorted

            _, b_u_rob   = denoised_blueprints.get(u, ([], []))
            _, b_u_van   = vanilla_blueprints.get(u, ([], []))
            _, b_u_clean = clean_blueprints.get(u, ([], []))
            c_u_rob   = robust_inclinations.get(u, 0.20)
            c_u_van   = vanilla_inclinations.get(u, 0.20)
            c_u_clean = clean_inclinations.get(u, 0.20)

            recs_robust[u] = merge_top_n(u, p_c_rob, t_c_rob, c_u_rob, b_u_rob, top_k=top_k)
            recs_vanilla[u] = merge_top_n(u, p_c_van, t_c_van, c_u_van, b_u_van, top_k=top_k)
            recs_clean[u]   = merge_top_n(u, p_c_rob, t_c_rob, c_u_clean, b_u_clean, top_k=top_k)

        # 11. Metrics evaluation
        m_rob = evaluate_recommendations(recs_robust, clean_split.test_dict, attacked_split.train_dict, H_robust, T_robust, robust_inclinations, top_k=top_k, target_items=target_set)
        m_van = evaluate_recommendations(recs_vanilla, clean_split.test_dict, attacked_split.train_dict, H_vanilla, T_vanilla, vanilla_inclinations, top_k=top_k, target_items=target_set)
        m_cln = evaluate_recommendations(recs_clean, clean_split.test_dict, clean_split.train_dict, H_robust, T_robust, clean_inclinations, top_k=top_k, target_items=target_set)
        m_unc = evaluate_recommendations(recs_uncalib, clean_split.test_dict, attacked_split.train_dict, H_robust, T_robust, clean_inclinations, top_k=top_k, target_items=target_set)

        rob_extra = compute_robustness_metrics(m_rob, m_cln, attacked_split.ground_truth_labels, fused_weights, recommendations=recs_robust, top_k=top_k)
        van_extra = compute_robustness_metrics(m_van, m_cln, attacked_split.ground_truth_labels, unweighted, recommendations=recs_vanilla, top_k=top_k)
        if "Denoising-ROC-AUC" in rob_extra:
            rob_extra["Denoising Detection ROC-AUC"] = rob_extra["Denoising-ROC-AUC"]
            rob_extra["Denoising-ROC"] = rob_extra["Denoising-ROC-AUC"]
        if "Denoising-ROC-AUC" in van_extra:
            van_extra["Denoising Detection ROC-AUC"] = van_extra["Denoising-ROC-AUC"]
            van_extra["Denoising-ROC"] = van_extra["Denoising-ROC-AUC"]
        m_rob.update(rob_extra)
        m_van.update(van_extra)

        results_by_seed["robust"].append(m_rob)
        results_by_seed["vanilla_attacked"].append(m_van)
        results_by_seed["vanilla_clean"].append(m_cln)
        results_by_seed["uncalibrated"].append(m_unc)

        # Evaluate metrics for all 4 competitive baselines
        baseline_recs_map = {
            "plain_lightgcn": (recs_lgcn, unweighted),
            "plain_sgl": (recs_sgl, unweighted),
            "tce_sgdl": (recs_tce, tce_weights),
            "trust_grs": (recs_trust, trust_weights)
        }

        for b_key, (b_recs, b_w) in baseline_recs_map.items():
            m_base = evaluate_recommendations(b_recs, clean_split.test_dict, attacked_split.train_dict, H_robust, T_robust, clean_inclinations, top_k=top_k, target_items=target_set)
            extra_base = compute_robustness_metrics(m_base, m_cln, attacked_split.ground_truth_labels, b_w, recommendations=b_recs, top_k=top_k)
            if "Denoising-ROC-AUC" in extra_base:
                extra_base["Denoising Detection ROC-AUC"] = extra_base["Denoising-ROC-AUC"]
                extra_base["Denoising-ROC"] = extra_base["Denoising-ROC-AUC"]
            m_base.update(extra_base)
            results_by_seed[b_key].append(m_base)

        # Memory cleanup between seeds
        del m_lgcn, m_sgl, m_tce, m_trust, M_pop_rob, M_tail_rob, M_pop_van, M_tail_van
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Compute Statistics across seeds
    all_model_keys = ["plain_lightgcn", "plain_sgl", "tce_sgdl", "trust_grs", "uncalibrated", "vanilla_clean", "vanilla_attacked", "robust"]
    summary_stats = {}
    for model_key in all_model_keys:
        summary_stats[model_key] = {}
        if not results_by_seed[model_key]:
            continue
        all_metrics_keys = results_by_seed[model_key][0].keys()
        for k in all_metrics_keys:
            vals = [res[k] for res in results_by_seed[model_key] if isinstance(res.get(k), (int, float))]
            if vals:
                summary_stats[model_key][k] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals))
                }

    # Significance testing between Robust and Baselines / Attacked Vanilla
    p_values = {}
    for comp_key in ["plain_lightgcn", "plain_sgl", "tce_sgdl", "trust_grs", "vanilla_attacked"]:
        p_values[f"vs_{comp_key}"] = {}
        if comp_key not in results_by_seed or not results_by_seed[comp_key]:
            continue
        for k in summary_stats["robust"].keys():
            r_vals = [res[k] for res in results_by_seed["robust"] if isinstance(res.get(k), (int, float))]
            c_vals = [res[k] for res in results_by_seed[comp_key] if isinstance(res.get(k), (int, float))]
            if len(r_vals) == len(c_vals) and len(r_vals) > 1:
                try:
                    _, p_val = stats.ttest_rel(r_vals, c_vals)
                    p_values[f"vs_{comp_key}"][k] = float(p_val) if not np.isnan(p_val) else 1.0
                except:
                    p_values[f"vs_{comp_key}"][k] = 1.0

    # Populate top-level p_values for vs_vanilla_attacked backward compatibility
    for k, val in p_values.get("vs_vanilla_attacked", {}).items():
        p_values[k] = val

    print(f"\n--- {domain.upper()} Multi-Seed Summary (Mean +- Std) ---")
    header_str = f"{'Metric':<24} | {'Robust':<18} | {'Vanilla Att':<18} | {'LightGCN':<18} | {'SGL':<18} | {'T-CE/SGDL':<18} | {'Trust-GRS':<18}"
    print(header_str)
    print("-" * len(header_str))

    display_keys = ["Recall@10", "nDCG@10", "RMSE-PC", "MRMC", "LTC@10", "g-LTC@10", "STER@10", "Novelty", "GKPI", "Denoising-ROC-AUC"]
    for k in display_keys:
        def fmt_m_s(m_key):
            if m_key in summary_stats and k in summary_stats[m_key]:
                m = summary_stats[m_key][k]["mean"]
                s = summary_stats[m_key][k]["std"]
                return f"{m:.4f} +- {s:.4f}"
            return "N/A"

        r_str = fmt_m_s("robust")
        v_str = fmt_m_s("vanilla_attacked")
        lgcn_str = fmt_m_s("plain_lightgcn")
        sgl_str = fmt_m_s("plain_sgl")
        tce_str = fmt_m_s("tce_sgdl")
        trust_str = fmt_m_s("trust_grs")
        print(f"{k:<24} | {r_str:<18} | {v_str:<18} | {lgcn_str:<18} | {sgl_str:<18} | {tce_str:<18} | {trust_str:<18}")

    output_pkg = {
        "domain": domain,
        "seeds": seeds,
        "summary_stats": summary_stats,
        "p_values": p_values,
        "raw_per_seed": results_by_seed,
        "mean_roc_auc": float(np.mean([x["roc_auc"] for x in roc_pr_list])) if roc_pr_list else 0.0,
        "mean_avg_precision": float(np.mean([x["avg_precision"] for x in roc_pr_list])) if roc_pr_list else 0.0
    }

    return output_pkg


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Multi-Seed Multi-Domain Benchmark")
    parser.add_argument("--dataset", default="full", choices=["full", "sample"], help="full|sample")
    parser.add_argument("--domains", default="movielens,amazon,yelp", help="Comma-separated domains to run")
    parser.add_argument("--seeds", default="42,43,44,45,46", help="Comma-separated seeds")
    parser.add_argument("--eval_users", default=None, type=int, help="Limit number of eval users per domain")
    parser.add_argument("--device", default=None, help="cuda|cpu override")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    domain_list = [d.strip().lower() for d in args.domains.split(",") if d.strip()]

    all_multiseed = {}
    for dom in domain_list:
        out = run_single_domain_multiseed(dom, seeds=seeds, device=device, dataset_scale=args.dataset, eval_users_limit=args.eval_users)
        all_multiseed[dom] = out
    
    out_file = os.path.join(ROBUST_DIR, "results", "multiseed_benchmark.json")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(all_multiseed, f, indent=4)
        
    print(f"\nAll multi-seed experiments complete! Results saved to {out_file}")
