import os
import sys
import yaml
import torch
import numpy as np

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
from src.evaluation.robustness_metrics import compute_robustness_metrics
from src.evaluation.reporter import ExperimentReporter

def main():
    config_path = os.path.join(os.path.dirname(__file__), "../configs/default_config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() and cfg["training"]["device"] == "cuda" else "cpu"
    print(f"[*] Running Full RRFN-LLM-SUPER Pipeline on device: {device}")

    # Load Clean Data
    loader = MovieLensLoader(
        data_dir=cfg["data"]["raw_dir"],
        min_user_interactions=cfg["data"]["min_user_interactions"],
        min_item_interactions=cfg["data"]["min_item_interactions"]
    )
    ratings_df, movies_df, users_df = loader.load_data()
    clean_split = preprocess_dataset(ratings_df, movies_df, users_df)

    attacks = ["random_flip", "bandwagon", "nuke_bomb", "agas"]
    noise_rates = [0.05, 0.10]
    top_k = cfg["model"]["top_k"]

    reporter = ExperimentReporter(output_dir=cfg["evaluation"]["report_dir"])
    simulator = AttackSimulator(seed=cfg["data"]["seed"])
    prompt_builder = LLMPromptBuilder(movies_df)

    for attack_type in attacks:
        for noise_rate in noise_rates:
            print(f"\n{'='*60}\n[*] Running RRFN-LLM-SUPER: {attack_type} (rate={noise_rate})\n{'='*60}")

            # 1. Attack Simulation
            attacked_split = simulator.inject_attack(clean_split, attack_type, noise_rate)

            # 2. Model Factory
            def model_factory():
                backbone_type = cfg["model"]["backbone"]
                if backbone_type == "neumf":
                    return NeuMF(attacked_split.num_users, attacked_split.num_items, cfg["model"]["embedding_dim"])
                elif backbone_type == "lightgcn":
                    return LightGCN(attacked_split.num_users, attacked_split.num_items, cfg["model"]["embedding_dim"])
                else:
                    return VaeCF(attacked_split.num_users, attacked_split.num_items, cfg["model"]["embedding_dim"])

            # 3. Warm-Start Training (Initial representation learning)
            print("[*] Stage 1: Warm-start representation learning...")
            warm_model = model_factory().to(device)
            initial_weights = {k: 1.0 for k in attacked_split.weight_dict}
            warm_loader = build_data_loader(attacked_split.train_dict, initial_weights, batch_size=cfg["training"]["batch_size"])

            trainer = DualModelTrainer(
                model_factory=model_factory,
                device=device,
                lr=cfg["training"]["learning_rate"],
                epochs=min(15, cfg["training"]["epochs"]),
                early_stopping_patience=cfg["training"]["early_stopping_patience"],
                delta_T_reg=cfg["rrfn"]["delta_T_reg"]
            )
            trainer.warm_train(warm_model, warm_loader, warm_epochs=cfg["rrfn"]["warm_epochs"])

            # 4. RRFN Anchor Selection & Noise Transition Matrix
            print("[*] Stage 2: RRFN anchor selection and transition matrix estimation...")
            anchor_sel = AnchorSelector(
                anchor_percentile=cfg["rrfn"]["anchor_percentile"],
                min_anchors_per_class=cfg["rrfn"]["min_anchors_per_class"]
            )
            anchors = anchor_sel.select_anchors(warm_model, warm_loader, device=device)
            trans_module = NoiseTransitionMatrix(num_classes=5).to(device)
            trans_module.estimate_from_anchors(anchors)
            T_hat = trans_module.T_hat

            # 5. Compute Reliability Signals
            print("[*] Stage 3: Computing multi-view reliability signals...")
            # (a) Statistical signal: R_RRFN
            R_RRFN = compute_R_RRFN(warm_model, warm_loader, device=device)

            # (b) Semantic signal: R_LLM (with pre-filtering)
            llm_auditor = LLMAuditor(
                prompt_builder=prompt_builder,
                provider=cfg["llm"]["provider"],
                model_name=cfg["llm"]["model_name"],
                cache_path=cfg["llm"]["cache_path"],
                prefilter_threshold=cfg["llm"]["prefilter_threshold"]
            )
            R_LLM = llm_auditor.audit_all(
                attacked_split.train_dict,
                R_RRFN,
                ground_truth_labels=attacked_split.ground_truth_labels,
                user_mean_ratings=attacked_split.user_mean_ratings
            )

            # (c) Behavioral signal: R_bomb
            accel = compute_temporal_acceleration(attacked_split.train_dict, time_window_hours=cfg["bombing_detector"]["time_window_hours"])
            polarity = compute_polarity_skew(attacked_split.train_dict, time_window_hours=cfg["bombing_detector"]["time_window_hours"])
            semantic_sim = compute_semantic_similarity(attacked_split.train_dict)
            R_bomb = compute_bomb_scores(
                accel, polarity, semantic_sim,
                omega_1=cfg["bombing_detector"]["omega_1"],
                omega_2=cfg["bombing_detector"]["omega_2"],
                omega_3=cfg["bombing_detector"]["omega_3"],
                sigmoid_bias=cfg["bombing_detector"]["sigmoid_bias"]
            )

            # 6. Multi-View Reliability Fusion
            fused_weights = fuse_reliability_scores(
                R_RRFN, R_LLM, R_bomb,
                alpha=cfg["fusion"]["alpha"],
                beta=cfg["fusion"]["beta"],
                gamma=cfg["fusion"]["gamma"],
                min_weight=cfg["fusion"]["min_weight"]
            )

            # 7. Reliability-Weighted Pareto Catalog Partitioning
            print("[*] Stage 4: Robust Pareto catalog partitioning...")
            V_eff = compute_effective_volume(attacked_split.train_dict, fused_weights, attacked_split.num_items)
            head_items, tail_items = pareto_partition(V_eff, cfg["catalog"]["pareto_alpha"])
            print(f"[*] Robust Split: Head = {len(head_items)} items, Tail = {len(tail_items)} items.")

            # 8. Decoupled Dual Training with Risk-Consistent Loss
            print("[*] Stage 5: Decoupled dual training with risk-consistent loss...")
            pop_loader = build_data_loader(attacked_split.train_dict, fused_weights, item_filter=head_items, batch_size=cfg["training"]["batch_size"])
            tail_loader = build_data_loader(attacked_split.train_dict, fused_weights, item_filter=tail_items, batch_size=cfg["training"]["batch_size"])

            M_pop, _ = trainer.train_single_model(pop_loader, attacked_split.val_dict, head_items, T_hat=T_hat)
            M_tail, _ = trainer.train_single_model(tail_loader, attacked_split.val_dict, tail_items, T_hat=T_hat)

            # 9. Robust Calibration & Blueprint Generation
            print("[*] Stage 6: Robust inclination & denoised blueprint generation...")
            inclinations = compute_robust_inclination(
                attacked_split.train_dict, head_items, fused_weights,
                shrinkage_tau=cfg["super"]["shrinkage_tau"],
                global_head_prior=cfg["super"]["global_head_prior"]
            )
            blueprints = build_denoised_blueprints(
                attacked_split.train_dict, fused_weights, attacked_split.user_mean_ratings, head_items,
                filter_threshold=cfg["super"]["blueprint_filter_threshold"]
            )

            # 10. Top-N Merging & Evaluation
            recommendations = {}
            head_list = list(head_items)
            tail_list = list(tail_items)

            eval_users = list(clean_split.test_dict.keys())
            for u in eval_users:
                scores_pop = M_pop.score_items(u, head_list, device=device)
                scores_tail = M_tail.score_items(u, tail_list, device=device)

                pop_sorted_idx = torch.topk(scores_pop, k=min(top_k, len(head_list))).indices.cpu().numpy()
                tail_sorted_idx = torch.topk(scores_tail, k=min(top_k, len(tail_list))).indices.cpu().numpy()

                pop_cands = [head_list[idx] for idx in pop_sorted_idx]
                tail_cands = [tail_list[idx] for idx in tail_sorted_idx]

                _, b_u = blueprints.get(u, ([], []))
                rec = merge_top_n(u, pop_cands, tail_cands, inclinations.get(u, 0.2), b_u, top_k=top_k)
                recommendations[u] = rec

            metrics = evaluate_recommendations(
                recommendations, clean_split.test_dict, attacked_split.train_dict,
                head_items, tail_items, inclinations, top_k=top_k
            )

            robustness = compute_robustness_metrics(
                metrics, None, attacked_split.ground_truth_labels, fused_weights,
                recommendations=recommendations, top_k=top_k
            )

            reporter.record(
                method="RRFN-LLM-SUPER (Full)",
                backbone=cfg["model"]["backbone"],
                attack_type=attack_type,
                noise_rate=noise_rate,
                metrics=metrics,
                robustness=robustness
            )

    reporter.print_summary()
    reporter.export_csv("robust_super_final_results.csv")
    reporter.export_latex("robust_super_results_table.tex")

if __name__ == "__main__":
    main()
