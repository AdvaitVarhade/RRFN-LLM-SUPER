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

def run_single_ablation(
    variant_name: str,
    attacked_split,
    clean_split,
    prompt_builder,
    cfg,
    device,
    target_items,
    attack_type="bandwagon",
    noise_rate=0.10
):
    print(f"\n---> Running Ablation Variant: {variant_name}")
    top_k = cfg["model"]["top_k"]

    def model_factory():
        return NeuMF(attacked_split.num_users, attacked_split.num_items, cfg["model"]["embedding_dim"])

    # 1. Warm start
    trainer = DualModelTrainer(
        model_factory=model_factory,
        device=device,
        lr=cfg["training"]["learning_rate"],
        epochs=min(10, cfg["training"]["epochs"]),
        early_stopping_patience=cfg["training"]["early_stopping_patience"]
    )
    warm_model = model_factory().to(device)
    initial_weights = {k: 1.0 for k in attacked_split.weight_dict}
    warm_loader = build_data_loader(attacked_split.train_dict, initial_weights, batch_size=cfg["training"]["batch_size"])
    trainer.warm_train(warm_model, warm_loader, warm_epochs=cfg["rrfn"]["warm_epochs"])

    # 2. RRFN Signal
    if variant_name == "A-noRRFN":
        R_RRFN = {k: 0.50 for k in initial_weights}
        T_hat = None
    else:
        anchor_sel = AnchorSelector(anchor_percentile=cfg["rrfn"]["anchor_percentile"], min_anchors_per_class=cfg["rrfn"]["min_anchors_per_class"])
        anchors = anchor_sel.select_anchors(warm_model, warm_loader, device=device)
        trans_module = NoiseTransitionMatrix(num_classes=5).to(device)
        trans_module.estimate_from_anchors(anchors)
        T_hat = trans_module.T_hat
        R_RRFN = compute_R_RRFN(warm_model, warm_loader, device=device)

    # 3. LLM Signal
    if variant_name == "A-noLLM":
        R_LLM = {k: 0.50 for k in initial_weights}
    else:
        llm_auditor = LLMAuditor(
            prompt_builder=prompt_builder,
            provider=cfg["llm"]["provider"],
            model_name=cfg["llm"]["model_name"],
            cache_path=cfg["llm"]["cache_path"],
            prefilter_threshold=cfg["llm"]["prefilter_threshold"]
        )
        R_LLM = llm_auditor.audit_all(
            attacked_split.train_dict, R_RRFN,
            ground_truth_labels=attacked_split.ground_truth_labels,
            user_mean_ratings=attacked_split.user_mean_ratings
        )

    # 4. Bombing Detector Signal
    if variant_name == "A-noBomb":
        R_bomb = {k: 0.50 for k in initial_weights}
    else:
        accel = compute_temporal_acceleration(attacked_split.train_dict, time_window_hours=cfg["bombing_detector"]["time_window_hours"])
        polarity = compute_polarity_skew(attacked_split.train_dict, time_window_hours=cfg["bombing_detector"]["time_window_hours"])
        semantic_sim = compute_semantic_similarity(attacked_split.train_dict)
        R_bomb = compute_bomb_scores(accel, polarity, semantic_sim, omega_1=cfg["bombing_detector"]["omega_1"], omega_2=cfg["bombing_detector"]["omega_2"])

    # 5. Fusion
    alpha = 0.0 if variant_name == "A-noRRFN" else cfg["fusion"]["alpha"]
    beta  = 0.0 if variant_name == "A-noLLM" else cfg["fusion"]["beta"]
    gamma = 0.0 if variant_name == "A-noBomb" else cfg["fusion"]["gamma"]
    if alpha == 0 and beta == 0 and gamma == 0:
        alpha, beta, gamma = 1.0, 0.0, 0.0

    fused_weights = fuse_reliability_scores(R_RRFN, R_LLM, R_bomb, alpha=alpha, beta=beta, gamma=gamma, min_weight=cfg["fusion"]["min_weight"])

    # 6. Pareto Catalog Split
    if variant_name == "A-noWeightedPareto":
        unweighted = {k: 1.0 for k in fused_weights}
        V_eff = compute_effective_volume(attacked_split.train_dict, unweighted, attacked_split.num_items)
    else:
        V_eff = compute_effective_volume(attacked_split.train_dict, fused_weights, attacked_split.num_items)
    head_items, tail_items = pareto_partition(V_eff, cfg["catalog"]["pareto_alpha"])

    # 7. Dual Training
    pop_loader = build_data_loader(attacked_split.train_dict, fused_weights, item_filter=head_items, batch_size=cfg["training"]["batch_size"])
    tail_loader = build_data_loader(attacked_split.train_dict, fused_weights, item_filter=tail_items, batch_size=cfg["training"]["batch_size"])

    M_pop, _ = trainer.train_single_model(pop_loader, attacked_split.val_dict, head_items, T_hat=T_hat)
    M_tail, _ = trainer.train_single_model(tail_loader, attacked_split.val_dict, tail_items, T_hat=T_hat)

    # 8. Calibration & Blueprint
    shrinkage_tau = 0.0 if variant_name == "A-noShrinkage" else cfg["super"]["shrinkage_tau"]
    inclinations = compute_robust_inclination(
        attacked_split.train_dict, head_items, fused_weights,
        shrinkage_tau=shrinkage_tau,
        global_head_prior=cfg["super"]["global_head_prior"]
    )

    if variant_name == "A-rawBlueprint":
        blueprints = build_denoised_blueprints(
            attacked_split.train_dict, {k: 1.0 for k in fused_weights}, attacked_split.user_mean_ratings, head_items,
            filter_threshold=0.0
        )
    else:
        blueprints = build_denoised_blueprints(
            attacked_split.train_dict, fused_weights, attacked_split.user_mean_ratings, head_items,
            filter_threshold=cfg["super"]["blueprint_filter_threshold"]
        )

    # 9. Top-N Merging
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

    # 10. Evaluation
    metrics = evaluate_recommendations(recommendations, clean_split.test_dict, attacked_split.train_dict, head_items, tail_items, inclinations, top_k=top_k)
    robustness = compute_robustness_metrics(metrics, None, attacked_split.ground_truth_labels, fused_weights, target_items=target_items, recommendations=recommendations, top_k=top_k)

    return metrics, robustness

def main():
    config_path = os.path.join(os.path.dirname(__file__), "../configs/default_config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() and cfg["training"]["device"] == "cuda" else "cpu"
    print(f"[*] Running Ablation Study Suite on device: {device}")

    loader = MovieLensLoader(
        data_dir=cfg["data"]["raw_dir"],
        min_user_interactions=cfg["data"]["min_user_interactions"],
        min_item_interactions=cfg["data"]["min_item_interactions"]
    )
    ratings_df, movies_df, users_df = loader.load_data()
    clean_split = preprocess_dataset(ratings_df, movies_df, users_df)

    simulator = AttackSimulator(seed=cfg["data"]["seed"])
    prompt_builder = LLMPromptBuilder(movies_df)
    reporter = ExperimentReporter(output_dir=cfg["evaluation"]["report_dir"])

    attack_type = "bandwagon"
    noise_rate = 0.10
    attacked_split = simulator.inject_attack(clean_split, attack_type, noise_rate)
    target_items = [i for i, c in sorted(attacked_split.item_counts.items(), key=lambda x: x[1])[:5]]

    variants = [
        "A-full",
        "A-noLLM",
        "A-noRRFN",
        "A-noBomb",
        "A-noWeightedPareto",
        "A-rawBlueprint",
        "A-noShrinkage"
    ]

    for v in variants:
        metrics, robustness = run_single_ablation(
            v, attacked_split, clean_split, prompt_builder, cfg, device, target_items,
            attack_type=attack_type, noise_rate=noise_rate
        )
        reporter.record(
            method=f"Ablation-{v}",
            backbone=cfg["model"]["backbone"],
            attack_type=attack_type,
            noise_rate=noise_rate,
            metrics=metrics,
            robustness=robustness
        )

    reporter.print_summary()
    reporter.export_csv("ablation_study_results.csv")
    reporter.export_latex("ablation_study_table.tex")

if __name__ == "__main__":
    main()
