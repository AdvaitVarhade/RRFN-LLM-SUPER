import os
import sys
import yaml
import torch
import numpy as np

# Add robust_super root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.loader import MovieLensLoader
from src.data.preprocessor import preprocess_dataset
from src.models.neumf import NeuMF
from src.models.lightgcn import LightGCN
from src.models.vaecf import VaeCF
from src.catalog.pareto_partition import compute_effective_volume, pareto_partition
from src.trainer.dual_trainer import DualModelTrainer, build_data_loader
from src.super_engine.inclination import compute_robust_inclination
from src.super_engine.blueprint import build_denoised_blueprints
from src.super_engine.merger import merge_top_n
from src.evaluation.metrics import evaluate_recommendations
from src.evaluation.reporter import ExperimentReporter

def main():
    config_path = os.path.join(os.path.dirname(__file__), "../configs/default_config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() and cfg["training"]["device"] == "cuda" else "cpu"
    print(f"[*] Running Clean SUPER Baseline on device: {device}")

    # 1. Load Data
    loader = MovieLensLoader(
        data_dir=cfg["data"]["raw_dir"],
        min_user_interactions=cfg["data"]["min_user_interactions"],
        min_item_interactions=cfg["data"]["min_item_interactions"]
    )
    ratings_df, movies_df, users_df = loader.load_data()
    split = preprocess_dataset(ratings_df, movies_df, users_df)
    print(f"[*] Dataset: {split.num_users} users, {split.num_items} items, {sum(len(v) for v in split.train_dict.values())} train interactions.")

    # 2. Pareto Catalog Partition (Clean)
    uniform_weights = {k: 1.0 for k in split.weight_dict}
    V_eff = compute_effective_volume(split.train_dict, uniform_weights, split.num_items)
    head_items, tail_items = pareto_partition(V_eff, cfg["catalog"]["pareto_alpha"])
    print(f"[*] Pareto Partition: Head = {len(head_items)} items, Tail = {len(tail_items)} items.")

    # 3. Model Factory
    def model_factory():
        backbone_type = cfg["model"]["backbone"]
        if backbone_type == "neumf":
            return NeuMF(split.num_users, split.num_items, cfg["model"]["embedding_dim"])
        elif backbone_type == "lightgcn":
            return LightGCN(split.num_users, split.num_items, cfg["model"]["embedding_dim"])
        else:
            return VaeCF(split.num_users, split.num_items, cfg["model"]["embedding_dim"])

    # 4. Decoupled Dual Training (M_pop on Head, M_tail on Tail)
    trainer = DualModelTrainer(
        model_factory=model_factory,
        device=device,
        lr=cfg["training"]["learning_rate"],
        epochs=cfg["training"]["epochs"],
        early_stopping_patience=cfg["training"]["early_stopping_patience"]
    )

    pop_loader = build_data_loader(split.train_dict, uniform_weights, item_filter=head_items, batch_size=cfg["training"]["batch_size"])
    tail_loader = build_data_loader(split.train_dict, uniform_weights, item_filter=tail_items, batch_size=cfg["training"]["batch_size"])

    print("[*] Training Head Model (M_pop)...")
    M_pop, _ = trainer.train_single_model(pop_loader, split.val_dict, head_items)
    print("[*] Training Tail Model (M_tail)...")
    M_tail, _ = trainer.train_single_model(tail_loader, split.val_dict, tail_items)

    # 5. Calibration & Top-N Merge
    inclinations = compute_robust_inclination(
        split.train_dict, head_items, uniform_weights,
        shrinkage_tau=cfg["super"]["shrinkage_tau"],
        global_head_prior=cfg["super"]["global_head_prior"]
    )
    blueprints = build_denoised_blueprints(
        split.train_dict, uniform_weights, split.user_mean_ratings, head_items,
        filter_threshold=0.0
    )

    top_k = cfg["model"]["top_k"]
    recommendations = {}
    head_list = list(head_items)
    tail_list = list(tail_items)

    print("[*] Generating Clean Baseline Top-N recommendations...")
    eval_users = list(split.test_dict.keys())
    for u in eval_users:
        scores_pop = M_pop.score_items(u, head_list, device=device)
        scores_tail = M_tail.score_items(u, tail_list, device=device)

        pop_sorted_idx = torch.topk(scores_pop, k=min(top_k, len(head_list))).indices.cpu().numpy()
        tail_sorted_idx = torch.topk(scores_tail, k=min(top_k, len(tail_list))).indices.cpu().numpy()

        pop_cands = [head_list[idx] for idx in pop_sorted_idx]
        tail_cands = [tail_list[idx] for idx in tail_sorted_idx]

        _, b_u = blueprints[u]
        rec = merge_top_n(u, pop_cands, tail_cands, inclinations[u], b_u, top_k=top_k)
        recommendations[u] = rec

    # 6. Evaluation
    metrics = evaluate_recommendations(
        recommendations, split.test_dict, split.train_dict,
        head_items, tail_items, inclinations, top_k=top_k
    )

    reporter = ExperimentReporter(output_dir=cfg["evaluation"]["report_dir"])
    reporter.record(
        method="SUPER (Clean Baseline)",
        backbone=cfg["model"]["backbone"],
        attack_type="none",
        noise_rate=0.0,
        metrics=metrics
    )
    reporter.print_summary()
    reporter.export_csv("baseline_metrics.csv")

if __name__ == "__main__":
    main()
