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
    print(f"[*] Running Vanilla SUPER Under Attack Benchmark on device: {device}")

    # Load Clean Data
    loader = MovieLensLoader(
        data_dir=cfg["data"]["raw_dir"],
        min_user_interactions=cfg["data"]["min_user_interactions"],
        min_item_interactions=cfg["data"]["min_item_interactions"]
    )
    ratings_df, movies_df, users_df = loader.load_data()
    clean_split = preprocess_dataset(ratings_df, movies_df, users_df)

    # Attack suite to evaluate
    attacks = ["random_flip", "bandwagon", "nuke_bomb", "agas"]
    noise_rates = [0.05, 0.10]
    top_k = cfg["model"]["top_k"]

    reporter = ExperimentReporter(output_dir=cfg["evaluation"]["report_dir"])
    simulator = AttackSimulator(seed=cfg["data"]["seed"])

    for attack_type in attacks:
        for noise_rate in noise_rates:
            print(f"\n{'='*50}\n[*] Evaluating Attack: {attack_type} (rate={noise_rate})\n{'='*50}")

            # 1. Inject Attack
            attacked_split = simulator.inject_attack(clean_split, attack_type, noise_rate)

            # 2. Vanilla Pareto Partition (Unweighted counts)
            unweighted_weights = {k: 1.0 for k in attacked_split.weight_dict}
            V_eff = compute_effective_volume(attacked_split.train_dict, unweighted_weights, attacked_split.num_items)
            head_items, tail_items = pareto_partition(V_eff, cfg["catalog"]["pareto_alpha"])

            # 3. Model Factory
            def model_factory():
                backbone_type = cfg["model"]["backbone"]
                if backbone_type == "neumf":
                    return NeuMF(attacked_split.num_users, attacked_split.num_items, cfg["model"]["embedding_dim"])
                elif backbone_type == "lightgcn":
                    return LightGCN(attacked_split.num_users, attacked_split.num_items, cfg["model"]["embedding_dim"])
                else:
                    return VaeCF(attacked_split.num_users, attacked_split.num_items, cfg["model"]["embedding_dim"])

            # 4. Decoupled Dual Training (Vanilla)
            trainer = DualModelTrainer(
                model_factory=model_factory,
                device=device,
                lr=cfg["training"]["learning_rate"],
                epochs=min(15, cfg["training"]["epochs"]),
                early_stopping_patience=cfg["training"]["early_stopping_patience"]
            )

            pop_loader = build_data_loader(attacked_split.train_dict, unweighted_weights, item_filter=head_items, batch_size=cfg["training"]["batch_size"])
            tail_loader = build_data_loader(attacked_split.train_dict, unweighted_weights, item_filter=tail_items, batch_size=cfg["training"]["batch_size"])

            M_pop, _ = trainer.train_single_model(pop_loader, attacked_split.val_dict, head_items)
            M_tail, _ = trainer.train_single_model(tail_loader, attacked_split.val_dict, tail_items)

            # 5. Vanilla Calibration & Top-N Merge
            inclinations = compute_robust_inclination(
                attacked_split.train_dict, head_items, unweighted_weights,
                shrinkage_tau=0.0, global_head_prior=0.20 # no shrinkage in vanilla
            )
            blueprints = build_denoised_blueprints(
                attacked_split.train_dict, unweighted_weights, attacked_split.user_mean_ratings, head_items,
                filter_threshold=0.0 # no filtering in vanilla
            )

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

            # 6. Evaluation
            metrics = evaluate_recommendations(
                recommendations, clean_split.test_dict, attacked_split.train_dict,
                head_items, tail_items, inclinations, top_k=top_k
            )

            reporter.record(
                method="Vanilla SUPER (Uncorrected)",
                backbone=cfg["model"]["backbone"],
                attack_type=attack_type,
                noise_rate=noise_rate,
                metrics=metrics
            )

    reporter.print_summary()
    reporter.export_csv("attack_vulnerability_results.csv")

if __name__ == "__main__":
    main()
