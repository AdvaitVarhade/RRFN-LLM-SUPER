"""H9 Experiment: Advanced Dynamic Blueprint Merge for FedSUPER-LLM.

Implements and benchmarks novel dynamic merge algorithms:
  1. FedSUPER-Static: Baseline static integer floor merge (Yavru et al., 2026).
  2. FedSUPER-Dynamic-Stochastic: Expectation-preserving stochastic randomized quota merge.
  3. FedSUPER-Dynamic-AdaptiveAlpha: User-adaptive alpha catalog thresholding merge.
  4. FedSUPER-Dynamic-ConfidenceElastic: Score-gap aware confidence elastic merge.
  5. LLM-FedSUPER-Static: SBERT semantic fusion (lambda=0.70) + static merge.
  6. LLM-FedSUPER-Dynamic-Stochastic: SBERT semantic fusion (lambda=0.70) + stochastic quota merge.
  7. LLM-FedSUPER-Hybrid: SBERT semantic fusion (lambda=0.70) + hybrid adaptive stochastic elastic merge.

Evaluates on MovieLens-1M dataset across accuracy (Recall@K, NDCG@K), fairness (Gini, Coverage),
long-tail exposure (APLT, LTC), popularity calibration (Rmse-PC, MRMC), and composite GKPI.
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, Any, Tuple
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from recdata import load_ml1m, leave_one_out_split, build_train_matrix
from metrics import full_rank_eval, fdn_a_composite
from super import (
    pareto_partition,
    user_popularity_inclination,
    stochastic_user_quota,
    user_adaptive_alpha_partition,
    confidence_elastic_merge,
    dynamic_probabilistic_quota_merge,
    calibrated_score_fusion,
    super_blueprint_merge,
    evaluate_calibration,
    mask_user_trainpos,
    gkpi_score,
)
from train import train_federated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FedSUPER-LLM Advanced Dynamic Blueprint Merge Harness (H9)"
    )
    parser.add_argument("--k", type=int, default=20, help="Top-K ranking cut-off (default: 20)")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.20,
        help="Pareto cumulative interaction volume threshold (default: 0.20)",
    )
    parser.add_argument(
        "--rounds", type=int, default=30, help="Federated training rounds (default: 30)"
    )
    parser.add_argument(
        "--clients_per_round",
        type=int,
        default=256,
        help="Clients sampled per FL round (default: 256)",
    )
    parser.add_argument("--dim", type=int, default=64, help="Embedding dimension (default: 64)")
    parser.add_argument(
        "--lr", type=float, default=0.3, help="Client local learning rate (default: 0.3)"
    )
    parser.add_argument(
        "--llm_lambda", type=float, default=0.7, help="LLM score blend weight (default: 0.7)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for metrics and logs",
    )
    parser.add_argument(
        "--use_cached_embeddings",
        action="store_true",
        default=True,
        help="Load precomputed SBERT embeddings from H3 cache",
    )
    parser.add_argument(
        "--no_cached_embeddings",
        dest="use_cached_embeddings",
        action="store_false",
        help="Do not use cached embeddings",
    )
    parser.add_argument(
        "--use_cached_scores",
        action="store_true",
        default=False,
        help="Load cached FedNCF scores if available",
    )
    return parser.parse_args()


def load_or_compute_sbert_embeddings(
    h3_dir: str, df: Any, n_users: int, n_items: int, u_map: Dict[int, int], i_map: Dict[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """Loads precomputed SBERT embeddings from H3 cache, or computes them using Sentence-BERT."""
    item_path = os.path.join(h3_dir, "item_emb.npz")
    user_path = os.path.join(h3_dir, "user_emb.npz")

    if os.path.exists(item_path) and os.path.exists(user_path):
        print(f"Loading precomputed SBERT embeddings from {h3_dir}...")
        item_emb = np.load(item_path)["item_emb"].astype(np.float32)
        user_emb = np.load(user_path)["user_emb"].astype(np.float32)
        return user_emb, item_emb

    print("Precomputed SBERT embeddings not found. Computing via sentence-transformers...")
    from sentence_transformers import SentenceTransformer

    st_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Load movies.dat
    movies_path = os.path.join(ROOT, "data", "ml-1m", "movies.dat")
    item_texts = ["Unknown movie"] * n_items
    item_titles = [""] * n_items
    with open(movies_path, "r", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("::")
            if len(parts) >= 3:
                raw_id = int(parts[0])
                if raw_id in i_map:
                    idx = i_map[raw_id]
                    title = parts[1]
                    genres = parts[2].replace("|", ", ")
                    item_texts[idx] = f"{title} ({genres})"
                    item_titles[idx] = title

    print(f"Encoding {n_items} items with SentenceTransformer...")
    item_emb = st_model.encode(item_texts, batch_size=256, show_progress_bar=False, normalize_embeddings=True)
    item_emb = item_emb.astype(np.float32)

    # Compute user profiles based on positive interaction history
    print(f"Encoding {n_users} user profiles...")
    user_texts = []
    for u in range(n_users):
        u_df = df[df["u"] == u]
        if len(u_df) > 0:
            sample_items = u_df.sort_values("rating", ascending=False)["i"].head(10).values
            sample_titles = [item_titles[it] for it in sample_items if item_titles[it]]
            sample_str = ", ".join(sample_titles[:5])
            user_texts.append(f"A user who enjoys movies such as: {sample_str}.")
        else:
            user_texts.append("A general movie enthusiast.")

    user_emb = st_model.encode(user_texts, batch_size=256, show_progress_bar=False, normalize_embeddings=True)
    user_emb = user_emb.astype(np.float32)

    os.makedirs(h3_dir, exist_ok=True)
    np.savez_compressed(item_path, item_emb=item_emb)
    np.savez_compressed(user_path, user_emb=user_emb)
    print(f"Saved computed SBERT embeddings to {h3_dir}.")
    return user_emb, item_emb


def mask_trainpos(scores: np.ndarray, train_matrix: np.ndarray) -> np.ndarray:
    """Masks training positive items with -inf."""
    s = scores.copy()
    s[train_matrix > 0] = -np.inf
    return s


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    # Set random seeds for exact reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print(f"FedSUPER-LLM ADVANCED DYNAMIC BLUEPRINT MERGE (H9)")
    print(f"Device: {device} | K: {args.k} | Alpha: {args.alpha:.2f} | Rounds: {args.rounds} | Seed: {args.seed}")
    print("=" * 80)

    # Set output directory
    if args.output_dir is not None:
        output_dir = os.path.abspath(args.output_dir)
    else:
        output_dir = os.path.join(ROOT, "experiments", "H9-advanced-merge", "results")
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Load MovieLens-1M dataset
    print("\n[Step 1/6] Loading MovieLens-1M Dataset...")
    df, n_users, n_items, pop_prob, u_map, i_map = load_ml1m(min_rating=4)
    train_df, test_dict = leave_one_out_split(df, n_users, n_items)
    train_matrix = build_train_matrix(train_df, n_users, n_items)
    pop_count = train_df.groupby("i").size().reindex(range(n_items), fill_value=0).values.astype(np.int32)
    pop_prob = pop_prob.astype(np.float32)

    # Step 2: Pareto Catalog Partitioning
    print(f"\n[Step 2/6] Performing Pareto Catalog Partitioning (alpha = {args.alpha:.2f})...")
    head_idx, tail_idx = pareto_partition(pop_count, alpha=args.alpha)
    head_set = set(head_idx.tolist())
    tail_set = set(range(n_items)) - head_set

    head_volume_share = pop_count[head_idx].sum() / max(pop_count.sum(), 1)
    print(f"  Head Pool (|H|): {len(head_idx)} items ({len(head_idx)/n_items*100:.2f}% of catalog)")
    print(f"  Tail Pool (|T|): {len(tail_idx)} items ({len(tail_idx)/n_items*100:.2f}% of catalog)")
    print(f"  Head Interaction Volume Share: {head_volume_share*100:.2f}%")

    train_df_head = train_df[train_df["i"].isin(head_idx)].copy()
    train_df_tail = train_df[train_df["i"].isin(tail_idx)].copy()
    sm_head = build_train_matrix(train_df_head, n_users, n_items)
    sm_tail = build_train_matrix(train_df_tail, n_users, n_items)
    print(f"  Head Training Interactions: {len(train_df_head):,}")
    print(f"  Tail Training Interactions: {len(train_df_tail):,}")

    # Step 3: Train Federated Dual Models (M_pop and M_tail)
    print("\n[Step 3/6] Training Federated Dual Backbone Models (FedAvg / BPR)...")
    t0 = time.time()
    print(f"  Training M_pop on D_pop ({args.rounds} rounds, {args.clients_per_round} clients/round, lr={args.lr})...")
    m_pop = train_federated(
        sm_head,
        n_users,
        n_items,
        dim=args.dim,
        rounds=args.rounds,
        clients_per_round=args.clients_per_round,
        local_epochs=2,
        lr=args.lr,
        dp_eps=None,
        max_grad_norm=1.0,
        device=device,
        seed=args.seed,
        verbose=False,
    )
    scores_pop = m_pop.score_matrix().astype(np.float32)
    print(f"  M_pop trained in {time.time() - t0:.1f}s")

    t1 = time.time()
    print(f"  Training M_tail on D_tail ({args.rounds} rounds, {args.clients_per_round} clients/round, lr={args.lr})...")
    m_tail = train_federated(
        sm_tail,
        n_users,
        n_items,
        dim=args.dim,
        rounds=args.rounds,
        clients_per_round=args.clients_per_round,
        local_epochs=2,
        lr=args.lr,
        dp_eps=None,
        max_grad_norm=1.0,
        device=device,
        seed=args.seed + 1,
        verbose=False,
    )
    scores_tail = m_tail.score_matrix().astype(np.float32)
    print(f"  M_tail trained in {time.time() - t1:.1f}s")

    # Step 4: SBERT Semantic Profile Embeddings & Intra-Pool Blending
    print("\n[Step 4/6] Processing SBERT Embeddings and Calibrated Score Fusion...")
    h3_dir = os.path.join(ROOT, "experiments", "H3-llm-profiling", "results")
    user_emb, item_emb = load_or_compute_sbert_embeddings(h3_dir, df, n_users, n_items, u_map, i_map)

    # Compute cosine similarity matrix S_LLM = U_emb @ I_emb^T
    print("  Computing SBERT Semantic Similarity Matrix (U_emb @ I_emb^T)...")
    llm_scores_full = (user_emb @ item_emb.T).astype(np.float32)
    print(f"  SBERT Score Range: [{llm_scores_full.min():.4f}, {llm_scores_full.max():.4f}]")

    # Intra-pool calibrated score fusion with lambda = args.llm_lambda (0.70)
    print(f"  Applying Intra-Pool Calibrated Score Fusion (lambda = {args.llm_lambda:.2f})...")
    fused_pop = calibrated_score_fusion(
        scores_pop, llm_scores_full, head_idx, lam=args.llm_lambda, standardize=True
    )
    fused_tail = calibrated_score_fusion(
        scores_tail, llm_scores_full, tail_idx, lam=args.llm_lambda, standardize=True
    )

    # Also prepare masked CF-only scores
    masked_pop = np.full(scores_pop.shape, -np.inf, dtype=np.float32)
    masked_pop[:, list(head_set)] = scores_pop[:, list(head_set)]
    masked_tail = np.full(scores_tail.shape, -np.inf, dtype=np.float32)
    masked_tail[:, list(tail_set)] = scores_tail[:, list(tail_set)]

    # Step 5: Multi-Variant Dynamic Merge Strategy Sweep
    print("\n[Step 5/6] Executing Multi-Variant Recommendation Slates & Evaluations...")
    eval_ks = [args.k]
    if 10 not in eval_ks:
        eval_ks.append(10)

    # Strategy dictionary definition: (name, pop_scores, tail_scores, mode, kwargs)
    strategies = [
        ("FedSUPER-Static", masked_pop, masked_tail, "static", {}),
        ("FedSUPER-Dynamic-Stochastic", masked_pop, masked_tail, "stochastic", {}),
        ("FedSUPER-Dynamic-AdaptiveAlpha", masked_pop, masked_tail, "adaptive_alpha", {"alpha_base": args.alpha, "beta": 0.15, "pop_count": pop_count}),
        ("FedSUPER-Dynamic-ConfidenceElastic", masked_pop, masked_tail, "confidence_elastic", {"threshold": 0.10}),
        ("LLM-FedSUPER-Static", fused_pop, fused_tail, "static", {}),
        ("LLM-FedSUPER-Dynamic-Stochastic", fused_pop, fused_tail, "stochastic", {}),
        ("LLM-FedSUPER-Hybrid", fused_pop, fused_tail, "hybrid", {"alpha_base": args.alpha, "beta": 0.15, "pop_count": pop_count, "threshold": 0.10}),
    ]

    all_results: Dict[str, Any] = {}
    detailed_metrics: Dict[str, Dict[str, float]] = {}

    for strat_name, s_pop, s_tail, mode, extra_kwargs in strategies:
        print(f"  Evaluating variant: {strat_name:35s} (mode={mode})...")
        strat_dict: Dict[str, Any] = {}

        for k_val in eval_ks:
            reclist = dynamic_probabilistic_quota_merge(
                s_pop,
                s_tail,
                train_matrix,
                head_idx,
                N=k_val,
                mode=mode,
                seed=args.seed,
                **extra_kwargs,
            )

            # Evaluate with full_rank_eval
            metrics = full_rank_eval(
                s_pop,
                test_dict,
                K=k_val,
                head_idx=head_idx,
                train_matrix=train_matrix,
                pop_prob=pop_prob,
                reclist_matrix=reclist,
                method="reclist",
            )

            strat_dict[f"k={k_val}"] = {k: v for k, v in metrics.items() if k != "decile_props"}
            if k_val == args.k:
                detailed_metrics[strat_name] = strat_dict[f"k={k_val}"]

        all_results[strat_name] = strat_dict

    # Step 6: Save Results JSON & Print ASCII Comparison Table
    print("\n[Step 6/6] Formatting Results and Writing Artifacts...")
    json_path = os.path.join(output_dir, "metrics_h9.json")

    # Serialize results to JSON
    json_data = {
        "metadata": {
            "experiment": "H9-advanced-merge",
            "k": args.k,
            "alpha": args.alpha,
            "rounds": args.rounds,
            "clients_per_round": args.clients_per_round,
            "dim": args.dim,
            "lr": args.lr,
            "llm_lambda": args.llm_lambda,
            "seed": args.seed,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "device": device,
        },
        "variants": all_results,
        "summary_k20": {name: all_results[name][f"k={args.k}"] for name in all_results},
    }

    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"Structured metrics successfully saved to {json_path}")

    # Print Formatted ASCII Comparison Table for primary K
    print("\n" + "=" * 125)
    print(f"FedSUPER-LLM ADVANCED MERGE BENCHMARK SUMMARY (K = {args.k}, Alpha = {args.alpha:.2f}):")
    print("-" * 125)
    print(
        f"{'Variant Strategy':35s} | {'Recall@K':>9s} | {'NDCG@K':>9s} | {'APLT':>8s} | {'LTC':>8s} | "
        f"{'Rmse-PC':>9s} | {'MRMC':>8s} | {'Gini':>8s} | {'Coverage':>9s} | {'GKPI':>8s}"
    )
    print("-" * 125)

    for strat_name, _ in strategies:
        m = detailed_metrics[strat_name]
        print(
            f"{strat_name:35s} | {m['recall@K']:9.4f} | {m['ndcg@K']:9.4f} | {m['aplt']:8.4f} | {m['ltc']:8.4f} | "
            f"{m['rmse_pc']:9.4f} | {m['mrmc']:8.4f} | {m['gini_fairness']:8.4f} | {m['coverage']:9.4f} | {m['gkpi']:8.4f}"
        )
    print("=" * 125)

    # Acceptance Criteria Verification
    primary_variant = "LLM-FedSUPER-Hybrid"
    primary_metrics = detailed_metrics[primary_variant]
    rec_val = primary_metrics["recall@K"]
    rmse_val = primary_metrics["rmse_pc"]

    baseline_rec_target = 0.0370
    baseline_rmse_ceiling = 0.0560

    c1_status = "PASS"
    c2_status = "PASS" if rec_val > baseline_rec_target else "FAIL"
    c3_status = "PASS" if rmse_val <= baseline_rmse_ceiling else "FAIL"

    rec_delta = rec_val - baseline_rec_target
    rec_pct = (rec_delta / baseline_rec_target) * 100.0

    print("\n" + "=" * 80)
    print(f"H9 ACCEPTANCE CRITERIA VERIFICATION (K={args.k}):")
    print("-" * 80)
    print(f"[{c1_status}] Criterion 1: run_h9.py executed successfully from start to finish.")
    print(
        f"[{c2_status}] Criterion 2: Recall@{args.k} = {rec_val:.4f} > {baseline_rec_target:.3f} "
        f"(Delta: {rec_delta:+.4f}, {rec_pct:+.1f}%)"
    )
    print(
        f"[{c3_status}] Criterion 3: Rmse-PC = {rmse_val:.4f} <= {baseline_rmse_ceiling:.3f} "
        f"(Delta: {rmse_val - baseline_rmse_ceiling:+.4f}, Popularity calibrated)"
    )
    print("=" * 80 + "\n")

    return json_data


if __name__ == "__main__":
    cli_args = parse_args()
    run_experiment(cli_args)
