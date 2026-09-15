import os, sys, json
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from recdata import load_ml1m, leave_one_out_split, build_train_matrix
from metrics import full_rank_eval
from super import user_popularity_inclination, pareto_partition, logit_adjusted_calibration, mask_user_trainpos
from train import train_federated

RESULT_DIR = os.path.join(ROOT, "experiments", "H8-logit-adjusted", "results")
os.makedirs(RESULT_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

df, n_users, n_items, pop_prob, _, _ = load_ml1m(min_rating=4)
train_df, test = leave_one_out_split(df, n_users, n_items)
train_matrix = build_train_matrix(train_df, n_users, n_items)
pop_count = train_df.groupby('i').size().reindex(range(n_items), fill_value=0).values.astype(np.int32)
pop_prob = pop_prob.astype(np.float32)

head_idx, tail_idx = pareto_partition(pop_count, alpha=0.20)
pop_u = user_popularity_inclination(train_matrix, head_idx)

print("\n=== Training Base FedNCF ===")
m_fed = train_federated(train_matrix, n_users, n_items, dim=64,
                        rounds=40, clients_per_round=256, local_epochs=2,
                        lr=0.3, device=DEVICE, seed=42, verbose=True)

raw_scores = m_fed.score_matrix()

def eval_topk(name, scores, train_matrix):
    sc_masked = mask_user_trainpos(scores, train_matrix)
    m = full_rank_eval(sc_masked, test, K=10, pop_prob=pop_prob,
                        head_idx=head_idx, train_matrix=train_matrix)
    print(f"{name}: " + str({k: round(v,4) for k,v in m.items() if k != 'decile_props'}))
    return m

results = {}

# Baseline: No calibration
results["FedNCF-Raw"] = eval_topk("FedNCF-Raw", raw_scores, train_matrix)

# Sweep Logit-Adjustment tau parameter
TAUS = [0.1, 0.5, 1.0, 2.0, 5.0]
for tau in TAUS:
    name = f"Logit-Adjusted (tau={tau})"
    print(f"\n=== {name} ===")
    adj_scores = logit_adjusted_calibration(raw_scores, pop_count, pop_u, tau=tau)
    results[name] = eval_topk(name, adj_scores, train_matrix)

with open(os.path.join(RESULT_DIR, "metrics_h8.json"), "w") as f:
    json.dump(results, f, indent=2)

print("\nDone H8.")
