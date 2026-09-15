import os, sys, json, time
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from recdata import load_ml1m, leave_one_out_split, build_train_matrix
from metrics import full_rank_eval
from super import (tri_tier_partition, tri_tier_user_inclinations, 
                   tri_tier_blueprint_merge, gkpi_score, mask_user_trainpos)
from train import train_federated

RESULT_DIR = os.path.join(ROOT, "experiments", "H6-tri-tier", "results")
os.makedirs(RESULT_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

df, n_users, n_items, pop_prob, _, _ = load_ml1m(min_rating=4)
train_df, test = leave_one_out_split(df, n_users, n_items)
train_matrix = build_train_matrix(train_df, n_users, n_items)
pop_count = train_df.groupby('i').size().reindex(range(n_items), fill_value=0).values.astype(np.int32)
pop_prob = pop_prob.astype(np.float32)

print("\n=== Tri-Tier Partition (Head=0.20, Torso=0.50, Tail=0.30) ===")
head_idx, torso_idx, tail_idx = tri_tier_partition(pop_count, alpha_head=0.20, alpha_torso=0.50)
print(f"Head size: {len(head_idx)}, Torso size: {len(torso_idx)}, Tail size: {len(tail_idx)}")

train_df_head = train_df[train_df['i'].isin(head_idx)].copy()
train_df_torso = train_df[train_df['i'].isin(torso_idx)].copy()
train_df_tail = train_df[train_df['i'].isin(tail_idx)].copy()

sm_head = build_train_matrix(train_df_head, n_users, n_items)
sm_torso = build_train_matrix(train_df_torso, n_users, n_items)
sm_tail = build_train_matrix(train_df_tail, n_users, n_items)

def eval_topk(name, scores, train_matrix, method='argsort', reclist=None):
    if method == 'reclist' and reclist is not None:
        m = full_rank_eval(scores, test, K=10, pop_prob=pop_prob,
                            method='reclist', reclist_matrix=reclist,
                            head_idx=head_idx, train_matrix=train_matrix)
    else:
        sc_masked = mask_user_trainpos(scores, train_matrix)
        m = full_rank_eval(sc_masked, test, K=10, pop_prob=pop_prob,
                            head_idx=head_idx, train_matrix=train_matrix)
    print(f"{name}: " + str({k: round(v,4) for k,v in m.items() if k != 'decile_props'}))
    return m

results = {}

print("\n=== Training FedNCF on Head (D_head) ===")
m_fed_head = train_federated(sm_head, n_users, n_items, dim=64,
                              rounds=30, clients_per_round=128, local_epochs=2,
                              lr=0.3, device=DEVICE, seed=42, verbose=True)

print("\n=== Training FedNCF on Torso (D_torso) ===")
m_fed_torso = train_federated(sm_torso, n_users, n_items, dim=64,
                              rounds=30, clients_per_round=128, local_epochs=2,
                              lr=0.3, device=DEVICE, seed=42, verbose=True)

print("\n=== Training FedNCF on Tail (D_tail) ===")
m_fed_tail = train_federated(sm_tail, n_users, n_items, dim=64,
                              rounds=30, clients_per_round=128, local_epochs=2,
                              lr=0.3, device=DEVICE, seed=42, verbose=True)

head_scores = mask_user_trainpos(m_fed_head.score_matrix(), train_matrix)
torso_scores = mask_user_trainpos(m_fed_torso.score_matrix(), train_matrix)
tail_scores = mask_user_trainpos(m_fed_tail.score_matrix(), train_matrix)

reclist_tri = tri_tier_blueprint_merge(head_scores, torso_scores, tail_scores, train_matrix, head_idx, torso_idx, N=10)

results["Tri-Tier-FedSUPER"] = eval_topk("Tri-Tier FedSUPER", head_scores, train_matrix, method='reclist', reclist=reclist_tri)

with open(os.path.join(RESULT_DIR, "metrics_h6.json"), "w") as f:
    json.dump(results, f, indent=2)

print("\nDone H6.")
