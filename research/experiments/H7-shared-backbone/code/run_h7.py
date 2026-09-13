import os, sys, json
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from recdata import load_ml1m, leave_one_out_split, build_train_matrix
from metrics import full_rank_eval
from super import pareto_partition, super_blueprint_merge, mask_user_trainpos
from train_shared import train_shared_backbone_federated

RESULT_DIR = os.path.join(ROOT, "experiments", "H7-shared-backbone", "results")
os.makedirs(RESULT_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

df, n_users, n_items, pop_prob, _, _ = load_ml1m(min_rating=4)
train_df, test = leave_one_out_split(df, n_users, n_items)
train_matrix = build_train_matrix(train_df, n_users, n_items)
pop_count = train_df.groupby('i').size().reindex(range(n_items), fill_value=0).values.astype(np.int32)
pop_prob = pop_prob.astype(np.float32)

head_idx, tail_idx = pareto_partition(pop_count, alpha=0.20)

train_df_head = train_df[train_df['i'].isin(head_idx)].copy()
train_df_tail = train_df[train_df['i'].isin(tail_idx)].copy()
sm_head = build_train_matrix(train_df_head, n_users, n_items)
sm_tail = build_train_matrix(train_df_tail, n_users, n_items)

print("\n=== Training Shared-Backbone FedNCF ===")
shared_model = train_shared_backbone_federated(
    train_matrix, sm_head, sm_tail, n_users, n_items, dim=64,
    rounds=40, clients_per_round=128, local_epochs=2, lr=0.3,
    device=DEVICE, seed=42
)

head_scores = mask_user_trainpos(shared_model.score_matrix(head_idx=0), train_matrix)
tail_scores = mask_user_trainpos(shared_model.score_matrix(head_idx=1), train_matrix)

reclist = super_blueprint_merge(head_scores, tail_scores, train_matrix, head_idx, N=10)

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
results["Shared-Backbone-FedSUPER"] = eval_topk("Shared-Backbone FedSUPER", head_scores, train_matrix, method='reclist', reclist=reclist)

with open(os.path.join(RESULT_DIR, "metrics_h7.json"), "w") as f:
    json.dump(results, f, indent=2)

print("\nDone H7.")
