import os, sys, time
import numpy as np
import torch
import torch.nn as nn
from model import SharedBackboneRecMF, bpr_loss, pick_negatives

def train_shared_backbone_federated(train_matrix, sm_head, sm_tail, n_users, n_items, dim=64, 
                                    rounds=30, clients_per_round=128, local_epochs=2, lr=0.3, 
                                    device="cuda" if torch.cuda.is_available() else "cpu", seed=42):
    """
    Federated training loop for SharedBackboneRecMF.
    Updates the shared user/item backbone on ALL items a client has interacted with.
    Updates the pool-specific projection head (head_idx=0 for head, 1 for tail) for the specific item.
    """
    torch.manual_seed(seed)
    
    server_model = SharedBackboneRecMF(n_users, n_items, dim=dim, seed=seed, num_heads=2).to(device)
    
    local_user_emb = nn.Embedding(n_users, dim).to(device)
    local_user_emb.weight.data.uniform_(-0.01, 0.01)

    rng = np.random.default_rng(seed + 1)
    active_users = np.where(train_matrix.sum(axis=1) > 0)[0]
    
    for rd in range(rounds):
        chosen = rng.choice(active_users, size=min(clients_per_round, len(active_users)), replace=False)
        
        # Server snapshot
        start_item_bb = server_model.item_backbone.weight.detach().clone()
        start_proj_head = server_model.head_projections[0].weight.detach().clone()
        start_proj_tail = server_model.head_projections[1].weight.detach().clone()
        start_bias_head = server_model.item_biases[0].weight.detach().clone()
        start_bias_tail = server_model.item_biases[1].weight.detach().clone()
        
        # Accumulators
        bb_sum = torch.zeros_like(start_item_bb)
        bb_cnt = torch.zeros(start_item_bb.shape[0], device=device)
        
        proj_head_sum = torch.zeros_like(start_proj_head)
        proj_tail_sum = torch.zeros_like(start_proj_tail)
        clients_head = 0
        clients_tail = 0
        
        bias_head_sum = torch.zeros_like(start_bias_head)
        bias_head_cnt = torch.zeros(start_bias_head.shape[0], device=device)
        bias_tail_sum = torch.zeros_like(start_bias_tail)
        bias_tail_cnt = torch.zeros(start_bias_tail.shape[0], device=device)
        
        losses = []
        
        for u_cl in chosen:
            user_row = train_matrix[u_cl]
            pos_items = np.where(user_row > 0)[0]
            if len(pos_items) < 2:
                continue
                
            local_item_bb = start_item_bb.clone().detach().requires_grad_(True)
            local_proj_head = start_proj_head.clone().detach().requires_grad_(True)
            local_proj_tail = start_proj_tail.clone().detach().requires_grad_(True)
            local_bias_head = start_bias_head.clone().detach().requires_grad_(True)
            local_bias_tail = start_bias_tail.clone().detach().requires_grad_(True)
            local_user_vec = local_user_emb.weight[u_cl].detach().clone().requires_grad_(True)
            
            opt = torch.optim.SGD([local_item_bb, local_proj_head, local_proj_tail, 
                                   local_bias_head, local_bias_tail, local_user_vec], lr=lr)
                                   
            sweep = rng.choice(pos_items, size=min(local_epochs * 4, len(pos_items)), replace=False)
            touched_bb = set()
            touched_bias_head = set()
            touched_bias_tail = set()
            
            used_head = False
            used_tail = False
            
            for pidx in sweep:
                pidx = int(pidx)
                nidx = int(pick_negatives(user_row, n_items, rng=rng)[0])
                
                # Determine which head to use based on which submatrix the item belongs to
                is_head_item = sm_head[u_cl, pidx] > 0
                head_idx = 0 if is_head_item else 1
                
                eu = local_user_vec.unsqueeze(0)
                ei_pos = local_item_bb[pidx].unsqueeze(0)
                ei_neg = local_item_bb[nidx].unsqueeze(0)
                
                if head_idx == 0:
                    ei_pos_proj = ei_pos @ local_proj_head.T
                    ei_neg_proj = ei_neg @ local_proj_head.T
                    b_pos = local_bias_head[pidx].squeeze()
                    b_neg = local_bias_head[nidx].squeeze()
                    used_head = True
                    touched_bias_head.update([pidx, nidx])
                else:
                    ei_pos_proj = ei_pos @ local_proj_tail.T
                    ei_neg_proj = ei_neg @ local_proj_tail.T
                    b_pos = local_bias_tail[pidx].squeeze()
                    b_neg = local_bias_tail[nidx].squeeze()
                    used_tail = True
                    touched_bias_tail.update([pidx, nidx])
                    
                ps = (eu * ei_pos_proj).sum() + b_pos
                ns = (eu * ei_neg_proj).sum() + b_neg
                
                loss = bpr_loss(ps.unsqueeze(0), ns.unsqueeze(0))
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(loss.item())
                touched_bb.update([pidx, nidx])
                
            with torch.no_grad():
                local_user_emb.weight[u_cl].data = local_user_vec.detach()
                
            if touched_bb:
                t_idx = torch.tensor(list(touched_bb), device=device, dtype=torch.long)
                bb_sum[t_idx] += (local_item_bb.detach() - start_item_bb)[t_idx]
                bb_cnt[t_idx] += 1.0
                
            if used_head:
                proj_head_sum += (local_proj_head.detach() - start_proj_head)
                clients_head += 1
                t_idx = torch.tensor(list(touched_bias_head), device=device, dtype=torch.long)
                bias_head_sum[t_idx] += (local_bias_head.detach() - start_bias_head)[t_idx]
                bias_head_cnt[t_idx] += 1.0
                
            if used_tail:
                proj_tail_sum += (local_proj_tail.detach() - start_proj_tail)
                clients_tail += 1
                t_idx = torch.tensor(list(touched_bias_tail), device=device, dtype=torch.long)
                bias_tail_sum[t_idx] += (local_bias_tail.detach() - start_bias_tail)[t_idx]
                bias_tail_cnt[t_idx] += 1.0

        with torch.no_grad():
            mask = bb_cnt > 0
            if mask.any():
                server_model.item_backbone.weight.data[mask] += (bb_sum[mask] / bb_cnt[mask].unsqueeze(1))
            
            if clients_head > 0:
                server_model.head_projections[0].weight.data += (proj_head_sum / clients_head)
                mask = bias_head_cnt > 0
                if mask.any():
                    server_model.item_biases[0].weight.data[mask] += (bias_head_sum[mask] / bias_head_cnt[mask].unsqueeze(1))
                    
            if clients_tail > 0:
                server_model.head_projections[1].weight.data += (proj_tail_sum / clients_tail)
                mask = bias_tail_cnt > 0
                if mask.any():
                    server_model.item_biases[1].weight.data[mask] += (bias_tail_sum[mask] / bias_tail_cnt[mask].unsqueeze(1))

        if rd % 10 == 0 or rd == rounds - 1:
            print(f"round {rd:3d}: loss ~{np.mean(losses[-200:]) if losses else 0:.4f}")
            
    server_model.user_backbone.weight.data = local_user_emb.weight.data.to(device)
    server_model.eval()
    return server_model
