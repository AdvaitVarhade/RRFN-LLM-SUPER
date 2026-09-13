import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
from typing import List, Tuple, Optional

class LightGCN(nn.Module):
    """
    LightGCN backbone for collaborative filtering with 5-class rating classification.
    """
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        num_layers: int = 3,
        num_classes: int = 5
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

        self.adj_norm: Optional[torch.Tensor] = None
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        for m in self.predictor:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def set_adjacency(self, train_dict: dict, device: str = "cpu"):
        """Constructs normalized bipartite graph adjacency matrix."""
        rows, cols = [], []
        for u, interactions in train_dict.items():
            for item, _, _ in interactions:
                rows.append(u)
                cols.append(item)

        rows = np.array(rows)
        cols = np.array(cols)

        # Bipartite matrix R of size num_users x num_items
        R = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(self.num_users, self.num_items))
        
        # Complete adjacency matrix A
        adj = sp.bmat([[None, R], [R.T, None]], format="csr")
        
        # Normalized Laplacian: D^{-1/2} A D^{-1/2}
        rowsum = np.array(adj.sum(axis=1)).flatten()
        d_inv = np.power(rowsum, -0.5, where=rowsum > 0)
        d_inv[rowsum == 0] = 0.0
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(adj).dot(d_mat).tocoo()

        indices = torch.from_numpy(np.vstack((norm_adj.row, norm_adj.col)).astype(np.int64))
        values = torch.from_numpy(norm_adj.data.astype(np.float32))
        shape = torch.Size(norm_adj.shape)

        self.adj_norm = torch.sparse_coo_tensor(indices, values, shape, device=device)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Propagates embeddings across bipartite graph and predicts rating distribution."""
        all_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        embs = [all_embeddings]

        if self.adj_norm is not None:
            adj = self.adj_norm.to(all_embeddings.device)
            current_emb = all_embeddings
            for _ in range(self.num_layers):
                current_emb = torch.sparse.mm(adj, current_emb)
                embs.append(current_emb)
            final_embeddings = torch.stack(embs, dim=1).mean(dim=1)
        else:
            final_embeddings = all_embeddings

        final_user_embs = final_embeddings[:self.num_users]
        final_item_embs = final_embeddings[self.num_users:]

        u_emb = final_user_embs[user_ids]
        i_emb = final_item_embs[item_ids]

        combined = torch.cat([u_emb, i_emb], dim=-1)
        logits = self.predictor(combined)
        probs = F.softmax(logits, dim=-1)
        return probs

    def score_items(self, user_id: int, item_ids: List[int], device: str = "cpu") -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            u_tensor = torch.full((len(item_ids),), user_id, dtype=torch.long, device=device)
            i_tensor = torch.tensor(item_ids, dtype=torch.long, device=device)
            probs = self.forward(u_tensor, i_tensor)
            weights = torch.arange(1, self.num_classes + 1, dtype=torch.float32, device=device)
            return (probs * weights).sum(dim=-1)
