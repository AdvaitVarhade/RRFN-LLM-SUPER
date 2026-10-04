import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
from typing import List, Tuple, Optional

class LightGCN(nn.Module):
    """
    LightGCN backbone for collaborative filtering with 5-class rating classification.
    Features optimized evaluation embedding caching for sub-millisecond candidate scoring.
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
        self._cached_user_embs: Optional[torch.Tensor] = None
        self._cached_item_embs: Optional[torch.Tensor] = None
        self.register_buffer("rating_weights", torch.arange(1, num_classes + 1, dtype=torch.float32))
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

        if len(rows) == 0:
            indices = torch.empty((2, 0), dtype=torch.int64)
            values = torch.empty(0, dtype=torch.float32)
            total_nodes = self.num_users + self.num_items
            shape = torch.Size((total_nodes, total_nodes))
            self.adj_norm = torch.sparse_coo_tensor(indices, values, shape, device=device).coalesce()
            self._cached_user_embs = None
            self._cached_item_embs = None
            return

        # Bipartite matrix R of size num_users x num_items (binarized)
        R = sp.coo_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(self.num_users, self.num_items)).tocsr()
        R.data = np.ones_like(R.data, dtype=np.float32)
        
        # Complete adjacency matrix A
        adj = sp.bmat([[None, R], [R.T, None]], format="csr")
        
        # Normalized Laplacian: D^{-1/2} A D^{-1/2}
        rowsum = np.array(adj.sum(axis=1)).flatten()
        d_inv = np.zeros_like(rowsum, dtype=np.float32)
        pos_mask = rowsum > 0
        d_inv[pos_mask] = np.power(rowsum[pos_mask], -0.5)
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(adj).dot(d_mat).tocoo()

        indices = torch.from_numpy(np.vstack((norm_adj.row, norm_adj.col)).astype(np.int64))
        values = torch.from_numpy(norm_adj.data.astype(np.float32))
        shape = torch.Size(norm_adj.shape)

        self.adj_norm = torch.sparse_coo_tensor(indices, values, shape, device=device).coalesce()
        self._cached_user_embs = None
        self._cached_item_embs = None

    def compute_all_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes final user and item embeddings after L-layer graph propagation."""
        if not self.training and self._cached_user_embs is not None and self._cached_item_embs is not None:
            return self._cached_user_embs, self._cached_item_embs

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
        return final_user_embs, final_item_embs

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Propagates embeddings across bipartite graph and predicts rating distribution."""
        if not self.training and self._cached_user_embs is not None and self._cached_item_embs is not None:
            u_emb = self._cached_user_embs[user_ids]
            i_emb = self._cached_item_embs[item_ids]
        else:
            final_user_embs, final_item_embs = self.compute_all_embeddings()
            if not self.training:
                self._cached_user_embs = final_user_embs
                self._cached_item_embs = final_item_embs
            u_emb = final_user_embs[user_ids]
            i_emb = final_item_embs[item_ids]

        combined = torch.cat([u_emb, i_emb], dim=-1)
        logits = self.predictor(combined)
        probs = F.softmax(logits, dim=-1)
        return probs

    def score_items(self, user_id: int, item_ids: List[int], device: str = "cpu") -> torch.Tensor:
        """Fast inference evaluation using precomputed cached embeddings."""
        self.eval()
        if not item_ids:
            return torch.tensor([], device=device)

        with torch.no_grad():
            if self._cached_user_embs is None or self._cached_item_embs is None:
                u_all, i_all = self.compute_all_embeddings()
                self._cached_user_embs = u_all
                self._cached_item_embs = i_all

            u_vec = self._cached_user_embs[user_id].unsqueeze(0) # [1, D]
            i_vecs = self._cached_item_embs[item_ids]            # [K, D]
            u_expanded = u_vec.expand(len(item_ids), -1)         # [K, D]

            combined = torch.cat([u_expanded, i_vecs], dim=-1)
            logits = self.predictor(combined)
            probs = F.softmax(logits, dim=-1)
            return (probs * self.rating_weights).sum(dim=-1)

    def score_candidates_batch(self, user_ids: torch.Tensor, candidate_matrix: torch.Tensor) -> torch.Tensor:
        """
        Batched multi-user candidate scoring.
        user_ids: 1D Tensor of shape [B]
        candidate_matrix: 2D Tensor of shape [B, C] (or 1D [C] broadcasted across all B users)
        Returns: Tensor of shape [B, C] containing expected ratings.
        """
        self.eval()
        with torch.no_grad():
            if candidate_matrix.dim() == 1:
                candidate_matrix = candidate_matrix.unsqueeze(0).expand(user_ids.size(0), -1)

            B, C = candidate_matrix.shape
            if B == 0 or C == 0:
                return torch.empty((B, C), dtype=torch.float32, device=user_ids.device)

            dev = self.user_embedding.weight.device
            user_ids = user_ids.to(dev)
            candidate_matrix = candidate_matrix.to(dev)

            if self._cached_user_embs is None or self._cached_item_embs is None:
                u_all, i_all = self.compute_all_embeddings()
                self._cached_user_embs = u_all
                self._cached_item_embs = i_all

            u_embs = self._cached_user_embs[user_ids].unsqueeze(1).expand(B, C, -1)
            i_embs = self._cached_item_embs[candidate_matrix]

            combined = torch.cat([u_embs, i_embs], dim=-1)
            logits = self.predictor(combined)
            probs = F.softmax(logits, dim=-1)
            return (probs * self.rating_weights).sum(dim=-1)

    def _apply(self, fn):
        super()._apply(fn)
        if self.adj_norm is not None:
            self.adj_norm = fn(self.adj_norm)
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self._cached_user_embs = None
            self._cached_item_embs = None
        return self
