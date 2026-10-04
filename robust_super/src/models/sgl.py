import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
from .lightgcn import LightGCN

class SGL(LightGCN):
    """
    SGL: Self-supervised Graph Learning for Recommendation (Wu et al., SIGIR 2021).
    Generates topological contrastive views via graph structure augmentations:
    - Edge Dropout (ED)
    - Node Dropout (ND)
    - Random Walk (RW)
    """
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        num_layers: int = 3,
        num_classes: int = 5,
        drop_rate: float = 0.1,
        augment_type: str = "ED"
    ):
        super().__init__(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            num_classes=num_classes
        )
        self.drop_rate = float(drop_rate)
        self.augment_type = str(augment_type).upper()

    def _augment_adj(self, adj_sparse: torch.Tensor, drop_rate: float) -> torch.Tensor:
        """
        Applies edge dropout or node dropout to the normalized sparse adjacency tensor.
        """
        if adj_sparse is None or drop_rate <= 0.0:
            return adj_sparse

        adj_sparse = adj_sparse.coalesce()
        indices = adj_sparse.indices()  # [2, nnz]
        values = adj_sparse.values()    # [nnz]
        shape = adj_sparse.shape

        if self.augment_type == "ND":
            # Node dropout: drop random user/item nodes entirely
            num_nodes = shape[0]
            node_keep_prob = 1.0 - drop_rate
            node_mask = torch.bernoulli(torch.full((num_nodes,), node_keep_prob, device=indices.device)).bool()
            
            row_kept = node_mask[indices[0]]
            col_kept = node_mask[indices[1]]
            edge_mask = row_kept & col_kept
        else:
            # Default Edge Dropout (ED / RW)
            edge_keep_prob = 1.0 - drop_rate
            edge_mask = torch.bernoulli(torch.full(values.shape, edge_keep_prob, device=values.device)).bool()

        aug_indices = indices[:, edge_mask]
        aug_values = values[edge_mask] * (1.0 / max(1e-6, 1.0 - drop_rate))  # Rescaling

        return torch.sparse_coo_tensor(aug_indices, aug_values, shape, device=adj_sparse.device).coalesce()

    def _propagate_with_adj(self, custom_adj: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Propagates embeddings using a specific (e.g. augmented) adjacency matrix.
        """
        all_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        embs = [all_embeddings]

        if custom_adj is not None:
            adj = custom_adj.to(all_embeddings.device)
            current_emb = all_embeddings
            for _ in range(self.num_layers):
                current_emb = torch.sparse.mm(adj, current_emb)
                embs.append(current_emb)
            final_embeddings = torch.stack(embs, dim=1).mean(dim=1)
        else:
            final_embeddings = all_embeddings

        return final_embeddings

    def get_cl_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates two independently graph-augmented views for InfoNCE contrastive learning.
        """
        if self.adj_norm is None:
            # Fallback if adjacency is not set
            all_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
            return all_embeddings, all_embeddings

        aug_adj_1 = self._augment_adj(self.adj_norm, self.drop_rate)
        aug_adj_2 = self._augment_adj(self.adj_norm, self.drop_rate)

        view1 = self._propagate_with_adj(aug_adj_1)
        view2 = self._propagate_with_adj(aug_adj_2)
        return view1, view2

    def score_candidates_batch(self, user_ids: torch.Tensor, candidate_matrix: torch.Tensor) -> torch.Tensor:
        """Batched multi-user candidate scoring."""
        return super().score_candidates_batch(user_ids, candidate_matrix)
