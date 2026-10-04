import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
from .lightgcn import LightGCN

class SimGCL(LightGCN):
    """
    SimGCL: Simple Graph Contrastive Learning for Recommendation (He et al., SIGIR 2022).
    Generates perturbed contrastive views directly in embedding space by adding uniform noise,
    without computationally expensive graph augmentations.
    """
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        num_layers: int = 3,
        num_classes: int = 5,
        noise_eps: float = 0.1
    ):
        super().__init__(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            num_classes=num_classes
        )
        self.noise_eps = float(noise_eps)

    def _propagate_with_noise(self, noise_scale: float = 0.1) -> torch.Tensor:
        """
        GCN propagation with uniform noise perturbation added at each layer.
        """
        all_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        embs = [all_embeddings]

        if self.adj_norm is not None:
            adj = self.adj_norm.to(all_embeddings.device)
            current_emb = all_embeddings
            for _ in range(self.num_layers):
                current_emb = torch.sparse.mm(adj, current_emb)
                if noise_scale > 0.0:
                    # Uniform random noise in [-eps, +eps]
                    random_noise = torch.empty_like(current_emb).uniform_(-noise_scale, noise_scale)
                    # Normalize noise to preserve embedding scale
                    norm_noise = F.normalize(random_noise, p=2, dim=-1) * noise_scale
                    current_emb = current_emb + norm_noise
                embs.append(current_emb)
            final_embeddings = torch.stack(embs, dim=1).mean(dim=1)
        else:
            final_embeddings = all_embeddings

        return final_embeddings

    def get_cl_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates two independently perturbed embedding views for InfoNCE contrastive learning.
        Returns (view1, view2), each of shape [(num_users + num_items), embedding_dim].
        """
        view1 = self._propagate_with_noise(noise_scale=self.noise_eps)
        view2 = self._propagate_with_noise(noise_scale=self.noise_eps)
        return view1, view2

    def score_candidates_batch(self, user_ids: torch.Tensor, candidate_matrix: torch.Tensor) -> torch.Tensor:
        """Batched multi-user candidate scoring."""
        return super().score_candidates_batch(user_ids, candidate_matrix)
