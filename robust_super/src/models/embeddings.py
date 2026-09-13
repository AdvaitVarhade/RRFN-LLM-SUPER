import torch
import torch.nn as nn
from typing import Tuple

class UserItemEmbeddings(nn.Module):
    """
    Shared embedding layer for user and item latent representations.
    """
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim

        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.item_emb = nn.Embedding(num_items, embedding_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.user_emb(user_ids), self.item_emb(item_ids)
