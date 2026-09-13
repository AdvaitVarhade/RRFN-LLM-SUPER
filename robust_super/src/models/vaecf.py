import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

class VaeCF(nn.Module):
    """
    Variational Autoencoder Collaborative Filtering with 5-class rating probability output.
    """
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        latent_dim: int = 32,
        num_classes: int = 5,
        dropout: float = 0.2
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # User and item base embeddings
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.item_emb = nn.Embedding(num_items, embedding_dim)

        # Variational encoder
        self.enc_mu = nn.Linear(embedding_dim * 2, latent_dim)
        self.enc_logvar = nn.Linear(embedding_dim * 2, latent_dim)

        # Decoder & rating classifier
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, embedding_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)
        for m in [self.enc_mu, self.enc_logvar]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        for m in self.decoder:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        u = self.user_emb(user_ids)
        i = self.item_emb(item_ids)
        x = torch.cat([u, i], dim=-1)

        mu = self.enc_mu(x)
        logvar = self.enc_logvar(x)
        z = self.reparameterize(mu, logvar)

        logits = self.decoder(z)
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
