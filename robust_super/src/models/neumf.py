import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

class NeuMF(nn.Module):
    """
    Neural Matrix Factorization with 5-class rating probability output.
    Combines Generalized Matrix Factorization (GMF) and Multi-Layer Perceptron (MLP).
    """
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        mlp_layers: Optional[List[int]] = None,
        dropout: float = 0.2,
        num_classes: int = 5
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        if mlp_layers is None:
            mlp_layers = [128, 64, 32]

        # GMF embeddings
        self.user_embed_gmf = nn.Embedding(num_users, embedding_dim)
        self.item_embed_gmf = nn.Embedding(num_items, embedding_dim)

        # MLP embeddings
        self.user_embed_mlp = nn.Embedding(num_users, embedding_dim)
        self.item_embed_mlp = nn.Embedding(num_items, embedding_dim)

        # MLP layers
        mlp_modules = []
        input_dim = embedding_dim * 2
        for layer_size in mlp_layers:
            mlp_modules.append(nn.Linear(input_dim, layer_size))
            mlp_modules.append(nn.ReLU())
            mlp_modules.append(nn.Dropout(p=dropout))
            input_dim = layer_size
        self.mlp = nn.Sequential(*mlp_modules)

        # Final prediction head: projects GMF + MLP outputs to num_classes logits
        final_input_dim = embedding_dim + mlp_layers[-1]
        self.prediction_head = nn.Linear(final_input_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embed_gmf.weight)
        nn.init.xavier_uniform_(self.item_embed_gmf.weight)
        nn.init.xavier_uniform_(self.user_embed_mlp.weight)
        nn.init.xavier_uniform_(self.item_embed_mlp.weight)

        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        nn.init.xavier_uniform_(self.prediction_head.weight)
        nn.init.zeros_(self.prediction_head.bias)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """
        Returns softmax probabilities over 5 rating classes: [batch_size, 5].
        """
        # GMF branch
        u_gmf = self.user_embed_gmf(user_ids)
        i_gmf = self.item_embed_gmf(item_ids)
        gmf_vector = u_gmf * i_gmf

        # MLP branch
        u_mlp = self.user_embed_mlp(user_ids)
        i_mlp = self.item_embed_mlp(item_ids)
        mlp_input = torch.cat([u_mlp, i_mlp], dim=-1)
        mlp_vector = self.mlp(mlp_input)

        # Combined representation
        combined = torch.cat([gmf_vector, mlp_vector], dim=-1)
        logits = self.prediction_head(combined)
        probs = F.softmax(logits, dim=-1)
        return probs

    def score_items(self, user_id: int, item_ids: List[int], device: str = "cpu") -> torch.Tensor:
        """
        Computes expected rating scores for candidate items for user_id:
        Expected rating = sum_{k=1}^5 k * P(Y=k)
        """
        self.eval()
        with torch.no_grad():
            u_tensor = torch.full((len(item_ids),), user_id, dtype=torch.long, device=device)
            i_tensor = torch.tensor(item_ids, dtype=torch.long, device=device)
            probs = self.forward(u_tensor, i_tensor)  # [N, 5]
            weights = torch.arange(1, self.num_classes + 1, dtype=torch.float32, device=device)
            expected_ratings = (probs * weights).sum(dim=-1)
            return expected_ratings
