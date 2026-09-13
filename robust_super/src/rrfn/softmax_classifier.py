import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftmaxRatingClassifier(nn.Module):
    """
    Wrapper module ensuring any backbone model produces 5-class rating probabilities.
    """
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward call returns softmax probabilities: [batch_size, 5]
        """
        return self.backbone(user_ids, item_ids)

    def score_items(self, user_id: int, item_ids: list, device: str = "cpu") -> torch.Tensor:
        return self.backbone.score_items(user_id, item_ids, device=device)
