import torch
import torch.nn.functional as F

def infonce_loss(
    view1: torch.Tensor,
    view2: torch.Tensor,
    temperature: float = 0.2,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    Computes symmetric InfoNCE contrastive loss between two embedding views.
    
    Args:
        view1: Tensor of shape [N, D] (first augmented representation view)
        view2: Tensor of shape [N, D] (second augmented representation view)
        temperature: Softmax temperature parameter tau_cl (default: 0.2)
        reduction: 'mean' or 'sum'
    
    Returns:
        Scalar contrastive loss tensor.
    """
    if view1.size(0) == 0 or view2.size(0) == 0:
        return torch.tensor(0.0, device=view1.device, requires_grad=True)

    # L2 normalize representations along embedding dimension
    z1 = F.normalize(view1, p=2, dim=-1)
    z2 = F.normalize(view2, p=2, dim=-1)

    # Cosine similarity matrix [N, N]
    sim_matrix = torch.matmul(z1, z2.transpose(0, 1)) / max(1e-6, temperature)

    labels = torch.arange(z1.size(0), device=z1.device, dtype=torch.long)

    # Symmetric cross-entropy
    loss_12 = F.cross_entropy(sim_matrix, labels, reduction=reduction)
    loss_21 = F.cross_entropy(sim_matrix.transpose(0, 1), labels, reduction=reduction)

    return 0.5 * (loss_12 + loss_21)


def joint_risk_contrastive_loss(
    risk_loss: torch.Tensor,
    cl_loss: torch.Tensor,
    lambda_cl: float = 0.1
) -> torch.Tensor:
    """
    Combines RRFN risk-consistent loss with self-supervised graph contrastive loss.
    
    L_total = L_risk + lambda_cl * L_cl
    """
    return risk_loss + float(lambda_cl) * cl_loss
