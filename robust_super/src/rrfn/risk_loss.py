import torch
import torch.nn.functional as F

def risk_consistent_loss(
    clean_probs: torch.Tensor,
    transition_module: torch.nn.Module,
    observed_ratings: torch.Tensor,
    weights: torch.Tensor,
    lambda_reg: float = 0.01,
    epsilon: float = 1e-8
) -> torch.Tensor:
    """
    Computes importance-reweighted risk-consistent loss:
    L_RC = - mean( w_{ui} * log( p^{noisy}_{ui, y_tilde} ) ) + lambda_reg * ||delta_T||_F^2

    Args:
        clean_probs: [B, num_classes] - model output P(Y* | x_ui)
        transition_module: NoiseTransitionMatrix instance
        observed_ratings: [B] - 1-indexed (1..5) or 0-indexed observed ratings
        weights: [B] - interaction reliability weights w_{ui}
        lambda_reg: Frobenius norm penalty coefficient for delta_T
        epsilon: numerical stability floor
    """
    # Map 1-indexed ratings (1..5) to 0-indexed (0..4) if necessary
    if observed_ratings.max() > transition_module.num_classes - 1:
        labels = observed_ratings - 1
    else:
        labels = observed_ratings

    labels = labels.clamp(0, transition_module.num_classes - 1).long()

    # Pass clean probabilities through transition matrix
    noisy_probs = transition_module(clean_probs)  # [B, num_classes]

    # Extract probability assigned to observed noisy label
    batch_size = clean_probs.size(0)
    p_observed = noisy_probs[torch.arange(batch_size, device=clean_probs.device), labels]

    # Weighted negative log likelihood
    nll = -torch.log(p_observed + epsilon)
    weighted_loss = torch.mean(weights * nll)

    # Regularization on transition matrix slack delta_T
    reg = lambda_reg * torch.norm(transition_module.delta_T, p="fro") ** 2

    return weighted_loss + reg
