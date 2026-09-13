import pytest
import torch
import numpy as np
from src.rrfn.transition_matrix import NoiseTransitionMatrix
from src.rrfn.risk_loss import risk_consistent_loss

def test_transition_matrix_and_risk_loss():
    trans = NoiseTransitionMatrix(num_classes=5)
    
    # Check default identity
    assert trans.T_hat.shape == (5, 5)
    T_final = trans.get_T_final()
    assert T_final.shape == (5, 5)

    # Check row-stochastic property: sum across rows == 1.0
    row_sums = T_final.sum(dim=-1).detach().numpy()
    np.testing.assert_allclose(row_sums, np.ones(5), atol=1e-5)

    # Test risk consistent loss computation
    clean_probs = torch.softmax(torch.randn(10, 5), dim=-1)
    observed_ratings = torch.randint(1, 6, (10,))
    weights = torch.ones(10)

    loss = risk_consistent_loss(clean_probs, trans, observed_ratings, weights, lambda_reg=0.01)
    assert loss.item() > 0.0

    # Ensure gradient flows through delta_T
    loss.backward()
    assert trans.delta_T.grad is not None
