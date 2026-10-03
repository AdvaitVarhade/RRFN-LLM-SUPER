import torch
import pytest
from robust_super.src.rrfn.contrastive_loss import infonce_loss, joint_risk_contrastive_loss

def test_infonce_loss_positive_and_gradient():
    torch.manual_seed(42)
    v1 = torch.randn(16, 32, requires_grad=True)
    v2 = torch.randn(16, 32, requires_grad=True)

    loss = infonce_loss(v1, v2, temperature=0.2)
    assert loss.item() > 0.0
    
    loss.backward()
    assert v1.grad is not None
    assert v2.grad is not None

def test_infonce_loss_perfect_alignment():
    # If representations are identical, loss should be small
    torch.manual_seed(42)
    v = torch.randn(16, 32)
    loss = infonce_loss(v, v, temperature=0.5)
    assert loss.item() >= 0.0

def test_joint_risk_contrastive_loss_scaling():
    risk_loss = torch.tensor(1.5)
    cl_loss = torch.tensor(2.0)

    total_1 = joint_risk_contrastive_loss(risk_loss, cl_loss, lambda_cl=0.1)
    total_2 = joint_risk_contrastive_loss(risk_loss, cl_loss, lambda_cl=0.5)

    assert total_1.item() == pytest.approx(1.7)
    assert total_2.item() == pytest.approx(2.5)
    assert total_2 > total_1
