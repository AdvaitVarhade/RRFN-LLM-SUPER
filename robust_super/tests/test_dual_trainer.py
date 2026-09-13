import pytest
import torch
from src.models.neumf import NeuMF
from src.trainer.dual_trainer import DualModelTrainer, build_data_loader

def test_dual_trainer():
    num_users = 20
    num_items = 30
    train_dict = {
        u: [(i, (u + i) % 5 + 1, 1000 + i) for i in range(5)] for u in range(num_users)
    }
    weights = {(u, i): 1.0 for u in train_dict for i, _, _ in train_dict[u]}
    val_dict = {u: (6, 4, 2000) for u in range(num_users)}
    allowed_items = set(range(num_items))

    def factory():
        return NeuMF(num_users, num_items, embedding_dim=16, mlp_layers=[32, 16])

    trainer = DualModelTrainer(model_factory=factory, device="cpu", epochs=2, early_stopping_patience=2)
    loader = build_data_loader(train_dict, weights, batch_size=16)

    # Warm train test
    model = factory()
    trainer.warm_train(model, loader, warm_epochs=1)

    # Dual train test
    trained_model, trans_module = trainer.train_single_model(loader, val_dict, allowed_items)
    assert trained_model is not None
    assert trans_module is not None
