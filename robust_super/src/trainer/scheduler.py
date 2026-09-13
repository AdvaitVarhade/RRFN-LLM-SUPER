import torch
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau

class WarmupReduceLROnPlateau:
    """
    Combines linear warmup for the first N epochs with ReduceLROnPlateau decay.
    """
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int = 2,
        base_lr: float = 0.001,
        factor: float = 0.5,
        patience: int = 4,
        min_lr: float = 1e-6
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.current_epoch = 0
        self.plateau_scheduler = ReduceLROnPlateau(
            optimizer, mode="max", factor=factor, patience=patience, min_lr=min_lr
        )

    def step(self, metric: float):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            lr = self.base_lr * (self.current_epoch / max(1, self.warmup_epochs))
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
        else:
            self.plateau_scheduler.step(metric)

    def get_last_lr(self) -> list:
        return [pg["lr"] for pg in self.optimizer.param_groups]
