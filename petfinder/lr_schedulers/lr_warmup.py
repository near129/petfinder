import torch

def create_warmup_lr(optimizer, lr_scheduler, lr_warmup_decay=0.1, lr_warmup_epochs=5):
    warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=lr_warmup_decay, total_iters=lr_warmup_epochs
    )
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_lr_scheduler, lr_scheduler], milestones=[lr_warmup_epochs]
    )
    lr_scheduler.optimizer = optimizer  # pytorch-lightningで必要なため
    return lr_scheduler