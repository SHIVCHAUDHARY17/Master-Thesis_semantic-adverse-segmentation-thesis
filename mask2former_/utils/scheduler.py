# utils/scheduler.py
from torch.optim.lr_scheduler import _LRScheduler

class PolyLR(_LRScheduler):
    """
    Polynomial LR decay:
        lr(iter) = max( base_lr * (1 - iter / max_iters) ** power, min_lr )
    Notes
    -----
    - Safe for resume: clamps progress in [0, max_iters] so we never raise a
      negative base to a fractional power (which would yield complex numbers).
    - You can extend training: update max_iters and step() to the current iter.
    """
    def __init__(self, optimizer, max_iters, power=0.9, last_epoch=-1, min_lr=1e-6):
        self.power = float(power)
        self.max_iters = int(max(1, max_iters))  # avoid divide-by-zero
        self.min_lr = float(min_lr)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # progress ∈ [0, max_iters]
        progress = min(max(self.last_epoch, 0), self.max_iters)
        # factor ∈ [0, 1]
        factor = 1.0 - (progress / self.max_iters)
        # clamp to avoid (-eps) ** power when resuming past old max_iters
        factor = max(0.0, factor) ** self.power

        return [max(base_lr * factor, self.min_lr) for base_lr in self.base_lrs]

    # Optional convenience helpers
    def set_max_iters(self, new_max_iters: int):
        """Update max_iters when you extend training."""
        self.max_iters = int(max(1, new_max_iters))

    def set_iter(self, cur_iter: int):
        """
        Sync the scheduler to a specific iteration.
        Equivalent to calling step(cur_iter) on recent PyTorch versions.
        """
        self.last_epoch = int(cur_iter)
        # Recompute lrs right away to keep optimizer.param_groups in sync
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

