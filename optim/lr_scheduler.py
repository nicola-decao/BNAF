import torch


class ReduceLROnPlateau(torch.optim.lr_scheduler.ReduceLROnPlateau):
    def __init__(self, *args, early_stopping=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.early_stopping = early_stopping
        self.early_stopping_counter = 0

    def step(self, metrics, epoch=None, callback_best=None, callback_reduce=None):
        current = metrics
        if epoch is None:
            epoch = self.last_epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
            self.early_stopping_counter = 0
            if callback_best is not None:
                callback_best()
        else:
            self.num_bad_epochs += 1
            self.early_stopping_counter += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            if callback_reduce is not None:
                callback_reduce()
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

        return self.early_stopping_counter == self.early_stopping
