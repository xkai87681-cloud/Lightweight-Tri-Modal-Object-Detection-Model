import numpy as np


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001, monitor='val_loss', mode='min', verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

        if mode == 'min':
            self.best_score = np.inf
            self.is_better = lambda new, best: new < best - min_delta
        else:
            self.best_score = -np.inf
            self.is_better = lambda new, best: new > best + min_delta

    def __call__(self, current_score, epoch):
        if self.is_better(current_score, self.best_score):
            if self.verbose:
                improvement = current_score - self.best_score
                print(f"  [IMPROVE] {self.monitor} improved: {self.best_score:.4f} -> {current_score:.4f} (Delta={improvement:.4f})")

            self.best_score = current_score
            self.best_epoch = epoch
            self.counter = 0
            return False
        else:
            self.counter += 1

            if self.verbose:
                print(f"  [NO IMPROVE] {self.monitor} did not improve for {self.counter}/{self.patience} epochs (best: {self.best_score:.4f} at epoch {self.best_epoch})")

            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"\n[EARLY STOP] Early stopping triggered! No improvement for {self.patience} epochs.")
                    print(f"  Best {self.monitor}: {self.best_score:.4f} at epoch {self.best_epoch}")
                return True

        return False

    def state_dict(self):
        return {
            'counter': self.counter,
            'best_score': self.best_score,
            'best_epoch': self.best_epoch,
            'early_stop': self.early_stop
        }

    def load_state_dict(self, state_dict):
        self.counter = state_dict['counter']
        self.best_score = state_dict['best_score']
        self.best_epoch = state_dict['best_epoch']
        self.early_stop = state_dict['early_stop']

    def reset(self):
        self.counter = 0
        self.early_stop = False
        if self.mode == 'min':
            self.best_score = np.inf
        else:
            self.best_score = -np.inf
        print(f"  [RESET] Early stopping reset for new stage")


if __name__ == "__main__":
    print("Testing EarlyStopping:")

    early_stopping = EarlyStopping(patience=3, min_delta=0.01, mode='min')

    val_losses = [1.0, 0.9, 0.85, 0.84, 0.845, 0.843, 0.846, 0.847]

    for epoch, val_loss in enumerate(val_losses, start=1):
        print(f"\nEpoch {epoch}: val_loss = {val_loss:.3f}")
        should_stop = early_stopping(val_loss, epoch)

        if should_stop:
            print(f"Training stopped at epoch {epoch}")
            break

    print(f"\n[BEST] Best val_loss: {early_stopping.best_score:.3f} at epoch {early_stopping.best_epoch}")
