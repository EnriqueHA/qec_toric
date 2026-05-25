import torch
import time

class TrainingLogger:
    # From Moritz
    def __init__(self):
        self.logs = []
        self.best_accuracy = 0

    def on_epoch_begin(self, epoch):
        self.t0 = time.perf_counter()
        self.epoch = epoch
        print(f"EPOCH {epoch} starting")

    def on_epoch_end(self, logs=None):
        epoch_time = time.perf_counter() - self.t0
        if logs["accuracy"] > self.best_accuracy:
            self.best_accuracy = logs["accuracy"]
        print(
            f"EPOCH {self.epoch} finished in {epoch_time:.3f} seconds with lr = {logs['lr']:.2e}:\n"
            f"\tloss = {logs['loss']:.5f}, accuracy = {logs['accuracy']:.4f} ({self.best_accuracy:.4f})\n"
            f"\tmodel time = {logs['model_time']:.2f} seconds, "
            f"data time = {logs['data_time']:.2f} seconds"
        )
        self.logs.append(logs)

    def on_training_begin(self, args):
        print(f"Training with distance = {args.distance}")

    def on_training_end(self):
        pass
