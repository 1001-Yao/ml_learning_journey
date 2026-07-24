"""
utils.py
----------------------------------------
Utility functions for Leaf Classification
Compatible with model.py and train.py
"""

import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt

import torch


# =====================================================
# Random Seed
# =====================================================
def seed_everything(seed=42):
    """
    Fix random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =====================================================
# AverageMeter
# =====================================================
class AverageMeter:
    """
    Computes and stores average value.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, value, n=1):
        self.val = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count


# =====================================================
# Accuracy
# =====================================================
def accuracy(outputs, labels):

    _, preds = torch.max(outputs, dim=1)

    correct = (preds == labels).sum().item()

    return correct / labels.size(0)


# =====================================================
# Evaluate
# =====================================================
@torch.no_grad()
def evaluate(model,
             dataloader,
             criterion,
             device):

    model.eval()

    loss_meter = AverageMeter()

    acc_meter = AverageMeter()

    for images, labels in dataloader:

        images = images.to(device)

        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        acc = accuracy(outputs, labels)

        loss_meter.update(loss.item(), labels.size(0))

        acc_meter.update(acc, labels.size(0))

    return loss_meter.avg, acc_meter.avg


# =====================================================
# EarlyStopping
# =====================================================
class EarlyStopping:

    def __init__(
            self,
            patience=5,
            verbose=True,
            delta=0):

        self.patience = patience

        self.verbose = verbose

        self.delta = delta

        self.counter = 0

        self.best_score = None

        self.early_stop = False

    def __call__(self, score):

        if self.best_score is None:

            self.best_score = score

            return False

        if score <= self.best_score + self.delta:

            self.counter += 1

            if self.verbose:

                print(
                    f"EarlyStopping Counter: "
                    f"{self.counter}/{self.patience}"
                )

            if self.counter >= self.patience:

                self.early_stop = True

        else:

            self.best_score = score

            self.counter = 0

        return self.early_stop


# =====================================================
# Save Model
# =====================================================
def save_checkpoint(model,
                    optimizer,
                    epoch,
                    best_acc,
                    save_path):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_acc": best_acc
        },
        save_path
    )


# =====================================================
# Load Model
# =====================================================
def load_checkpoint(model,
                    optimizer,
                    checkpoint_path,
                    device):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    epoch = checkpoint["epoch"]

    best_acc = checkpoint["best_acc"]

    return epoch, best_acc


# =====================================================
# Plot Curves
# =====================================================
def plot_curves(train_loss,
                val_loss,
                train_acc,
                val_acc,
                save_dir):

    os.makedirs(save_dir, exist_ok=True)

    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)

    plt.plot(
        epochs,
        train_loss,
        label="Train Loss",
        linewidth=2
    )

    plt.plot(
        epochs,
        val_loss,
        label="Validation Loss",
        linewidth=2
    )

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.title("Loss Curve")

    plt.grid(True)

    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)

    plt.plot(
        epochs,
        train_acc,
        label="Train Accuracy",
        linewidth=2
    )

    plt.plot(
        epochs,
        val_acc,
        label="Validation Accuracy",
        linewidth=2
    )

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.title("Accuracy Curve")

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            save_dir,
            "training_curves.png"
        ),
        dpi=300
    )

    plt.close()


# =====================================================
# Learning Rate
# =====================================================
def get_lr(optimizer):

    return optimizer.param_groups[0]["lr"]


# =====================================================
# Timer
# =====================================================
class Timer:

    def __init__(self):

        self.start = time.time()

    def elapsed(self):

        return time.time() - self.start


# =====================================================
# Device
# =====================================================
def get_device():

    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


# =====================================================
# Count Parameters
# =====================================================
def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


# =====================================================
# Print Model Info
# =====================================================
def print_model_info(model):

    total = count_parameters(model)

    print("=" * 50)

    print(model)

    print("=" * 50)

    print(f"Trainable Parameters : {total:,}")

    print("=" * 50)