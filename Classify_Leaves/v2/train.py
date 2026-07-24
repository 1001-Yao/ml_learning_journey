"""
train.py
Part 1
----------------------------------------
Leaf Classification Training
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import KFold

from dataset import LeafDataset, DATA_DIR, train_transform
from model import get_model
from utils import (
    seed_everything,
    AverageMeter,
    accuracy,
    evaluate,
    EarlyStopping,
    save_checkpoint,
    plot_curves,
    get_lr,
    Timer,
    get_device,
    print_model_info
)

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Subset

# ==========================================
# Config
# ==========================================

SEED = 42

NUM_EPOCHS = 30

K_FOLDS = 5

BATCH_SIZE = 64

LEARNING_RATE = 3e-4

WEIGHT_DECAY = 1e-4

PATIENCE = 5

CHECKPOINT_DIR = "checkpoints"

DEVICE = get_device()

USE_AMP = torch.cuda.is_available()

# ==========================================
# Seed
# ==========================================

seed_everything(SEED)

# ==========================================
# Train One Epoch
# ==========================================

def train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        scheduler,
        scaler,
        device):

    model.train()

    loss_meter = AverageMeter()

    acc_meter = AverageMeter()

    for images, labels in train_loader:

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        if USE_AMP:

            with autocast():

                outputs = model(images)

                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()

        else:

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

        acc = accuracy(outputs, labels)

        loss_meter.update(
            loss.item(),
            labels.size(0)
        )

        acc_meter.update(
            acc,
            labels.size(0)
        )

    scheduler.step()

    return (
        loss_meter.avg,
        acc_meter.avg
    )

# ==========================================
# Create Optimizer
# ==========================================

def create_optimizer(model):

    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY

    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(

        optimizer,

        T_max=NUM_EPOCHS,

        eta_min=1e-6

    )

    return optimizer, scheduler


# ==========================================
# Create Loss
# ==========================================

def create_loss():

    return nn.CrossEntropyLoss(

        label_smoothing=0.1

    )


# ==========================================
# Create AMP
# ==========================================

def create_scaler():

    return GradScaler(enabled=USE_AMP)

# ==========================================
# Prepare Dataset
# ==========================================

def prepare_dataset():

    train_csv = os.path.join(DATA_DIR, "train.csv")

    train_df = pd.read_csv(train_csv, header=None)

    train_df.columns = ["image", "label"]

    label_encoder = LabelEncoder()

    train_df["label"] = label_encoder.fit_transform(
        train_df["label"]
    )

    encoded_csv = os.path.join(
        DATA_DIR,
        "train_encoded.csv"
    )

    train_df.to_csv(
        encoded_csv,
        index=False
    )

    num_classes = len(label_encoder.classes_)

    dataset = LeafDataset(

        csv_file=encoded_csv,

        img_root=DATA_DIR,

        transform=train_transform,

        is_train=True

    )

    return dataset, num_classes


# ==========================================
# Validate
# ==========================================

@torch.no_grad()

def validate(

        model,

        val_loader,

        criterion,

        device):

    model.eval()

    loss_meter = AverageMeter()

    acc_meter = AverageMeter()

    for images, labels in val_loader:

        images = images.to(device, non_blocking=True)

        labels = labels.to(device, non_blocking=True)

        outputs = model(images)

        loss = criterion(outputs, labels)

        acc = accuracy(outputs, labels)

        loss_meter.update(

            loss.item(),

            labels.size(0)

        )

        acc_meter.update(

            acc,

            labels.size(0)

        )

    return loss_meter.avg, acc_meter.avg


# ==========================================
# K Fold Training
# ==========================================

def k_fold_train(dataset, num_classes):

    kfold = KFold(

        n_splits=K_FOLDS,

        shuffle=True,

        random_state=SEED

    )

    fold_results = []

    all_train_loss = []

    all_val_loss = []

    all_train_acc = []

    all_val_acc = []

    timer = Timer()

    for fold, (train_idx, val_idx) in enumerate(

            kfold.split(dataset), 1):

        print()

        print("=" * 60)

        print(f"Fold {fold}/{K_FOLDS}")

        print("=" * 60)

        train_dataset = Subset(

            dataset,

            train_idx

        )

        val_dataset = Subset(

            dataset,

            val_idx

        )

        train_loader = DataLoader(

            train_dataset,

            batch_size=BATCH_SIZE,

            shuffle=True,

            num_workers=2,

            pin_memory=torch.cuda.is_available(),

            persistent_workers=True

        )

        val_loader = DataLoader(

            val_dataset,

            batch_size=BATCH_SIZE,

            shuffle=False,

            num_workers=2,

            pin_memory=torch.cuda.is_available(),

            persistent_workers=True

        )

        model = get_model(num_classes)

        model.to(DEVICE)

        print_model_info(model)

        criterion = create_loss()

        optimizer, scheduler = create_optimizer(model)

        scaler = create_scaler()

        stopper = EarlyStopping(

            patience=PATIENCE

        )

        train_loss_history = []

        val_loss_history = []

        train_acc_history = []

        val_acc_history = []

        best_acc = 0.0

        for epoch in range(NUM_EPOCHS):

            train_loss, train_acc = train_one_epoch(

                model,

                train_loader,

                criterion,

                optimizer,

                scheduler,

                scaler,

                DEVICE

            )

            val_loss, val_acc = validate(

                model,

                val_loader,

                criterion,

                DEVICE

            )

            train_loss_history.append(train_loss)

            val_loss_history.append(val_loss)

            train_acc_history.append(train_acc)

            val_acc_history.append(val_acc)

            lr = get_lr(optimizer)

            print(

                f"Epoch [{epoch+1:02d}/{NUM_EPOCHS}] "

                f"LR={lr:.6f} "

                f"Train Loss={train_loss:.4f} "

                f"Train Acc={train_acc:.4f} "

                f"Val Loss={val_loss:.4f} "

                f"Val Acc={val_acc:.4f}"

            )

            if val_acc > best_acc:

                best_acc = val_acc

                save_checkpoint(

                    model,

                    optimizer,

                    epoch,

                    best_acc,

                    os.path.join(

                        CHECKPOINT_DIR,

                        f"fold{fold}_best.pth"

                    )

                )

            if stopper(val_acc):

                print("Early Stopping")

                break

        fold_results.append(best_acc)

        all_train_loss.append(train_loss_history)

        all_val_loss.append(val_loss_history)

        all_train_acc.append(train_acc_history)

        all_val_acc.append(val_acc_history)

        print(

            f"Best Validation Accuracy: "

            f"{best_acc:.4f}"

        )

    print()

    print("=" * 60)

    print("K-Fold Finished")

    print("=" * 60)

    print(

        f"Average Accuracy : "

        f"{np.mean(fold_results):.4f}"

    )

    print(

        f"Std Accuracy : "

        f"{np.std(fold_results):.4f}"

    )

    print(

        f"Training Time : "

        f"{timer.elapsed()/60:.2f} min"

    )

    return (

        all_train_loss,

        all_val_loss,

        all_train_acc,

        all_val_acc

    )
# ==========================================
# Main
# ==========================================

def main():

    print("=" * 70)
    print("Leaf Classification Training")
    print("=" * 70)

    print(f"Device        : {DEVICE}")
    print(f"Epochs        : {NUM_EPOCHS}")
    print(f"K Fold        : {K_FOLDS}")
    print(f"Batch Size    : {BATCH_SIZE}")
    print(f"Learning Rate : {LEARNING_RATE}")
    print(f"Weight Decay  : {WEIGHT_DECAY}")
    print(f"AMP           : {USE_AMP}")
    print("=" * 70)

    dataset, num_classes = prepare_dataset()

    (
        all_train_loss,
        all_val_loss,
        all_train_acc,
        all_val_acc

    ) = k_fold_train(

        dataset,

        num_classes

    )

    # ======================================
    # Average Curves
    # ======================================

    min_epoch = min(

        len(x)

        for x in all_train_loss

    )

    avg_train_loss = np.mean(

        [x[:min_epoch] for x in all_train_loss],

        axis=0

    )

    avg_val_loss = np.mean(

        [x[:min_epoch] for x in all_val_loss],

        axis=0

    )

    avg_train_acc = np.mean(

        [x[:min_epoch] for x in all_train_acc],

        axis=0

    )

    avg_val_acc = np.mean(

        [x[:min_epoch] for x in all_val_acc],

        axis=0

    )

    plot_curves(

        avg_train_loss,

        avg_val_loss,

        avg_train_acc,

        avg_val_acc,

        CHECKPOINT_DIR

    )

    print()

    print("=" * 70)

    print("Training Finished Successfully!")

    print("=" * 70)

    print(

        f"Training Curves Saved To : "

        f"{CHECKPOINT_DIR}/training_curves.png"

    )

    print(

        f"Best Models Saved To : "

        f"{CHECKPOINT_DIR}"

    )

    print("=" * 70)


# ==========================================
# Program Entry
# ==========================================

if __name__ == "__main__":

    main()