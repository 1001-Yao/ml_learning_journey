"""
dataset.py
--------------------------------------------------
Dataset for Leaf Classification
Compatible with:
    model.py
    utils.py
    train.py
"""

import os
import pandas as pd
import torch

from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ======================================================
# Path
# ======================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")


# ======================================================
# Image Transform
# ======================================================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


train_transform = transforms.Compose([

    transforms.Resize((256, 256)),

    transforms.RandomResizedCrop(
        224,
        scale=(0.8, 1.0)
    ),

    transforms.RandomHorizontalFlip(),

    transforms.RandomVerticalFlip(0.2),

    transforms.RandomRotation(20),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    ),

    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        IMAGENET_MEAN,
        IMAGENET_STD
    ),

    transforms.RandomErasing(
        p=0.3,
        scale=(0.02, 0.2)
    )
])


test_transform = transforms.Compose([

    transforms.Resize((256, 256)),

    transforms.CenterCrop(224),

    transforms.ToTensor(),

    transforms.Normalize(
        IMAGENET_MEAN,
        IMAGENET_STD
    )

])


# ======================================================
# Dataset
# ======================================================

class LeafDataset(Dataset):

    def __init__(
            self,
            csv_file,
            img_root,
            transform=None,
            is_train=True):

        self.data = pd.read_csv(csv_file)

        self.img_root = img_root

        self.transform = transform

        self.is_train = is_train

        self.image_col = "image"

        self.label_col = "label"

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        row = self.data.iloc[idx]

        img_path = row[self.image_col]

        if self.is_train:

            img_path = img_path.replace(
                "images/",
                "images_train/"
            )

        else:

            img_path = img_path.replace(
                "images/",
                "images_test/"
            )

        full_path = os.path.join(
            self.img_root,
            img_path
        )

        if not os.path.exists(full_path):

            raise FileNotFoundError(full_path)

        image = Image.open(full_path).convert("RGB")

        if self.transform:

            image = self.transform(image)

        if self.is_train:

            label = int(row[self.label_col])

            return image, label

        else:

            return image, img_path


# ======================================================
# Load Data
# ======================================================

def load_data(
        batch_size=64,
        num_workers=None):

    if num_workers is None:

        if os.name == "nt":
            num_workers = 2
        else:
            num_workers = 4

    train_csv = os.path.join(
        DATA_DIR,
        "train.csv"
    )

    test_csv = os.path.join(
        DATA_DIR,
        "test.csv"
    )

    train_df = pd.read_csv(
        train_csv,
        header=None
    )

    train_df.columns = [
        "image",
        "label"
    ]

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

    test_df = pd.read_csv(test_csv)

    processed_test = os.path.join(
        DATA_DIR,
        "test_processed.csv"
    )

    test_df.to_csv(
        processed_test,
        index=False
    )

    num_classes = len(
        label_encoder.classes_
    )

    train_dataset = LeafDataset(

        csv_file=encoded_csv,

        img_root=DATA_DIR,

        transform=train_transform,

        is_train=True

    )

    test_dataset = LeafDataset(

        csv_file=processed_test,

        img_root=DATA_DIR,

        transform=test_transform,

        is_train=False

    )

    pin_memory = torch.cuda.is_available()

    persistent_workers = num_workers > 0

    train_loader = DataLoader(

        train_dataset,

        batch_size=batch_size,

        shuffle=True,

        drop_last=True,

        num_workers=num_workers,

        pin_memory=pin_memory,

        persistent_workers=persistent_workers

    )

    test_loader = DataLoader(

        test_dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=num_workers,

        pin_memory=pin_memory,

        persistent_workers=persistent_workers

    )

    print("=" * 60)

    print("Dataset Loaded")

    print("=" * 60)

    print(f"Train Images : {len(train_dataset)}")

    print(f"Test Images  : {len(test_dataset)}")

    print(f"Classes      : {num_classes}")

    print(f"Workers      : {num_workers}")

    print(f"Pin Memory   : {pin_memory}")

    print("=" * 60)

    return (
        train_loader,
        test_loader,
        num_classes,
        label_encoder
    )


# ======================================================
# Test
# ======================================================

if __name__ == "__main__":

    train_loader, test_loader, num_classes, encoder = load_data()

    images, labels = next(iter(train_loader))

    print(images.shape)

    print(labels.shape)

    print("Dataset OK!")