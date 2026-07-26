import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ========== 路径设置 ==========
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')


# ========== Dataset 类 ==========
class LeafDataset(Dataset):
    def __init__(self, csv_file, img_root, transform=None, is_train=True):
        self.data = pd.read_csv(csv_file)
        self.img_root = img_root
        self.transform = transform
        self.is_train = is_train
        self.image_col = 'image'
        self.label_col = 'label'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = row[self.image_col]

        if self.is_train:
            img_path = img_path.replace('images/', 'images_train/')
        else:
            img_path = img_path.replace('images/', 'images_test/')

        full_path = os.path.join(self.img_root, img_path)
        image = Image.open(full_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.is_train:
            label = row[self.label_col]
            return image, label
        else:
            return image, img_path


# ========== 数据增强 ==========
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_transform = val_transform


# ========== 数据加载函数 ==========
def load_data(batch_size=64, num_workers=4, val_ratio=0.2, random_state=42):
    """
    加载数据并返回 DataLoader
    返回: train_loader, val_loader, test_loader, num_classes, label_encoder
    """
    train_csv = os.path.join(DATA_DIR, 'train.csv')
    test_csv = os.path.join(DATA_DIR, 'test.csv')

    # ========== 1. 处理训练集 ==========
    train_df = pd.read_csv(train_csv, header=None)
    train_df.columns = ['image', 'label']

    label_encoder = LabelEncoder()
    train_df['label'] = label_encoder.fit_transform(train_df['label'])
    num_classes = len(label_encoder.classes_)

    # ========== 2. 划分训练集和验证集 ==========
    train_df, val_df = train_test_split(
        train_df,
        test_size=val_ratio,
        random_state=random_state,
        stratify=train_df['label']
    )

    train_df.to_csv(os.path.join(DATA_DIR, 'train_split.csv'), index=False)
    val_df.to_csv(os.path.join(DATA_DIR, 'val.csv'), index=False)

    # ========== 3. 处理测试集 ==========
    test_df = pd.read_csv(test_csv)

    if len(test_df.columns) == 1:
        test_df.columns = ['image']
    else:
        test_df = test_df.iloc[:, [0]]
        test_df.columns = ['image']

    test_df.to_csv(os.path.join(DATA_DIR, 'test_processed.csv'), index=False)

    # ========== 4. 创建 Dataset ==========
    train_dataset = LeafDataset(
        csv_file=os.path.join(DATA_DIR, 'train_split.csv'),
        img_root=DATA_DIR,
        transform=train_transform,
        is_train=True
    )

    val_dataset = LeafDataset(
        csv_file=os.path.join(DATA_DIR, 'val.csv'),
        img_root=DATA_DIR,
        transform=val_transform,
        is_train=True
    )

    test_dataset = LeafDataset(
        csv_file=os.path.join(DATA_DIR, 'test_processed.csv'),
        img_root=DATA_DIR,
        transform=test_transform,
        is_train=False
    )

    # ========== 5. 创建 DataLoader ==========
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")
    print(f"测试集样本数: {len(test_dataset)}")
    print(f"类别数: {num_classes}")

    return train_loader, val_loader, test_loader, num_classes, label_encoder


if __name__ == '__main__':
    train_loader, val_loader, test_loader, num_classes, label_encoder = load_data(
        batch_size=32,
        val_ratio=0.2
    )
    print("\n✅ 数据加载测试成功！")