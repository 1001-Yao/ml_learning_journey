import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from d2l import torch as d2l

# ========== 路径设置 ==========
# 获取当前脚本所在目录 (v1)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 项目根目录 (Classify_Leaves)
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
# 数据目录 (与 v1 同级)
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

# 检查数据目录是否存在
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"数据目录不存在: {DATA_DIR}")

# ========== 动态定义 Dataset 类 (确保路径正确) ==========
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


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

        # 增加图片不存在的容错处理
        if not os.path.exists(full_path):
            alt_path = os.path.join(self.img_root, os.path.basename(img_path))
            if os.path.exists(alt_path):
                full_path = alt_path
            else:
                raise FileNotFoundError(f"图片未找到: {full_path}")

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
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ========== 设备配置 ==========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ========== ResNet 模型定义 (保留手写部分) ==========
class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, strides=1, use_1x1=False):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, stride=strides, padding=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

        if use_1x1:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None

    def forward(self, x):
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y = self.relu(y + x)
        return y


def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels, strides=2, use_1x1=True))
        else:
            blk.append(Residual(input_channels, num_channels, strides=1, use_1x1=False))
        input_channels = num_channels
    return blk


class ResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
        self.b3 = nn.Sequential(*resnet_block(64, 128, 2, first_block=False))
        self.b4 = nn.Sequential(*resnet_block(128, 256, 2, first_block=False))
        self.b5 = nn.Sequential(*resnet_block(256, 512, 2, first_block=False))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = nn.AdaptiveAvgPool2d((1, 1))(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def get_net(num_classes):
    return ResNet(num_classes).to(device)


# ========== 评估函数 ==========
def evaluate_accuracy(net, data_iter):
    net.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            correct += (y_hat.argmax(dim=1) == y).sum().item()
            total += y.numel()
    return correct / total


# ========== 训练函数 ==========
def train(net, train_iter, val_iter, num_epochs, lr, weight_decay):
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    train_accs, val_accs = [], []

    for epoch in range(num_epochs):
        net.train()
        for X, y in train_iter:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()

        train_acc = evaluate_accuracy(net, train_iter)
        val_acc = evaluate_accuracy(net, val_iter)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f'Epoch {epoch + 1}: Train Acc {train_acc:.4f}, Val Acc {val_acc:.4f}')

    return train_accs, val_accs


# ========== K 折数据划分 ==========
def get_k_fold_data(k, i, dataset):
    assert k > 1
    fold_size = len(dataset) // k
    train_indices, val_indices = [], []

    for j in range(k):
        idx_start = j * fold_size
        idx_end = (j + 1) * fold_size
        indices = list(range(idx_start, idx_end))

        if j == i:
            val_indices.extend(indices)
        else:
            train_indices.extend(indices)

    return Subset(dataset, train_indices), Subset(dataset, val_indices)


# ========== K 折交叉验证主函数 ==========
def k_fold(k, dataset, num_epochs, lr, weight_decay, batch_size, num_classes):
    train_acc_sum, val_acc_sum = 0.0, 0.0

    for i in range(k):
        print(f'\n===== Fold {i + 1}/{k} =====')
        train_data, val_data = get_k_fold_data(k, i, dataset)

        train_iter = DataLoader(train_data, batch_size, shuffle=True, num_workers=0, pin_memory=True)
        val_iter = DataLoader(val_data, batch_size, shuffle=False, num_workers=0, pin_memory=True)

        net = get_net(num_classes)
        train_accs, val_accs = train(net, train_iter, val_iter, num_epochs, lr, weight_decay)

        final_train_acc = train_accs[-1]
        final_val_acc = val_accs[-1]
        train_acc_sum += final_train_acc
        val_acc_sum += final_val_acc

        print(f'Fold {i + 1} Final: Train Acc {final_train_acc:.4f}, Val Acc {final_val_acc:.4f}')

        if i == 0:
            d2l.plot(range(1, num_epochs + 1), [train_accs, val_accs],
                     xlabel='epoch', ylabel='accuracy',
                     legend=['train', 'valid'], ylim=[0, 1])
            plt.show()

    return train_acc_sum / k, val_acc_sum / k


# ========== 主程序入口 ==========
if __name__ == '__main__':
    # 1. 数据准备
    train_csv_path = os.path.join(DATA_DIR, 'train.csv')

    # 检查 train.csv 是否存在
    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(f"train.csv 不存在: {train_csv_path}")

    # 读取原始 CSV
    df = pd.read_csv(train_csv_path, header=None, names=['image', 'label'])

    # 标签编码
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['label'])
    num_classes = len(le.classes_)

    # 保存带标签编码的临时 CSV
    encoded_csv_path = os.path.join(DATA_DIR, 'train_encoded.csv')
    df.to_csv(encoded_csv_path, index=False)

    # 初始化 Dataset
    full_dataset = LeafDataset(
        csv_file=encoded_csv_path,
        img_root=DATA_DIR,
        transform=train_transform,
        is_train=True
    )

    # 2. 设置参数
    k, num_epochs, lr, weight_decay, batch_size = 5, 10, 0.001, 1e-4, 32

    print(f"Total samples: {len(full_dataset)}, Classes: {num_classes}")
    print(f"Params: K={k}, Epochs={num_epochs}, LR={lr}, Batch Size={batch_size}")

    # 3. 开始训练
    avg_train_acc, avg_val_acc = k_fold(k, full_dataset, num_epochs, lr, weight_decay, batch_size, num_classes)

    # 4. 输出最终结果
    print(f'\n=== Final Result ===')
    print(f'Average Train Acc: {avg_train_acc:.4f}')
    print(f'Average Val Acc:   {avg_val_acc:.4f}')
