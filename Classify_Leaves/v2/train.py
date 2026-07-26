import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from torchvision import models, transforms
import matplotlib.pyplot as plt

# ========== 路径设置 ==========
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

# ========== 数据增强 ==========
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ========== 模型定义 ==========
def get_model(num_classes, device):
    # 使用 weights 参数代替 deprecated 的 pretrained
    # 加载预训练权重（核心步骤）
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # 解冻最后两层进行微调,其他层的反向传播设置为false,即权重不会更新（冻结），即保留了ImageNet上学到的通用特征
    for name, param in model.named_parameters():
        if "layer4" in name or "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)


# ========== 训练函数 ==========
def train_model(model, train_loader, val_loader, epochs=10, lr=1e-4, device=None):
    if device is None:
        device = next(model.parameters()).device

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    train_accs, val_accs = [], []

    for epoch in range(epochs):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        train_acc = evaluate_accuracy(model, train_loader, device)
        val_acc = evaluate_accuracy(model, val_loader, device)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f'Epoch {epoch + 1}/{epochs}: Train Acc {train_acc:.4f}, Val Acc {val_acc:.4f}')

    return train_accs, val_accs


# ========== 评估函数 ==========
def evaluate_accuracy(model, data_iter, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return correct / total


# ========== K折验证 ==========
def k_fold_cross_validation(k, dataset, epochs=10, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    train_acc_sum, val_acc_sum = 0, 0

    for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(dataset)))):
        print(f'\n===== Fold {fold + 1}/{k} =====')

        train_set = Subset(dataset, train_idx)
        val_set = Subset(dataset, val_idx)

        train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=0)

        model = get_model(num_classes, device)
        train_accs, val_accs = train_model(model, train_loader, val_loader, epochs, lr, device)

        train_acc_sum += train_accs[-1]
        val_acc_sum += val_accs[-1]

        print(f'Fold {fold + 1} Final: Train Acc {train_accs[-1]:.4f}, Val Acc {val_accs[-1]:.4f}')

        if fold == 0:
            plt.plot(range(1, epochs + 1), train_accs, label='train')
            plt.plot(range(1, epochs + 1), val_accs, label='valid')
            plt.xlabel('epoch')
            plt.ylabel('accuracy')
            plt.legend()
            plt.ylim([0, 1])
            plt.show()

    return train_acc_sum / k, val_acc_sum / k


# ========== 主程序 ==========
if __name__ == '__main__':
    # 数据准备
    train_csv = os.path.join(DATA_DIR, 'train.csv')
    df = pd.read_csv(train_csv, header=None, names=['image', 'label'])

    le = LabelEncoder()
    df['label'] = le.fit_transform(df['label'])
    num_classes = len(le.classes_)

    encoded_csv = os.path.join(DATA_DIR, 'train_encoded.csv')
    df.to_csv(encoded_csv, index=False)

    # 创建数据集
    from dataset import LeafDataset

    full_dataset = LeafDataset(encoded_csv, DATA_DIR, transform=train_transform, is_train=True)

    # 参数设置
    k, epochs, lr = 5, 20, 1e-4  # 学习率设为1e-4

    print(f"Total samples: {len(full_dataset)}, Classes: {num_classes}")
    print(f"Params: K={k}, Epochs={epochs}, LR={lr}")

    # 开始训练
    avg_train_acc, avg_val_acc = k_fold_cross_validation(k, full_dataset, epochs, lr)

    print(f'\n=== Final Result ===')
    print(f'Average Train Acc: {avg_train_acc:.4f}')
    print(f'Average Val Acc:   {avg_val_acc:.4f}')
