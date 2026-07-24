import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import matplotlib
import matplotlib.pyplot as plt
import time
import numpy as np

matplotlib.use('Agg')

# 从 dataset.py 导入
from dataset import DATA_DIR, LeafDataset, train_transform, test_transform

# ========== 设备 ==========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


# ========== ResNet 模型定义 ==========
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

        y = y + x
        y = self.relu(y)
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
        x = self.fc(x)
        return x


def get_net(num_classes):
    return ResNet(num_classes).to(device)


# ========== 评估函数 ==========
def evaluate_accuracy(net, data_iter):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            outputs = net(X)
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return correct / total if total > 0 else 0.0


def evaluate_loss(net, data_iter, criterion):
    net.eval()
    total_loss = 0
    total = 0
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            outputs = net(X)
            loss = criterion(outputs, y)
            total_loss += loss.item() * y.size(0)
            total += y.size(0)
    return total_loss / total if total > 0 else 0.0


# ========== 训练函数 ==========
def train_once(net, train_iter, val_iter, num_epochs, learning_rate, weight_decay):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    for epoch in range(num_epochs):
        net.train()
        total_loss = 0
        correct = 0
        total = 0

        for X, y in train_iter:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = net(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y.size(0)
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

        scheduler.step()

        train_loss = total_loss / total
        train_acc = correct / total
        val_loss = evaluate_loss(net, val_iter, criterion)
        val_acc = evaluate_accuracy(net, val_iter)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(
            f"  Epoch {epoch + 1}/{num_epochs}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

    return train_losses, train_accs, val_losses, val_accs


# ========== K 折交叉验证 ==========
def get_k_fold_data(k, i, dataset):
    assert k > 1
    fold_size = len(dataset) // k

    train_indices = []
    val_indices = []

    for j in range(k):
        start = j * fold_size
        end = (j + 1) * fold_size if j < k - 1 else len(dataset)
        indices = list(range(start, end))

        if j == i:
            val_indices.extend(indices)
        else:
            train_indices.extend(indices)

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    return train_dataset, val_dataset


def k_fold_train(k, dataset, num_epochs, learning_rate, weight_decay, batch_size):
    all_train_losses = []
    all_val_losses = []
    all_train_accs = []
    all_val_accs = []

    for i in range(k):
        print(f"\n{'=' * 50}")
        print(f"第 {i + 1}/{k} 折")
        print(f"{'=' * 50}")

        train_dataset, val_dataset = get_k_fold_data(k, i, dataset)

        # num_workers=0 避免 Windows 多进程问题
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=True
        )

        print(f"训练集: {len(train_dataset)} 张, 验证集: {len(val_dataset)} 张")

        net = get_net(num_classes)
        train_losses, train_accs, val_losses, val_accs = train_once(
            net, train_loader, val_loader, num_epochs, learning_rate, weight_decay
        )

        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)
        all_train_accs.append(train_accs)
        all_val_accs.append(val_accs)

        best_val_acc = max(val_accs)
        best_epoch = val_accs.index(best_val_acc) + 1
        print(f"第 {i + 1} 折最佳验证准确率: {best_val_acc:.4f} ({best_val_acc * 100:.2f}%) (Epoch {best_epoch})")

    return all_train_losses, all_val_losses, all_train_accs, all_val_accs


# ========== 主程序 ==========
def main():
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder

    # 设置环境变量
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    print("=" * 70)
    print("叶子分类训练 (K 折交叉验证)")
    print("=" * 70)

    # ---------- 加载完整训练集 ----------
    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    train_csv = os.path.join(DATA_DIR, 'train.csv')

    train_df = pd.read_csv(train_csv, header=None)
    train_df.columns = ['image', 'label']

    label_encoder = LabelEncoder()
    train_df['label'] = label_encoder.fit_transform(train_df['label'])

    encoded_csv = os.path.join(DATA_DIR, 'train_encoded.csv')
    train_df.to_csv(encoded_csv, index=False)

    global num_classes
    num_classes = len(label_encoder.classes_)
    print(f"类别数: {num_classes}")

    full_dataset = LeafDataset(
        csv_file=encoded_csv,
        img_root=DATA_DIR,
        transform=train_transform,
        is_train=True
    )
    print(f"总样本数: {len(full_dataset)}")

    # ---------- 训练参数 ----------
    k = 5
    num_epochs = 10
    learning_rate = 0.001
    weight_decay = 1e-4
    batch_size = 32

    print(f"\n训练参数:")
    print(f"  K 折: {k}")
    print(f"  训练轮数: {num_epochs}")
    print(f"  学习率: {learning_rate}")
    print(f"  权重衰减: {weight_decay}")
    print(f"  批次大小: {batch_size}")
    print("=" * 70)

    # ---------- K 折交叉验证 ----------
    start_time = time.time()

    all_train_losses, all_val_losses, all_train_accs, all_val_accs = k_fold_train(
        k, full_dataset, num_epochs, learning_rate, weight_decay, batch_size
    )

    total_time = time.time() - start_time

    # ---------- 结果汇总 ----------
    print("\n" + "=" * 70)
    print("K 折交叉验证完成！")
    print("=" * 70)
    print(f"\n总训练时间: {total_time:.2f} 秒 ({total_time / 60:.2f} 分钟)")

    best_val_accs = [max(accs) for accs in all_val_accs]

    print("\n各折最佳验证准确率:")
    for i, acc in enumerate(best_val_accs):
        print(f"  第 {i + 1} 折: {acc:.4f} ({acc * 100:.2f}%)")

    avg_acc = np.mean(best_val_accs)
    std_acc = np.std(best_val_accs)

    print("\n" + "-" * 70)
    print("最终指标汇总:")
    print("-" * 70)
    print(f"平均最佳验证准确率: {avg_acc:.4f} ({avg_acc * 100:.2f}%)")
    print(f"验证准确率标准差: {std_acc:.4f} ({std_acc * 100:.2f}%)")
    print("-" * 70)

    # ---------- 绘制平均曲线 ----------
    try:
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        avg_train_loss = np.mean(all_train_losses, axis=0)
        avg_val_loss = np.mean(all_val_losses, axis=0)
        plt.plot(range(1, num_epochs + 1), avg_train_loss, 'b-', linewidth=2, label='Train Loss')
        plt.plot(range(1, num_epochs + 1), avg_val_loss, 'r-', linewidth=2, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Average Loss Curve')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        avg_train_acc = np.mean(all_train_accs, axis=0)
        avg_val_acc = np.mean(all_val_accs, axis=0)
        plt.plot(range(1, num_epochs + 1), avg_train_acc, 'b-', linewidth=2, label='Train Acc')
        plt.plot(range(1, num_epochs + 1), avg_val_acc, 'r-', linewidth=2, label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Average Accuracy Curve')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(DATA_DIR, 'training_curves.png'), dpi=150)
        plt.show()
        print(f"\n训练曲线已保存至: {os.path.join(DATA_DIR, 'training_curves.png')}")
    except Exception as e:
        print(f"(绘图失败: {e})")


# ========== 程序入口保护 ==========
if __name__ == '__main__':
    main()