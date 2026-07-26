import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torchvision import models, transforms
import matplotlib.pyplot as plt

# ========== 导入 dataset 模块 ==========
from dataset_no_K import load_data, train_transform, val_transform

# ========== 路径设置 ==========
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # 上一级目录
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')  # data 在项目根目录下

# print(f"数据目录: {DATA_DIR}")


# ========== 模型定义 ==========
def get_model(num_classes, device, dropout_rate=0.5):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    for name, param in model.named_parameters():
        if "layer4" in name or "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(in_features, num_classes)
    )

    return model.to(device)


# ========== 评估函数 ==========
def evaluate_accuracy(model, data_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return correct / total


# ========== 训练函数 ==========
def train_model(model, train_loader, val_loader, epochs=20, lr=1e-4,
                device=None, label_smoothing=0.1, save_path='best_model.pth'):
    if device is None:
        device = next(model.parameters()).device

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )

    train_accs, val_accs = [], []
    best_val_acc = 0
    best_model_state = None

    print(f"\n{'=' * 60}")
    print(f"开始训练 (Epochs={epochs}, LR={lr}, Label Smoothing={label_smoothing})")
    print(f"{'=' * 60}\n")

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_acc = evaluate_accuracy(model, train_loader, device)
        val_acc = evaluate_accuracy(model, val_loader, device)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        avg_loss = train_loss / len(train_loader)
        print(f'Epoch {epoch + 1:3d}/{epochs}: Loss {avg_loss:.4f}, '
              f'Train Acc {train_acc:.4f}, Val Acc {val_acc:.4f}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, save_path)
            print(f'  ✅ 保存最佳模型 (Val Acc: {best_val_acc:.4f})')

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f'\n✅ 训练完成！最佳验证准确率: {best_val_acc:.4f}')
        print(f'   模型已保存到: {save_path}')

    return train_accs, val_accs, best_val_acc


# ========== 绘制训练曲线 ==========
def plot_training_history(train_accs, val_accs, best_val_acc, save_path=None):
    epochs = len(train_accs)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_accs, label='Train Accuracy',
             marker='o', linewidth=2)
    plt.plot(range(1, epochs + 1), val_accs, label='Validation Accuracy',
             marker='s', linewidth=2)

    best_epoch = np.argmax(val_accs) + 1
    plt.axhline(y=best_val_acc, color='r', linestyle='--', alpha=0.5,
                label=f'Best Val: {best_val_acc:.4f}')
    plt.scatter(best_epoch, best_val_acc, color='red', s=100, zorder=5)

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Training History (Best Val Acc: {best_val_acc:.4f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"训练曲线已保存到: {save_path}")

    plt.show()


# ========== 测试集预测 ==========
def predict_test(model, test_loader, label_encoder, device, save_path=None):
    model.eval()
    predictions = []
    image_paths = []

    print("\n正在对测试集进行预测...")

    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 2:
                images, paths = batch
            else:
                images = batch[0]
                paths = batch[1] if len(batch) > 1 else [f"unknown_{i}" for i in range(len(images))]

            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            predictions.extend(predicted.cpu().numpy())
            image_paths.extend(paths)

    predicted_labels = label_encoder.inverse_transform(predictions)

    result_df = pd.DataFrame({
        'image': image_paths,
        'label': predicted_labels
    })

    if save_path is None:
        save_path = os.path.join(DATA_DIR, 'test_predictions.csv')

    result_df.to_csv(save_path, index=False)
    print(f"\n✅ 预测结果已保存到: {save_path}")
    print(f"   共 {len(result_df)} 个样本")

    print("\n前10个预测结果:")
    print(result_df.head(10))

    return result_df


# ========== 主程序 ==========
def main():
    # ===== 1. 设备配置 =====
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{'=' * 60}")
    print(f"设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"{'=' * 60}")

    # ===== 2. 超参数 =====
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    VAL_RATIO = 0.2
    EPOCHS = 20
    LR = 1e-4
    DROPOUT_RATE = 0.5
    LABEL_SMOOTHING = 0.1
    RANDOM_STATE = 42

    # ===== 3. 加载数据 =====
    print("\n加载数据...")
    train_loader, val_loader, test_loader, num_classes, label_encoder = load_data(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        val_ratio=VAL_RATIO,
        random_state=RANDOM_STATE
    )

    print(f"\n数据集信息:")
    print(f"  - 类别数: {num_classes}")
    print(f"  - 训练集批次: {len(train_loader)}")
    print(f"  - 验证集批次: {len(val_loader)}")
    print(f"  - 测试集批次: {len(test_loader)}")

    # ===== 4. 创建模型 =====
    model = get_model(num_classes, device, dropout_rate=DROPOUT_RATE)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数:")
    print(f"  - 可训练参数: {trainable_params:,}")
    print(f"  - 总参数: {total_params:,}")
    print(f"  - 微调比例: {trainable_params / total_params * 100:.2f}%")

    # ===== 5. 训练 =====
    save_path = os.path.join(DATA_DIR, 'best_model.pth')
    train_accs, val_accs, best_val_acc = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        lr=LR,
        device=device,
        label_smoothing=LABEL_SMOOTHING,
        save_path=save_path
    )

    # ===== 6. 绘制训练曲线 =====
    plot_path = os.path.join(DATA_DIR, 'training_history.png')
    plot_training_history(train_accs, val_accs, best_val_acc, save_path=plot_path)

    # ===== 7. 测试集预测 =====
    model.load_state_dict(torch.load(save_path))
    model.to(device)

    predict_test(model, test_loader, label_encoder, device)

    print(f"\n{'=' * 60}")
    print("🎉 全部流程完成！")
    print(f"最佳验证准确率: {best_val_acc:.4f}")
    print(f"模型保存位置: {save_path}")
    print(f"预测结果保存位置: {os.path.join(DATA_DIR, 'test_predictions.csv')}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()