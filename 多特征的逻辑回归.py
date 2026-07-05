import numpy as np
# 造数据，四个样本，两个特征x
X = np.array([
    [30.0, 40.0],   # 样本1：成绩差 → 未通过
    [50.0, 60.0],   # 样本2：中等 → 未通过
    [70.0, 80.0],   # 样本3：较好 → 通过
    [90.0, 95.0],   # 样本4：优秀 → 通过
])
y = np.array([0, 0, 1, 1])   # ← 标签必须是 0 或 1！

X_b=np.c_[np.ones((X.shape[0],1)),X]  # 把“全1列”作为第一列，原 X 的所有列依次排在后面，组成新的数组 X_b。

#定义模型
class LogisticRegression:
    def __init__(self,n_features):
        self.w=np.zeros((n_features))  # 3个特征w1,w2,b, 参数 w 初始化为全 0, [0. 0. 0.]

    def sigmoid(self, z):
        """Sigmoid 函数: 把任意实数压缩到 (0,1)"""
        return 1 / (1 + np.exp(-z))

    def predict(self,X):   # w*x,返回的是概率值 (0~1)
        z=X.dot(self.w)
        return self.sigmoid(z)

    def classify(self, X):
        """根据概率做硬分类：≥0.5 判为 1，否则判为 0"""
        probs = self.predict(X)
        return (probs >= 0.5).astype(int)

    def train(self,X,y,lr=0.01,epochs=1000):  # pochs是循环次数,lr是学习率（a）
        m=len(y)
        for i in range(epochs):
            y_pred=self.predict(X)
            epsilon = 1e-8
            loss = -np.mean(
                y * np.log(y_pred + epsilon) +
                (1 - y) * np.log(1 - y_pred + epsilon)
            )

            gradient=X.T.dot(y_pred-y)/m
            self.w-=lr*gradient

            if i % 100 == 0:
                print(f"Epoch {i:4d} | Loss: {loss:.6f} | w: {self.w.round(4)}")

# === 训练 ===
model=LogisticRegression(n_features=3) # 3个特征
model.train(X_b,y,0.05,500)

print("" + "=" * 50)
print("最终结果")
print("=" * 50)

print(f"[参数]")
print(f"  w[0] (偏置 b): {model.w[0]:.4f}")
print(f"  w[1] (权重1):   {model.w[1]:.4f}")
print(f"  w[2] (权重2):   {model.w[2]:.4f}")

print(f"[预测概率]")
probs = model.predict(X_b)
for i in range(len(y)):
    print(f"  样本{i+1}: {probs[i]:.4f}  (真实标签: {y[i]})")

print(f"[分类结果]")
labels = model.classify(X_b)
for i in range(len(y)):
    match = "✓" if labels[i] == y[i] else "✗"
    print(f"  样本{i+1}: 预测={labels[i]}, 真实={y[i]}  {match}")

accuracy = np.mean(labels == y)
print(f"[准确率] {accuracy * 100:.1f}%")

print(f"[损失变化]")
print(f"  初始损失: {model.loss_history[0]:.6f}")
print(f"  最终损失: {model.loss_history[-1]:.6f}")
print(f"  损失下降: {model.loss_history[0] - model.loss_history[-1]:.6f}")