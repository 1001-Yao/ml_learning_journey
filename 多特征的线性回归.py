import numpy as np
# 造数据，四个样本，两个特征x
X=np.array([
    [1.0,2.0],
    [2.0,3.0],
    [3.0,1.0],
    [4.0,2.0],
])
# 标签
y=np.array([5.0,8.0,7.0,11.0])

X_b=np.c_[np.ones((X.shape[0],1)),X]  # 把“全1列”作为第一列，原 X 的所有列依次排在后面，组成新的数组 X_b。

#定义模型
class LinearRegression:
    def __init__(self,n_features):
        self.w=np.zeros((n_features))  # 3个特征w1,w2,b, 参数 w 初始化为全 0, [0. 0. 0.]

    def predict(self,X):   # w*x
        return X.dot(self.w)

    def train(self,X,y,lr=0.01,epochs=1000):  # pochs是循环次数,lr是学习率（a）
        m=len(y)
        for i in range(epochs):
            y_pred=self.predict(X)
            loss=np.mean((y_pred-y)**2) /2 # 损失函数Jw,b
            gradient=X.T.dot(y_pred-y)/m
            self.w-=lr*gradient

            if i % 100 == 0:
                print(f"Epoch {i}, Loss: {loss:.4f}, w: {self.w.round(3)}")  # 保留三位小数

# === 训练 ===
model=LinearRegression(n_features=3) # 3个特征
model.train(X_b,y,0.05,500)

print("\n最终参数：",model.w.round(3))
print("预测:", model.predict(X_b).round(2))
