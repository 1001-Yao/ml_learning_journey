import matplotlib
import matplotlib.pyplot as plt
import torch
from torch import nn
from d2l import torch as d2l
import Read_and_preprocess_data as rpd
matplotlib.use('TkAgg')

loss=nn.MSELoss()
in_features=rpd.train_features.shape[1]

def get_net():
    net =nn.Sequential(
        nn.Linear(in_features,1)
    )
    return net

# 先对预测值和真实值取对数，再计算均方根误差（RMSE）。这样做可以降低大房价样本对误差的过度影响
def log_rmse(net,features,labels):
    # 把预测值限制在 1 以上，将小于一的值设置为一，因为要取log
   clipped_preds= torch.clamp(net(features),1,float('inf'))
   rmse=torch.sqrt(loss(torch.log(clipped_preds),torch.log(labels)))
   return rmse.item() # 将单个元素的张量转换为 Python 数值的方法

def train(net,train_features,train_labels,test_features,test_labels,
          num_epochs,learning_rate,weight_decay,batch_size):
    train_ls,test_ls=[],[]
    train_iter=d2l.load_array((train_features,train_labels),batch_size)

    # 使用Adam优化算法，其初始对学习率不敏感,会用自适应学习率算法来更新 net 的所有参数，使损失函数最小化
    optimizer=torch.optim.Adam(net.parameters(),
                               lr=learning_rate,
                               weight_decay=weight_decay)

    for epoch in range(num_epochs):
        for X,y in train_iter:
            optimizer.zero_grad() # 清空上一轮迭代的梯度，防止梯度累加
            l=loss(net(X),y)
            l.backward()
            optimizer.step()
        # 记录模型在训练集和测试集上的表现（Log RMSE），最后把这两个记录列表返回。
        #  append: 将epoch 计算出的 Log RMSE 添加到列表末尾，
        #  结果：train_ls 变成 [epoch1 的 RMSE, epoch2 的 RMSE, epoch3 的 RMSE, ...]
        train_ls.append(log_rmse(net,train_features,train_labels))
        # 如果有测试集标签（比如在训练时做验证），也记录下来
        if test_labels is not None:
            test_ls.append(log_rmse(net,test_features,test_labels))
    return train_ls,test_ls

def get_k_fold_data(k,i,X,y): # i是哪一折当验证集
    assert k>1
    fold_size=X.shape[0]//k # 算出每折有多少个样本
    X_train, y_train = None, None
    X_valid, y_valid = None, None  # 先初始化为 None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size) # slice 生成一个切片，如（0，50）->[0:50]表示一个范围
        X_part, y_part = X[idx, :], y[idx] # X[idx, :] 取当前折的所有行和所有列，y[idx] 取对应的标签。
        if j == i: #j 是当前折，和 i 比较
            X_valid, y_valid = X_part, y_part
        elif X_train is None: #首次初始化训练集"的逻辑：第一折训练数据直接赋值
            X_train, y_train = X_part, y_part
        else:  #每次用 1 折当验证集，剩下的 K-1 折合并起来当训练集。
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid

def k_flod(k,X_train,y_train,num_epochs,learning_rate,weight_decay,batch_size):
    train_l_sum,valid_l_sum=0,0
    for i in range(k):
        data=get_k_fold_data(k,i,X_train,y_train)
        net=get_net()
        train_ls,valid_ls=train(net,*data,num_epochs,learning_rate,
                               weight_decay,batch_size)
        train_l_sum+=train_ls[-1]
        valid_l_sum+=valid_ls[-1]
        if i==0:
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
                     xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
                     legend=['train', 'valid'], yscale='log')
            plt.show()
        print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f}, '
              f'验证log rmse{float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k

