import numpy as np
import pandas as pd
import torch
import Download_data as dd

DownLaod =False

if DownLaod:

    dd.DATA_HUB['kaggle_house_train'] = ( #@save
        dd.DATA_URL + 'kaggle_house_pred_train.csv',
        '585e9cc93e70b39160e7921475f9bcd7d31219ce')
    dd.DATA_HUB['kaggle_house_test'] = ( #@save
     dd.DATA_URL + 'kaggle_house_pred_test.csv',
        'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')

    train_data = pd.read_csv(dd.download('kaggle_house_train'))
    test_data = pd.read_csv(dd.download('kaggle_house_test'))
else:
    # 从本地读取
    train_data = pd.read_csv('./data/kaggle_house_pred_train.csv')
    test_data = pd.read_csv('./data/kaggle_house_pred_test.csv')
#
# print(train_data.shape)
# print(test_data.shape)
#
# # iloc 是 Pandas 的位置索引器； 取出第 0 行到第 3 行，同时取出第 0、1、2、3 列以及最后 3 列的数据。
# print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])
# print(test_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])

# # 预处理！！
# 来把多个 DataFrame 或 Series 拼在一起的函数。对训练集：去掉id和最后标签（price）；对测试集，去掉id，因为没有标签
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

# 对缺失的值设置该列的平均值，为了量化数据，进行标准化

numeric_feature=all_features.dtypes[(all_features.dtypes !='object')& (all_features.dtypes != 'string')].index
# 标准化
all_features[numeric_feature] = all_features[numeric_feature].apply(
    lambda x: (x-x.mean())/(x.std()))
# 在标准化数据之后，所有均值消失，因此我们可以将缺失值设置为0
all_features[numeric_feature] = all_features[numeric_feature].fillna(0)


# 处理离散值
# 把文本类型的类别特征（如颜色、城市）转换成数值类型的 0/1 特征列
# dummy=true-> 把缺失值（NaN）也当作一个独立的类别，单独生成一列
all_features=pd.get_dummies(all_features, dummy_na=True)

# 强制转换所有列为 float32（非数值列会报错，正好帮我们发现）
all_features = all_features.astype(np.float32)


n_train=train_data.shape[0]
train_features=torch.tensor(all_features[:n_train].values, dtype=torch.float32)
test_features=torch.tensor(all_features[n_train:].values, dtype=torch.float32)
train_labels=torch.tensor(
    train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32
)