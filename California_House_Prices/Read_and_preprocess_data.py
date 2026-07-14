import numpy as np
import pandas as pd
import torch
import Download_data as dd

DownLaod =False

if DownLaod:

    dd.DATA_HUB['kaggle_house_train'] = ( #@save
        dd.DATA_URL + 'train.csv',
        '585e9cc93e70b39160e7921475f9bcd7d31219ce')
    dd.DATA_HUB['kaggle_house_test'] = ( #@save
     dd.DATA_URL + 'test.csv',
        'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')

    train_data = pd.read_csv(dd.download('kaggle_house_train'))
    test_data = pd.read_csv(dd.download('kaggle_house_test'))
else:
    # 从本地读取
    train_data = pd.read_csv('./data/train.csv')
    test_data = pd.read_csv('./data/test.csv')

print(train_data.shape)
print(test_data.shape)


label_col='Sold Price'
feature_cols=feature_cols = [col for col in train_data.columns if col not in ['Id', label_col]]
all_features = pd.concat((train_data[feature_cols], test_data[feature_cols]))

print(all_features.shape)


numeric_feature=all_features.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns
# 标准化
all_features[numeric_feature] = all_features[numeric_feature].apply(
    lambda x: (x-x.mean())/(x.std()))
all_features[numeric_feature] = all_features[numeric_feature].fillna(0)

# 处理离散值
all_features=pd.get_dummies(all_features, dummy_na=True)

for col in all_features.columns:
    try:
        all_features[col] = all_features[col].astype(np.float32)
    except:
        print(f"列 {col} 无法转换，类型: {all_features[col].dtype}")
# all_features = all_features.astype(np.float32)


n_train=train_data.shape[0]
train_features=torch.tensor(all_features[:n_train].values, dtype=torch.float32)
test_features=torch.tensor(all_features[n_train:].values, dtype=torch.float32)
train_labels=torch.tensor(
    train_data.SoldPrice.values.reshape(-1, 1), dtype=torch.float32
)
print('down')