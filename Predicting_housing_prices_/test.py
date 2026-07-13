import Read_and_preprocess_data as rpd
import train

k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
train_l, valid_l = train.k_flod(k, rpd.train_features, rpd.train_labels, num_epochs, lr,
                                weight_decay, batch_size)
print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, 'f'平均验证log rmse: {float(valid_l):f}')