import torch
from torch import nn
from d2l import torch as d2l

net = nn.Sequential(
    nn.Conv2d(1,96,11,4,padding=1),nn.ReLU(),
    nn.MaxPool2d(3,2), # 池化层通道不变
    nn.Conv2d(96,256,5,padding=2),nn.ReLU(),
    nn.MaxPool2d(3,2),
    nn.Conv2d(256,384,3,padding=1),nn.ReLU(),
    nn.Conv2d(384,384,3,padding=1),nn.ReLU(),
    nn.Conv2d(384,256,3,padding=1),nn.ReLU(),
    nn.MaxPool2d(3,2),
    nn.Flatten(),
    nn.Linear(6400,4096),nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(4096,4096),nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(4096,10)
)

X = torch.randn(1, 1, 224, 224)
for layer in net:
    X=layer(X)
    print(layer.__class__.__name__,'output shape:\t',X.shape)
batch_size = 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
lr, num_epochs = 0.01, 10
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())

# 结果：
# loss 0.329, train acc 0.880, test acc 0.882
# 1155.3 examples/sec on cuda:0