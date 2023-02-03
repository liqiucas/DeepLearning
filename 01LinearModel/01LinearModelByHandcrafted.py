import torch
import matplotlib.pyplot as plt
from dataSet import loadDataSet
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class MyDataset(Dataset):
    """  由于数据集是合成的，所以我们使用自定义的构造器和官方提供的构造器都测试一下
        if the dataset is small, we can load it into memory
        if the dataset is quite large, using the filename or fileindex as index
    """

    def __init__(self, dataset):
        # dataset=[X, y]是一个list，如果直接len(dataset)=2，所以我们需要使用X的len or X.shape[0]
        self.len = len(dataset[0])
        self.X = dataset[0]
        self.y = dataset[1]

    def __getitem__(self, index):
        """implement dataset[index]"""
        return self.X[index], self.y[index]

    def __len__(self):
        """len(batch) can return length"""
        return self.len


def net(X, w, b):
    return X @ w + b


def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def sgd(params, lr, batch_size):
    """小批量随机梯度下降"""
    with torch.no_grad():  # 更新参数的时候梯度不需要参与计算
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


def train(epochs, lr, batch_size):
    for epoch in range(epochs):
        for i, (x, y) in enumerate(train_loader, 0):
            """(0, seq[0]), (1, seq[1]), (2, seq[2]), ..."""
            y_hat = net(x, w, b)
            loss = squared_loss(y, y_hat)
            loss.sum().backward()  # 这里的loss是一个vector
            sgd([w, b], lr, batch_size)
        with torch.no_grad():
            # 计算更新一个epoch的总loss，注意这个还没有区分训练集和测试集
            train_loss = squared_loss(labels, net(features, w, b))
            print(f'epoch {epoch + 1}, loss {float(train_loss.mean()):f}')


if __name__ == "__main__":
    true_w = torch.tensor([2, -3.4, 5, -3, 1, -2])
    true_b = 4.2
    # 模型参数
    w = torch.normal(0, 0.01, size=true_w.shape, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    # 构造数据
    features, labels = loadDataSet.synthetic_data(true_w, true_b, 10000)
    print(true_w.shape, true_w.size())
    print('features: ', features[0], '\nlabels:', labels[0])
    # 打印图像
    plt.figure(figsize=(3, 3))
    plt.scatter(features[:, 0].detach().numpy(), labels.detach().numpy(), 1)  # detach() 不使用计算图
    plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)
    plt.show()

    dataset = MyDataset([features, labels])
    train_loader = DataLoader(dataset=dataset, batch_size=10, shuffle=True, num_workers=2)

    train(epochs=10, lr=0.03, batch_size=10)

    # 查看与真实值的差距
    print(f'w的估计误差：{true_w - w.reshape(true_w.shape)}')
