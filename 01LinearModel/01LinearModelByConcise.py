import torch
from torch import nn
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


def init_normal(m):
    if type(m) == nn.Linear:
        """如果当前层是线性层"""
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)


def train(epochs, weight_decay, lr, net):
    criterion = nn.MSELoss()
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.03)
    optimizer = torch.optim.SGD([
        {"params": net[0].weight, 'weight_decay': weight_decay},
        {"params": net[0].bias}], lr=lr)
    for epoch in range(epochs):
        for i, (x, y) in enumerate(train_loader, 0):
            """(0, seq[0]), (1, seq[1]), (2, seq[2]), ..."""
            y_hat = net(x)
            loss = criterion(y, y_hat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss = criterion(net(features), labels)
        print(f'epoch {epoch + 1}, loss {loss:f}')


if __name__ == "__main__":
    true_w = torch.tensor([2, -3.4, 5, -3, 1, -2])
    true_b = 4.2
    # 构造数据
    features, labels = loadDataSet.synthetic_data_linear_model(true_w, true_b, 10000)

    dataset = MyDataset([features, labels])
    train_loader = DataLoader(dataset=dataset, batch_size=10, shuffle=True, num_workers=2)
    # 构造网络,  X 作为输入有7个feature，输出为一个维度并初始化参数
    net = nn.Sequential(nn.Linear(len(true_w), 1))
    net.apply(init_normal)
    print(net[0].weight.data[0], net[0].bias.data[0])

    train(epochs=10, weight_decay=0, lr=0.03, net=net)

    # 查看与真实值的差距
    w = net[0].weight.data
    print('w的估计误差：', true_w - w.reshape(true_w.shape))
    b = net[0].bias.data
    print('b的估计误差：', true_b - b)
