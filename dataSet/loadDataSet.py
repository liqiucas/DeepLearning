import time
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader


def load_data_mnist(batch_size=32, num_workers=4, resize=None):
    """MNIST数据集加载"""
    transform = [transforms.ToTensor()]
    if resize:
        transform.insert(0, transforms.Resize(resize))
    transform = transforms.Compose(transform)

    train_set = torchvision.datasets.MNIST(root="./dataset/MNIST", train=True, transform=transform, download=True)
    test_set = torchvision.datasets.MNIST(root="./dataset/MNIST", train=False, transform=transform, download=True)

    # 第0个分类的第0张图片的大小：1, 28, 28 channel=1, hw=28,28
    print(len(train_set), len(test_set), train_set[0][0].shape)

    # X.shape [18, 1, 28, 28]  y.shape 18
    """ 画出第一个batch的图像
    X, y = next(iter(DataLoader(train_set, batch_size=18)))
    show_images(X.reshape(18, 28, 28), 2, 9, titles=None)
    """
    return (DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers),
            DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers))


def get_fashion_mnist_labels(labels):
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  # @save
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):  # 图片张量
            ax.imshow(img.numpy())
        else:  # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes


"""---------------------------------fashion_mnist----------------------------------------------"""


def load_data_fashion_mnist(batch_size=256, num_workers=4, resize=None):
    transform = [transforms.ToTensor()]
    if resize:
        transform.insert(0, transforms.Resize(resize))
    transform = transforms.Compose(transform)

    train_set = torchvision.datasets.FashionMNIST(root="./dataset/FashionMNIST", train=True, transform=transform,
                                                  download=True)
    test_set = torchvision.datasets.FashionMNIST(root="./dataset/FashionMNIST", train=False, transform=transform,
                                                 download=True)

    # 第0个分类的第0张图片的大小：1, 28, 28 channel=1, hw=28,28
    print(len(train_set), len(test_set), train_set[0][0].shape)

    # X.shape [18, 1, 28, 28]  y.shape 18
    """
    X, y = next(iter(DataLoader(train_set, batch_size=18)))
    show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))
    """
    return (DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers),
            DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers))


"""---------------------------------synthetic_data_linear_model----------------------------------------------"""


def synthetic_data_linear_model(w, b, num_examples):
    """为了更加直观的看到线性回归真实值和预测值之间的差距"""
    # Example : tensor([ 0.9554, -0.7643, -0.2272, -2.6271,  0.2974,  1.0444])
    X = torch.normal(mean=0, std=1, size=(num_examples, len(w)))
    # X @ w : 矩阵乘法，tensor会把他们看作 1xn * n*1 得到一个标量, b 会使用广播机制
    Y = X @ w + b
    # Y = torch.matmul(X, w) + b
    # 加入噪声
    Y += torch.normal(mean=0, std=0.01, size=Y.shape)
    return X, Y.reshape((-1, 1))


"""---------------------------------测试一个加载batch所需要的时间----------------------------------------------"""


def testTime(dataset="mnist"):
    """测试一个加载batch所需要的时间"""
    if dataset == "mnist":
        train_loader, test_loader = load_data_mnist(batch_size=32, num_workers=4)
    else:
        train_loader, test_loader = load_data_fashion_mnist(batch_size=256, num_workers=4)
    time_start = time.time()
    for X, y in train_loader:
        continue
    time_end = time.time()
    print(f'一个batch运行时间：{time_end - time_start:.2f} sec')
