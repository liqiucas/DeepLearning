import numpy as np
import matplotlib.pyplot as plt


def load_dataset():
    dataset = np.loadtxt("../dataset/perceptron2.txt", dtype=float)
    X = dataset[:, :-1]
    y = dataset[:, -1]
    return (X, y)


def PLA(X, y, W):
    """原生的 PLA 算法：
        1. 初始化参数
        2. 逐个遍历，如果当前样本分类错误，则修正 <=> 每修改一次就统计有没有错误，有错误咱们就从当前点开始
        3. 修正后所有数据后重新遍历，直到没有错误"""
    epoch = 0
    while True:
        for i in range(len(y)):
            sign = np.sign(np.matmul(X[i], W) * y[i])
            if sign <= 0:
                """修正"""
                W = W + y[i] * X[i]
        # 每次修正后查看准错误的个数，如果错误个数=0，则中止
        err = error(X, y, W)
        print(f'当前{epoch}轮次的错误率：{err / len(y):.2f}')
        epoch = epoch + 1

        if err == 0:
            return W


def error(X, y, W):
    count = 0
    for i in range(len(y)):
        sign = np.sign(np.matmul(X[i], W) * y[i])
        if sign <= 0:
            count = count + 1
    return count


def show_img(X, y, W):
    """W 的法向量方向就是分割线
        (x, y) 它的法向量就是 (y,-x) 和(-y, x) 自己调试一下是哪个方向
    """
    W_orth = [-W[1], W[0]]
    ax = plt.axes()
    ax.arrow(0, 0, *W_orth, color='g', linewidth=2.0, head_width=0.20, head_length=0.25)
    for i in range(len(y)):
        if y[i] > 0:
            plt.scatter(X[i, 0], X[i, 1], c='r', marker='*')
        else:
            plt.scatter(X[i, 0], X[i, 1], c='b', marker='.')
    plt.show()


if __name__ == '__main__':
    X, y = load_dataset()
    W = np.zeros(len(X[0]))
    W = PLA(X, y, W)
    print(W)
    show_img(X, y, W)
