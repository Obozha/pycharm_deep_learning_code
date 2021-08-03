import torch
import numpy as np
import random
import numpy as np


def write_data(labels, features):
    label_fo = open("labels.txt", "w")  # 写入磁盘
    label_fo.write(str(labels))
    label_fo.close()

    features_fo = open("features.txt", "w")
    features_fo.write(str(features))
    features_fo.close()


def data_iter(batch_size, features, labels):
    num_examples = len(features)  # 1000
    indices = list(range(num_examples))  # [0,1,2...1000]
    random.shuffle(indices)  # 样本的读取顺序是随机的
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i:min(i + batch_size, num_examples)])  # 最后一次可能不足一个batch
        yield features.index_select(0, j), labels.index_select(0, j)


def linreg(x, w, b):
    return torch.mm(x.double(), w.double()) + b


def squared_loss(y_hat, y):
    return (y_hat - y.view(y_hat.size())) ** 2 / 2


def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size  # 注意这里更改param时用的param.data


def exc_ml():
    # ====================================================准备数据=========================================================
    batch_size = 10  # 批次
    num_inputs = 2  # 两个特征数
    num_examples = 1000  # 1000个样本
    true_w = [2, -3.4]  # 两个权重
    true_b = 4.2  # 偏差

    features = torch.from_numpy(
        np.random.normal(0, 1, (num_examples, num_inputs)))
    # 正态分布得到2*1000个特征数的tensor，均值为0，scale为1

    labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b  # 真实值y（真实labels） 矢量计算
    labels += torch.from_numpy(np.random.normal(0, 0.01, size=labels.size()))  # 真实值（真实labels）y+噪声
    torch.set_printoptions(threshold=np.inf)
    write_data(labels, features)

    w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32)
    b = torch.zeros(1, dtype=torch.float32)

    w.requires_grad_(requires_grad=True)
    b.requires_grad_(requires_grad=True)

    lr = 0.03  # 学习率
    num_epochs = 3  # 周期个数

    # ====================================================准备model=======================================================
    net = linreg  # torch.mm(x.double(),w.double())+b

    # ====================================================准备loss函数=====================================================
    loss = squared_loss  # (y_hat-y.view(y_hat.size())) ** 2 / 2

    # ====================================================训练model=========================================================
    for epoch in range(3):
        # 在某一个迭代周期中，会使用训练数据集中所有样本一次（假设样本数能够被批量大小整除）。

        # x和y分别是小批量样本的特征和标签
        for x, y in data_iter(batch_size, features, labels):
            l = loss(net(x, w, b), y).sum()  # l是有关小批量x和y的损失
            l.backward()  # 小批量的损失对模型参数求梯度
            sgd([w, b], lr, batch_size)  # 使用小批量随机梯度下降迭代模型参数

            # param.data-=lr*param.grad/batch_size

            # 不要忘了梯度清零
            w.grad.data.zero_()
            b.grad.data.zero_()
        train_l = loss(net(features, w, b), labels)
        print("epoch %d,loss %f" % (epoch + 1, train_l.mean().item()))

    print(true_w, '\n', w)
    print(true_b, '\n', b)
