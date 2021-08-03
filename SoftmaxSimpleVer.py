import torch
import numpy as np
import d2lzh_pytorch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_hiddens)), dtype=torch.float)
b1 = torch.zeros(num_hiddens, dtype=torch.float)

W2 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens, num_outputs)), dtype=torch.float)
b2 = torch.zeros(num_outputs, dtype=torch.float)

params = [W1, b1, W2, b2]


def relu(X):
    return torch.max(input=X, other=torch.tensor(0.0))


def net(X):
    X = X.view((-1, num_inputs))
    H = relu(torch.matmul(X, W1) + b1)
    return torch.matmul(H, W2) + b2

def doSoftmaxSimpleVer():
    for param in params:
        param.requires_grad_(requires_grad=True)

    loss = torch.nn.CrossEntropyLoss()
    num_epochs, lr = 5, 100.0
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)
    X, y = iter(test_iter).next()
    true_labels = d2l.get_fashion_mnist_labels(y.numpy())
    pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
    titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
    d2l.show_fashion_mnist(X[0:9], titles[0:9])
