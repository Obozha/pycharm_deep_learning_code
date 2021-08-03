import CNN as cn
import CNN_LeNet
import CNN_AlexNet
import GoogleNet as gn
import VGG as v
import time
import torch
from torch import nn, optim
import d2lzh_pytorch as d2l
import NIN as nin
import Normalization as nor
import ResNet as rn
import DenseNet as dn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
conv_arch = ((1, 1, 64), (1, 64, 128), (2, 128, 256), (2, 256, 512), (2, 512, 512))
fc_features = 512 * 7 * 7  # c * w * h
fc_hidden_units = 4096  # 任意

radio = 8
small_conv_arch = [(1, 1, 64 // radio), (1, 64 // radio, 128 // radio), (2, 128 // radio, 256 // radio),
                   (2, 256 // radio, 512 // radio), (2, 512 // radio, 512 // radio)]


def testResNet():
    blk = rn.Residual(3, 3)
    X = torch.rand(4, 3, 6, 6)
    print(blk(X).shape)
    blk = rn.Residual(3, 6, use_1x1conv=True, stride=2)
    print(blk(X).shape)

    net = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

    net.add_module("resnet_block1", rn.resnet_block(64, 64, 2, first_block=True))
    net.add_module("resnet_block2", rn.resnet_block(64, 128, 2))
    net.add_module("resnet_block3", rn.resnet_block(128, 256, 2))
    net.add_module("resnet_block4", rn.resnet_block(256, 512, 2))

    net.add_module("global_avg_pool", d2l.GlobalAvgPool2d())
    net.add_module("fc", nn.Sequential(d2l.FlattenLayer(), nn.Linear(512, 10)))

    X = torch.rand((1, 1, 224, 224))
    for name, layer in net.named_children():
        X = layer(X)
        print(name, 'output shape:', X.shape)

    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size, resize=96)

    lr, num_epochs = 0.001, 5
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    d2l.train_ch5(net, train_iter, test_iter, num_epochs, batch_size, optimizer)

    X, y = iter(test_iter).next()
    true_labels = d2l.get_fashion_mnist_labels(y.numpy())
    pred_labels = d2l.get_fashion_mnist_labels(net(X.to(device)).argmax(dim=1).cpu().numpy())
    titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
    d2l.show_fashion_mnist(X[0:9], titles[0:9])


def testNormaliztion():
    net = nn.Sequential(nn.Conv2d(1, 6, 5),  # in_channels, out_channels,kernel_size
                        nor.BatchNorm(6, num_dims=4),
                        nn.Sigmoid(),
                        nn.MaxPool2d(2, 2),  # kernel_size, stride
                        nn.Conv2d(6, 16, 5),
                        nor.BatchNorm(16, num_dims=4),
                        nn.Sigmoid(),
                        nn.MaxPool2d(2, 2),
                        d2l.FlattenLayer(),
                        nn.Linear(16 * 4 * 4, 120),
                        nor.BatchNorm(120, num_dims=2),
                        nn.Sigmoid(),
                        nn.Linear(120, 84),
                        nor.BatchNorm(84, num_dims=2),
                        nn.Sigmoid(),
                        nn.Linear(84, 10)
                        )

    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

    lr, num_epochs = 0.001, 5
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    d2l.train_ch5(net, train_iter, test_iter, num_epochs, batch_size, optimizer)

    X, y = iter(test_iter).next()
    true_labels = d2l.get_fashion_mnist_labels(y.numpy())
    pred_labels = d2l.get_fashion_mnist_labels(net(X.to(device)).argmax(dim=1).cpu().numpy())
    titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
    d2l.show_fashion_mnist(X[0:9], titles[0:9])


def testGoogleNet():
    b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                       nn.ReLU(),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                       nn.Conv2d(64, 192, kernel_size=3, padding=1),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    b3 = nn.Sequential(
        gn.Inception(192, 64, (96, 128), (16, 32), 32),
        gn.Inception(256, 128, (128, 192), (32, 96), 64),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

    b4 = nn.Sequential(
        gn.Inception(480, 192, (96, 208), (16, 48), 64),
        gn.Inception(512, 160, (112, 224), (24, 64), 64),
        gn.Inception(512, 128, (128, 256), (24, 64), 64),
        gn.Inception(512, 112, (144, 288), (32, 64), 64),
        gn.Inception(528, 256, (160, 320), (32, 128), 128),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

    b5 = nn.Sequential(
        gn.Inception(832, 256, (160, 320), (32, 128), 128),
        gn.Inception(832, 384, (192, 384), (48, 128), 128),
        d2l.GlobalAvgPool2d()
    )

    net = nn.Sequential(b1, b2, b3, b4, b5, d2l.FlattenLayer(), nn.Linear(1024, 10))
    # X = torch.rand(1, 1, 96, 96)
    # for blk in net.children():
    #     X = blk(X)
    #     print("output", X.shape)

    batch_size = 50
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size, resize=224)

    lr, num_epochs = 0.001, 5
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    d2l.train_ch5(net, train_iter, test_iter, num_epochs, batch_size, optimizer)

    X, y = iter(test_iter).next()
    true_labels = d2l.get_fashion_mnist_labels(y.numpy())
    pred_labels = d2l.get_fashion_mnist_labels(net(X.to(device)).argmax(dim=1).cpu().numpy())
    titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
    d2l.show_fashion_mnist(X[0:9], titles[0:9])


def testNiN():
    net = nn.Sequential(
        nin.nin_block(1, 96, kernel_size=11, stride=4, padding=0),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nin.nin_block(96, 256, kernel_size=5, stride=1, padding=2),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nin.nin_block(256, 384, kernel_size=3, stride=1, padding=1),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Dropout(0.5),
        # 标签类别数10
        nin.nin_block(384, 10, kernel_size=3, stride=1, padding=1),
        d2l.GlobalAvgPool2d(),
        # 将四维的输出转成二维 ===> batchsize 10
        d2l.FlattenLayer()
    )
    # X = torch.rand(1, 1, 224, 224)
    # for name, blk in net.named_children():
    #     X = blk(X)
    #     print(name, 'output shape', X.shape)

    net = v.vgg(small_conv_arch, fc_features // radio, fc_hidden_units // radio)
    print(net)

    batch_size = 128
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=5, resize=50)

    lr, num_epochs = 0.001, 5
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    d2l.train_ch5(net, train_iter, test_iter, num_epochs, batch_size, optimizer)

    X, y = iter(test_iter).next()
    true_labels = d2l.get_fashion_mnist_labels(y.numpy())
    pred_labels = d2l.get_fashion_mnist_labels(net(X.to(device)).argmax(dim=1).cpu().numpy())
    titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
    d2l.show_fashion_mnist(X[0:9], titles[0:9])


def testVGG():
    # net = v.vgg(conv_arch, fc_features, fc_hidden_units)
    # X = torch.rand(1, 1, 224, 224)
    #
    # for name, blk in net.named_children():
    #     X = blk(X)
    #     print(name, 'output shape', X.shape)

    net = v.vgg(small_conv_arch, fc_features // radio, fc_hidden_units // radio)
    print(net)

    batch_size = 128
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size, resize=224)

    lr, num_epochs = 0.001, 5
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    d2l.train_ch5(net, train_iter, test_iter, num_epochs, batch_size, optimizer)

    X, y = iter(test_iter).next()
    true_labels = d2l.get_fashion_mnist_labels(y.numpy())
    pred_labels = d2l.get_fashion_mnist_labels(net(X.to(device)).argmax(dim=1).cpu().numpy())
    titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
    d2l.show_fashion_mnist(X[0:9], titles[0:9])


def testAlexNet():
    net = CNN_AlexNet.AlexNet()
    print(net)

    batch_size = 128
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size, resize=224)

    lr, num_epochs = 0.001, 1
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    d2l.train_ch5(net, train_iter, test_iter, num_epochs, batch_size, optimizer)

    X, y = iter(test_iter).next()
    true_labels = d2l.get_fashion_mnist_labels(y.numpy())
    pred_labels = d2l.get_fashion_mnist_labels(net(X.to(device)).argmax(dim=1).cpu().numpy())
    titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
    d2l.show_fashion_mnist(X[0:9], titles[0:9])


def testCNNLeNet():
    net = CNN_LeNet.LeNet()
    print(net)

    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

    lr, num_epochs = 0.001, 5
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    d2l.train_ch5(net, train_iter, test_iter, num_epochs, batch_size, optimizer)

    X, y = iter(test_iter).next()
    true_labels = d2l.get_fashion_mnist_labels(y.numpy())
    pred_labels = d2l.get_fashion_mnist_labels(net(X.to(device)).argmax(dim=1).cpu().numpy())
    titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
    d2l.show_fashion_mnist(X[0:9], titles[0:9])


def test1():
    X = torch.tensor(([0, 1, 2], [3, 4, 5], [6, 7, 8]))
    K = torch.tensor(([0, 1], [2, 3]))
    Y = cn.corr2d(X, K)
    print(Y)

    X = torch.ones(5, 8)
    X[:, 2:6] = 0
    print(X)

    K = torch.tensor([[1, -1]])
    print(K)
    Y = cn.corr2d(X, K)
    print(Y)

    conv2d = cn.Conv2D(kernel_size=(1, 2))
    step = 20
    lr = 0.01
    for i in range(step):
        Y_hat = conv2d(X)
        l = ((Y_hat - Y) ** 2).sum()
        l.backward()

        # 梯度下降
        conv2d.weight.data -= lr * conv2d.weight.grad
        conv2d.bias.data -= lr * conv2d.bias.grad

        # 梯度清0
        conv2d.weight.grad.fill_(0)
        conv2d.bias.grad.fill_(0)

        if (i + 1) % 5 == 0:
            print('Step %d,loss %.3f' % (i + 1, l.item()))

        print("weight", conv2d.weight.data)
        print("weight", conv2d.bias.data)

    # 注意这里是两侧分别填充1行或列，所以在两侧一共填充2行或列


def test2():
    conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
    X = torch.rand(8, 8)
    print(cn.comp_conv2d(conv2d, X).shape)

    conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
    print(cn.comp_conv2d(conv2d, X).shape)

    conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
    print(cn.comp_conv2d(conv2d, X).shape)

    X = torch.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
    print(X.shape)

    K = torch.tensor([[[0, 1], [2, 3]], [[1, 2], [3, 4]]])
    print(K.shape)

    print(cn.corr2d_multi_in(X, K))

    print(K.shape)
    K = torch.stack([K, K + 1, K + 2])
    print(K.shape)
    print(K)

    print(cn.corr2d_multi_in_out(X, K))

    X = torch.rand(3, 3, 3)
    K = torch.rand(2, 3, 1, 1)

    Y1 = cn.corr2d_multi_in_out_1x1(X, K)
    Y2 = cn.corr2d_multi_in_out(X, K)

    print((Y1 - Y2).norm().item() < 1e-6)

    X = torch.tensor(([0, 1, 2], [3, 4, 5], [6, 7, 8]))
    print(cn.pool2d(X, (2, 2)))
    print(cn.pool2d(X, (2, 2), 'avg'))


def test3():
    X = torch.arange(16, dtype=torch.float).view((1, 1, 4, 4))
    print(X)

    pool2d = nn.MaxPool2d(3)
    print(pool2d(X))

    pool2d = nn.MaxPool2d(3, padding=1, stride=2)
    print(pool2d(X))

    pool2d = nn.MaxPool2d((2, 4), padding=(1, 2), stride=(2, 3))
    print(pool2d(X))

    X = torch.cat((X, X + 1), dim=1)
    print(X)

    pool2d = nn.MaxPool2d(3, padding=1, stride=2)
    print(pool2d(X))


def testDenseNet():
    blk = dn.DenseBlock(2, 3, 10)
    X = torch.rand(4, 3, 8, 8)
    Y = blk(X)
    print(Y.shape)

    net = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

    num_channels, growth_rate = 64, 32  # num_channels为当前的通道数
    num_convs_in_dense_blocks = [4, 4, 4, 4]

    for i, num_convs in enumerate(num_convs_in_dense_blocks):
        DB = dn.DenseBlock(num_convs, num_channels, growth_rate)
        net.add_module("DenseBlock_%d" % i, DB)
        # 上一个稠密块的输出通道数
        num_channels = DB.out_channels
        # 在稠密快之间加入通道数减半的过渡层
        if i != len(num_convs_in_dense_blocks) - 1:
            net.add_module("transition_block_%d" % i, dn.transition_block(num_channels, num_channels // 2))
            num_channels = num_channels // 2

    net.add_module("BN", nn.BatchNorm2d(num_channels))
    net.add_module("relu", nn.ReLU())
    net.add_module("global_avg_pool", d2l.GlobalAvgPool2d())  # GlobalAvgPool2d的输出
    net.add_module("fc", nn.Sequential(d2l.FlattenLayer(), nn.Linear(num_channels, 10)))

    nn.Linear(num_channels, 10)

    # X = torch.rand((1, 1, 96, 96))
    # for name, layer in net.named_children():
    #     X = layer(X)
    #     print(name, 'output shape:\t', X.shape)

    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size, resize=96)

    lr, num_epochs = 0.001, 5
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    d2l.train_ch5(net, train_iter, test_iter, num_epochs, batch_size, optimizer)

    X, y = iter(test_iter).next()
    true_labels = d2l.get_fashion_mnist_labels(y.numpy())
    pred_labels = d2l.get_fashion_mnist_labels(net(X.to(device)).argmax(dim=1).cpu().numpy())
    titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
    d2l.show_fashion_mnist(X[0:9], titles[0:9])
