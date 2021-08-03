# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import torch
import time
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision

from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision import models

import os

from PIL import Image

import sys
import d2lzh_pytorch as d2l

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_cifar10(is_train, augs, batch_size, root='~/data/CIFAR', num_workers=0):
    dataset = torchvision.datasets.CIFAR10(root=root, train=is_train, transform=augs, download=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=num_workers)


def train_with_data_aug(train_augs, test_augs, lr=0.001):
    batch_size, net = 256, d2l.resnet18(10)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = torch.nn.CrossEntropyLoss()
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    d2l.train(net, train_iter, test_iter, num_epochs=10, loss=loss, optimizer=optimizer,
              device=device)


def train_fine_tuning(net, optimizer, batch_size=128, num_epochs=5):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_augs = transforms.Compose([
        transforms.RandomResizedCrop(size=224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    test_augs = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        normalize
    ])

    train_iter = DataLoader(ImageFolder(".\\data\\hotdog\\train", transform=train_augs), batch_size=batch_size,
                            shuffle=True)
    test_iter = DataLoader(ImageFolder(".\\data\\hotdog\\test", transform=test_augs), batch_size=batch_size,
                           shuffle=True)
    loss = torch.nn.CrossEntropyLoss()
    d2l.train(train_iter=train_iter, test_iter=test_iter, net=net, loss=loss, optimizer=optimizer, device=device,
              num_epochs=num_epochs)


if __name__ == '__main__':
    print()
    d2l.set_figsize()
    img = Image.open('data/catdog.jpg')

    dog_bbox, cat_bbox = [60, 45, 378, 516], [400, 112, 655, 493]

    fig = d2l.plt.imshow(img)
    fig.axes.add_patch(d2l.bbox_to_rect(dog_bbox, 'blue'))
    fig.axes.add_patch(d2l.bbox_to_rect(cat_bbox, 'red'))

    d2l.plt.show()

    # img =


def test_train_fine_tuning():
    data_dir = ".\\data"

    # 创建两个`ImageFolder`实例来分别读取训练数据集和测试数据集中的所有图像文件。
    train_imgs = ImageFolder(data_dir + "\\hotdog\\train")
    test_imgs = ImageFolder(data_dir + "\\hotdog\\test")

    hotdogs = [train_imgs[i][0] for i in range(8)]
    not_hotdogs = [train_imgs[-i - 1][0] for i in range(8)]
    d2l.show_images(hotdogs + not_hotdogs, 2, 8, scale=1.4)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_augs = transforms.Compose([
        transforms.RandomResizedCrop(size=224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    test_augs = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        normalize
    ])

    pretrained_net = models.resnet18(pretrained=True)
    print(pretrained_net.fc)

    pretrained_net.fc = nn.Linear(512, 2)
    print(pretrained_net.fc)

    output_params = list(map(id, pretrained_net.fc.parameters()))
    feature_params = filter(lambda p: id(p) not in output_params, pretrained_net.parameters())

    lr = 0.01
    optimizer = optim.SGD(params=[{'params': feature_params},
                                  {'params': pretrained_net.fc.parameters(), 'lr': lr * 10}],
                          lr=lr,
                          weight_decay=0.001
                          )
    train_fine_tuning(pretrained_net, optimizer)

    # test_imgs2 = ImageFolder(data_dir + "\\testImg")

    y = pretrained_net(test_augs(Image.open("data/testImg/test03.jpg")).view([1, 3, 224, -1]).to(device))
    print(y)


def test_convert_img2():
    all_images = torchvision.datasets.CIFAR10(train=True, root="~/data/CIFAR", download=True)
    d2l.show_images([all_images[i][0] for i in range(32)], 4, 8, scale=0.8)

    flip_aug = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor()
    ])

    no_aug = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    num_workers = 0 if sys.platform.startswith('win32') else 4
    train_with_data_aug(flip_aug, no_aug)


def test_convert_img():
    d2l.set_figsize()
    img = Image.open("data/cat1.jpg")
    d2l.plt.imshow(img)
    d2l.plt.show()

    # d2l.apply(img, torchvision.transforms.RandomHorizontalFlip())
    # d2l.apply(img, torchvision.transforms.RandomVerticalFlip())

    # scale 裁剪10~100% 的区域
    # ratio 宽高比 0.5 ~ 2
    #
    shape_aug = torchvision.transforms.RandomResizedCrop(200, scale=(0.1, 1), ratio=(0.1, 2))
    # d2l.apply(img, shape_aug)

    # d2l.apply(img, torchvision.transforms.ColorJitter(brightness=0.5))

    # d2l.apply(img, torchvision.transforms.ColorJitter(hue=0.5))

    # d2l.apply(img, torchvision.transforms.ColorJitter(contrast=0.5))

    color_aug = torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
    # d2l.apply(img, color_aug)

    augs = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(), color_aug, shape_aug])
    d2l.apply(img, augs)
