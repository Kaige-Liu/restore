"""
5.30 19:53修改
1. 添加了验证集（和训练集一样）
2. 从E:\code\deepsc_hiding_torch\checkpoints\image_hiding\2024-05-30-14-23\checkpoint_199_0.pth继续训练
3. CustomLinearLayer的bn和relu忘了删了，这里删除了(删不了了，因为之前的模型已经里是有的）
4. 现在跑20000个epoch
5. 不小心用了mse，改回第一范数了
————最新模型为E:\code\deepsc_hiding_torch\checkpoints\image_hiding\2024-05-31-00_28_25\checkpoint_915_0.pth
—————validate loss: 0.19076444790698588
"""
"""
5.31 10:07修改
1. embedding层的输出改为bn+sigmod 希望能改善图像蒙上雾的问题
2. 范数改回mse
3. 从头开始训练
"""
"""
5.31 16:02
1. 将embedding层的输出改为bn+sigmod 
2. 删除全连接层的bn和relu
3. 从头开始训练
"""
"""
所有的组件恢复原来的状态（没有bn和激活函数）
去除冻结操作
lambda_4恢复成0.01
学习率恢复成0.001
第一范数恢复成.mean(abs(x - y))
全连接层从12改成5
————E:\code\deepsc_hiding_torch\checkpoints\image_hiding\2024-06-03-10_17_33\checkpoint_535_0.pth 出现抖动较大
————0.05085635371506214
————误码率0.86 19.82718519595897
"""
"""
从E:\code\deepsc_hiding_torch\checkpoints\image_hiding\2024-06-03-10_17_33\checkpoint_473_0.pth继续训练
学习率改成0.00001
"""


import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.autograd import Function
import math

from torch.nn import Flatten
from torch.nn.init import xavier_uniform_
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.utils.tensorboard import SummaryWriter

import pickle

from torchvision.datasets import CIFAR10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CustomConvBlock(nn.Module):  # 那个超级复杂的卷积块
    def __init__(self, w, d, c):
        super(CustomConvBlock, self).__init__()
        self.conv_1_1132 = nn.Conv2d(c, 32, 1, stride=1, padding=0)
        # self.bn1 = nn.BatchNorm2d(32)  # 这个是通道数

        self.conv_2_1132 = nn.Conv2d(c, 32, 1, stride=1, padding=0)
        # self.bn2_1 = nn.BatchNorm2d(32)
        self.conv_2_3332 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        # self.bn2_2 = nn.BatchNorm2d(32)

        self.conv_3_1132 = nn.Conv2d(c, 32, 1, stride=1, padding=0)
        # self.bn3_1 = nn.BatchNorm2d(32)
        self.conv_3_3332 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        # self.bn3_2 = nn.BatchNorm2d(32)
        self.conv_3_3332_2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        # self.bn3_3 = nn.BatchNorm2d(32)

        self.conv111 = nn.Conv2d(96, c, 1, stride=1, padding=0)
        # self.bn111 = nn.BatchNorm2d(c)

        # self.relu = nn.Sigmoid()

    def forward(self, x):  # 这里需要改一下(改完了)
        # print("x shape:", x.shape)
        x1 = self.conv_1_1132(x)

        x21 = self.conv_2_1132(x)
        x2 = self.conv_2_3332(x21)

        x31 = self.conv_3_1132(x)
        x32 = self.conv_3_3332(x31)
        x3 = self.conv_3_3332_2(x32)

        b1 = torch.cat((x1, x2, x3), 1)  # 这里是将三个张量拼接起来，dim=1，也就是在第2个维度C上拼接
        b2 = self.conv111(b1)

        out = x + b2
        return b1, b2, out


class Encoder(nn.Module):  # 这个是信息隐藏的encoder，负责将二进制图片编码成一个32x32x48的隐藏文本图像的特征
    def __init__(self):
        super(Encoder, self).__init__()

        self.convBlock1 = CustomConvBlock(32, 32, 1)  # 卷积块
        self.convBlock2 = CustomConvBlock(32, 32, 1)  # 卷积块
        self.conv1124 = nn.Conv2d(1, 24, 1, stride=1, padding=0)
        # self.bn1124 = nn.BatchNorm2d(24)

        self.convBlock3 = CustomConvBlock(32, 32, 24)
        self.convBlock4 = CustomConvBlock(32, 32, 24)
        self.conv1148 = nn.Conv2d(24, 48, 1, stride=1, padding=0)
        # self.bn1148 = nn.BatchNorm2d(48)

        # self.relu = nn.Sigmoid()

    def forward(self, x):
        _, _, x = self.convBlock1(x)
        _, _, x = self.convBlock2(x)
        x = self.conv1124(x)
        _, _, x = self.convBlock3(x)
        _, _, x = self.convBlock4(x)
        x = self.conv1148(x)
        return x


class Decoder(nn.Module):  # 将特征（但是目前还缺少一个reshape 直接就当做是reshape之后的尺寸了，之后得改一下
    def __init__(self):
        super(Decoder, self).__init__()
        self.convBlock1 = CustomConvBlock(32, 32, 48)  # 卷积块
        self.convBlock2 = CustomConvBlock(32, 32, 48)  # 卷积块
        self.conv1124 = nn.Conv2d(48, 24, 1, stride=1, padding=0)
        # self.bn1124 = nn.BatchNorm2d(24)

        self.convBlock3 = CustomConvBlock(32, 32, 24)
        self.convBlock4 = CustomConvBlock(32, 32, 24)
        self.conv111 = nn.Conv2d(24, 1, 1, stride=1, padding=0)
        # self.bn111 = nn.BatchNorm2d(1)

        # self.relu = nn.Sigmoid()
        # self.sigmod = nn.Sigmoid()

    def forward(self, x, batch_size):
        # 首先将x reshape成32x32x48
        x = x.view(batch_size, 48, 32, 32)
        _, _, x = self.convBlock1(x)
        _, _, x = self.convBlock2(x)
        # print("经过2个块之后的输出：", x.shape)
        x = self.conv1124(x)
        _, _, x = self.convBlock3(x)
        _, _, x = self.convBlock4(x)
        x = self.conv111(x)  # 最后输出不用relu 用sigmod
        return x


def get_image(image_path):
    img_PIL = Image.open(image_path)
    img_tensor = transforms.ToTensor()(img_PIL)
    new_tensor = img_tensor.unsqueeze(0)  # 增加一个batch_size的维度
    return new_tensor


class Embedder(nn.Module):  # 嵌入层，负责将隐藏文本图像的特征注入到载体图像中，得到128x128x3的图像，要求很像载体图像
    def __init__(self):
        super(Embedder, self).__init__()
        self.convBlock_star = CustomConvBlock(128, 128, 3)  # 卷积块
        self.convBlock2 = CustomConvBlock(128, 128, 6)
        self.conv113 = nn.Conv2d(6, 3, 1, stride=1, padding=0)
        # self.bn113 = nn.BatchNorm2d(3)
        # self.sigmod = nn.Sigmoid()


    def forward(self, x, c, batch_size):  # 接受的输入x是32x32x48的
        # 首先reshape成128x128x3
        x = x.view(batch_size, 3, 128, 128)
        b1, b2, x = self.convBlock_star(x)
        # 下面将这个和另一个128x128x3的载体图片c拼接起来
        # 现在的问题就是x是[1, 3, 128, 128]的，c是[63, 3, 128, 128]的
        # 所以还是得让x是所有的隐藏信息的数据集
        # print("------------------")
        # print("x shape:", x.shape)
        # print("c shape:", c.shape)
        x = torch.cat((x, c), 1)
        _, _, x = self.convBlock2(x)
        x = self.conv113(x)
        return x

class CustomLinearLayer(nn.Module):
    def __init__(self, num_channels, num_outputs):
        super(CustomLinearLayer, self).__init__()
        self.num_channels = num_channels
        self.num_outputs = num_outputs
        self.weights = nn.Parameter(torch.randn(num_outputs, num_channels))  # 定义权重矩阵，初始值为随机数
        xavier_uniform_(self.weights)  # 这个位置不是很确定对不对
        # self.bn = nn.BatchNorm2d(num_outputs)
        # self.relu = nn.Sigmoid()

    def forward(self, x):
        batch_size, _, height, width = x.size()
        # print("x.shape: ", x.shape)
        x = x.view(batch_size, self.num_channels, -1)  # 将每个通道视为一个整体
        # print("x.shape: ", x.shape)
        # x = x.permute(0, 2, 1)  # 调整维度顺序，变成 (batch_size, height*width, num_channels)
        # print("x.shape: ", x.shape)
        output = torch.matmul(self.weights, x)  # 矩阵乘法，得到输出
        # print("output.shape: ", output.shape)
        output = output.view(batch_size, self.num_outputs, height, width)  # 调整输出形状
        # output = self.relu(self.bn(output))
        # print("weights:", self.weights)
        return output, self.weights


class Extractor(nn.Module):
    def __init__(self, N):
        super(Extractor, self).__init__()
        self.convBlock1 = CustomConvBlock(128, 128, N)  # 卷积块
        self.convBlock2 = CustomConvBlock(128, 128, N)  # 卷积块
        self.conv113 = nn.Conv2d(N, 3, 1, stride=1, padding=0)
        # self.bn113 = nn.BatchNorm2d(3)
        # self.sigmod = nn.Sigmoid()


    def forward(self, x):  # x是128x128xN的
        _, _, x = self.convBlock1(x)
        _, _, x = self.convBlock2(x)
        x = self.conv113(x)
        return x


# # 读取图片并转换为张量
# img_path = "D:\deeplearning\deepsc_hiding_torch\deepsc_hiding_torch\\binary_image.png"
# img_PIL = Image.open(img_path)
# img_tensor = transforms.ToTensor()(img_PIL)
# print("Image tensor shape:", img_tensor.shape)
#
# lkg = Encoder()
# lkg.to(device)
# new_tensor = img_tensor.unsqueeze(0).to(device)  # 增加一个batch_size的维度
#
# encoder_output = lkg(new_tensor)
# print("Output shape:", encoder_output.shape)
#
# # print("---------------------------")
# # lkg2 = Decoder()
# # decoder_output = lkg2(encoder_output)
# # print("decoder_output:", decoder_output.shape)
#
# lkg3 = Embedder()
# lkg3.to(device)
# encoder_output = encoder_output.to(device)
# embedder_output = lkg3(encoder_output)
# print("embedder_output:", embedder_output.shape)
#
# lkg4 = Invarianve(4).to(device)
# # 使用GPU加速
# embedder_output = embedder_output.to(device)
# invariance_output = lkg4(embedder_output, 4)
# print("invariance_output:", invariance_output.shape)
#
#
#
# # new_tensor = embedder_output.squeeze(dim=0)
# # # 展示图片
# # new_image = transforms.ToPILImage()(new_tensor)
# # new_image.show()  # 显示图片


# class En(nn.Module):
#     def __init__(self):
#         super(En, self).__init__()
#         self.encoder = Encoder()
#         self.embedder = Embedder()

class net(nn.Module):  # 整体的网络，包括5个组件
    def __init__(self, N):
        super(net, self).__init__()
        self.encoder = Encoder()
        self.embedder = Embedder()
        self.invariance = CustomLinearLayer(3, N)
        self.extractor = Extractor(N)
        self.decoder = Decoder()


# # 定义损失函数
# criterion = nn.CrossEntropyLoss(reduction='sum')


def train(N, model, batch_size, train_loader, hiding_data_loader, optimizer, lambda_1, lambda_2, lambda_3, lambda_4):
    model.train()

    for batch, (data, _) in enumerate(train_loader):
        print("batch:", batch)
        for ct, (hiding_data, _) in enumerate(hiding_data_loader):
            print("ct:", ct)
            data = data.to(device)
            hiding_data = hiding_data.to(device)

            encoder_output = model.encoder(hiding_data)
            embedder_output = model.embedder(encoder_output, data, batch_size)  # 最后这个是batch_size
            invariance_output, weights = model.invariance(embedder_output)  # 这里的N是冗余，所以之后应该加上一个正则化项，提高这里的鲁棒性
            extractor_output = model.extractor(invariance_output)
            decoder_output = model.decoder(extractor_output, batch_size)  # 最后这个是batch_size

            # loss = criterion(embedder_output, data)
            loss_1 = lambda_1 * l1_norm(decoder_output, hiding_data)
            print("loss_1:", loss_1.shape)
            print("l1:", l1_norm(decoder_output, hiding_data))
            print("loss_1:", loss_1)
            loss_2 = lambda_2 * l1_norm(embedder_output, data)
            print("loss_2:", loss_2.shape)
            b1w, b2w, _ = model.embedder.convBlock_star(encoder_output.view(batch_size, 3, 128, 128))
            b1m, b2m, _ = model.embedder.convBlock_star(embedder_output.view(batch_size, 3, 128, 128))
            g1w = gram_for_batch(b1w)
            print("g1w:", g1w.shape)
            g1m = gram_for_batch(b1m)
            g2w = gram_for_batch(b2w)
            g2m = gram_for_batch(b2m)
            loss_3 = lambda_3 * 0.5 * (l1_norm(g1w, g1m) + l1_norm(g2w, g2m))
            print("loss_3:", loss_3.shape)

            # 第一个求和项，这里是将invariance_output的所有元素全都平方了（存疑）
            term1 = torch.sum((1 - invariance_output ** 2) ** 2, dim=2)  # invariance_output是[bs, N, 128*128],结果是[bs, N]
            print("term1:", term1.shape)
            # 第二个求和项
            term2 = torch.sum(weights ** 2, dim=1).unsqueeze(0)  # 在第0维度增加一个维度，结果[1, bs, N]
            print("term2:", term2.shape)
            # 计算最终的 P
            p = torch.sum(term1) * torch.sum(term2)  # 这里对吗，把不同的bs也都加到一起了
            print("p:", p.shape)
            loss = loss_1 + loss_2 + loss_3 + lambda_4 * p.item()
            print("loss:", loss.shape)
            print("loss:", loss.item())
            print("loss:", loss)
            print("loss type:", type(loss))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    loss_item = loss.item()
    print("loss:", loss_item)
    average_loss = loss_item / (len(train_loader) * len(hiding_data_loader))
    print("average loss:", average_loss)
    return average_loss  # 这是每一个batch_size的载体图像下，每一个batch_size的隐藏文本的平均loss


# def loss_function(x, trg, padding_idx):
#     loss = criterion(x, trg)  # x与预期的交叉熵
#     mask = (trg != padding_idx).type_as(loss.data)  # mask去掉padding的部分
#     loss *= mask  # 将padding的部分的loss置为0，因为我们通常会使用填充标记来对齐不同长度的序列，但是这些填充部分不应该对损失产生影响
#
#     return loss.mean()  # 返回loss的平均值

def freeze_net(net, is_requires_grad):
    for param in net.parameters():
        param.requires_grad = is_requires_grad
    if is_requires_grad:
        net.train()
    else:
        net.eval()

def train_all(epoch, model, batch_size, data, hiding_data, optimizer_all, lambda_1, lambda_2, lambda_3, lambda_4):
    # freeze_net(model.en, True)  # 只更新en
    # freeze_net(model.invariance, True)
    # freeze_net(model.extractor, True)
    # freeze_net(model.decoder, True)
    model.train()

    # 基本输出
    encoder_output = model.encoder(hiding_data)
    embedder_output = model.embedder(encoder_output, data, batch_size)  # 最后这个是batch_size
    invariance_output, weights = model.invariance(embedder_output)  # 这里的N是冗余，所以之后应该加上一个正则化项，提高这里的鲁棒性
    extractor_output = model.extractor(invariance_output)
    decoder_output = model.decoder(extractor_output, batch_size)  # 最后这个是batch_size

    # 损失函数(注意得到的事每个单个数据的损失，所以计算的时候注意取平均或者除以bs)
    loss_1 = lambda_1 * l1_norm(decoder_output, hiding_data)
    loss_2 = lambda_2 * l1_norm(embedder_output, data)
    b1w, b2w, _ = model.embedder.convBlock_star(encoder_output.view(batch_size, 3, 128, 128))
    b1m, b2m, _ = model.embedder.convBlock_star(embedder_output)
    g1w = gram_for_batch(b1w)
    g1m = gram_for_batch(b1m)
    g2w = gram_for_batch(b2w)
    g2m = gram_for_batch(b2m)
    loss_3 = lambda_3 * 0.5 * (l1_norm(g1w, g1m) + l1_norm(g2w, g2m))

    # 对weights中的所有元素平方求和
    loss_4 = lambda_4 * torch.sum(weights ** 2)
    loss = loss_1 + loss_2 + loss_3 + loss_4

    optimizer_all.zero_grad()
    loss.backward()  # 保留计算图
    optimizer_all.step()

    loss_item = loss.item()
    # print("loss:", loss_item)  # 这就是每个单个数据的loss
    return loss_item, loss_1.item(), loss_2.item(), loss_3.item(), loss_4.item()


def train_en(epoch, model, batch_size, data, hiding_data, optimizer_en, lambda_1, lambda_2, lambda_3, lambda_4):
    freeze_net(model.en, True)  # 只更新en
    freeze_net(model.invariance, False)
    freeze_net(model.extractor, False)
    freeze_net(model.decoder, False)

    # 基本输出
    encoder_output = model.en.encoder(hiding_data)
    embedder_output = model.en.embedder(encoder_output, data, batch_size)  # 最后这个是batch_size
    # invariance_output, weights = model.invariance(embedder_output)  # 这里的N是冗余，所以之后应该加上一个正则化项，提高这里的鲁棒性
    # extractor_output = model.extractor(invariance_output)
    # decoder_output = model.decoder(extractor_output, batch_size)  # 最后这个是batch_size

    # 损失函数(注意得到的事每个单个数据的损失，所以计算的时候注意取平均或者除以bs)
    loss_2 = lambda_2 * l1_norm(embedder_output, data)
    b1w, b2w, _ = model.en.embedder.convBlock_star(encoder_output.view(batch_size, 3, 128, 128))
    b1m, b2m, _ = model.en.embedder.convBlock_star(embedder_output)
    g1w = gram_for_batch(b1w)
    g1m = gram_for_batch(b1m)
    g2w = gram_for_batch(b2w)
    g2m = gram_for_batch(b2m)
    loss_3 = lambda_3 * 0.5 * (l1_norm(g1w, g1m) + l1_norm(g2w, g2m))

    loss = loss_2 + loss_3

    optimizer_en.zero_grad()
    loss.backward()  # 保留计算图
    optimizer_en.step()

    return loss_2.item(), loss_3.item()

# 单个数据的训练
def validate(model, batch_size, data, hiding_data, lambda_1, lambda_2, lambda_3, lambda_4):
    # freeze_net(model.encoder, False)  # 全都不更新
    # freeze_net(model.embedder, False)
    # freeze_net(model.invariance, False)
    # freeze_net(model.extractor, False)
    # freeze_net(model.decoder, False)
    model.eval()

    with torch.no_grad():
        # 基本输出
        encoder_output = model.encoder(hiding_data)
        embedder_output = model.embedder(encoder_output, data, batch_size)  # 最后这个是batch_size
        invariance_output, weights = model.invariance(embedder_output)  # 这里的N是冗余，所以之后应该加上一个正则化项，提高这里的鲁棒性
        extractor_output = model.extractor(invariance_output)
        decoder_output = model.decoder(extractor_output, batch_size)  # 最后这个是batch_size

        # 损失函数(注意得到的事每个单个数据的损失，所以计算的时候注意取平均或者除以bs)
        loss_1 = lambda_1 * l1_norm(decoder_output, hiding_data)
        loss_2 = lambda_2 * l1_norm(embedder_output, data)
        b1w, b2w, _ = model.embedder.convBlock_star(encoder_output.view(batch_size, 3, 128, 128))
        b1m, b2m, _ = model.embedder.convBlock_star(embedder_output)
        g1w = gram_for_batch(b1w)
        g1m = gram_for_batch(b1m)
        g2w = gram_for_batch(b2w)
        g2m = gram_for_batch(b2m)
        loss_3 = lambda_3 * 0.5 * (l1_norm(g1w, g1m) + l1_norm(g2w, g2m))
        # 对weights中的所有元素平方求和
        loss_4 = lambda_4 * torch.sum(weights ** 2)
    return loss_1.item(), loss_2.item(), loss_3.item(), loss_4.item()


def l1_norm(x, y):  # 张量的第一范数差，最后取平均是因为求的是单个数据的范数差
    return torch.mean(torch.abs(x - y))
    # # mse
    # return torch.mean((x - y) ** 2)
def gram_for_batch(y):  # 输入[batch, c, h, w],结果是[batch, c, c]的Gram矩阵(其实是张量)
    """ Returns the gram matrix of y (used to compute style loss) """
    (b, c, h, w) = y.size()
    features = y.view(b, c, h * w)
    features_t = features.transpose(1, 2)  # C和w*h转置
    gram = features.bmm(features_t) / (c * h * w)  # 将features与features_t相乘,最后除以那个其实不确定为什么
    # print("gram:", gram.shape)
    return gram  # [bs, 96, 96],所以之后求第一范数差的时候正好需要除以一个bs，完美对应

if __name__ == '__main__':
    batch_size = 100
    learning_rate = 0.00001
    N = 5  # 全连接层的输出通道数
    now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    writer = SummaryWriter(log_dir="./logs/image_hiding/" + now)

    # dataset是本地的dog文件夹中的数据集
    data_dir = "E:\code\deepsc_hiding_torch\dataset\ILSVRC-2012\ILSVRC2012_img_val"
    data_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    train_dataset = datasets.ImageFolder(root=data_dir, transform=data_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    # random_split = torch.utils.data.random_split(train_dataset, [1000, 50000])  # 这是将数据集分成两部分，前面是验证集，后面是训练集
    # val_loader = DataLoader(random_split[0], batch_size=batch_size, shuffle=True, drop_last=True)
    # train_loader = DataLoader(random_split[1], batch_size=batch_size, shuffle=True, drop_last=True)

    # 隐藏信息
    # img_path = "D:\deeplearning\deepsc_hiding_torch\deepsc_hiding_torch\\binary_image.png"
    # img_PIL = Image.open(img_path)
    # img_tensor = transforms.ToTensor()(img_PIL)
    # hiding_data = img_tensor.unsqueeze(0).to(device)  # 增加一个batch_size的维度

    # 隐藏信息数据集
    # hiding_data_dir = "D:\deeplearning\deepsc_hiding_torch\deepsc_hiding_torch\dataset\MNIST_data\min"
    hiding_data_transform = transforms.Compose([
        transforms.Grayscale(),  # 转换为灰度图像
        transforms.ToTensor()  # 转换为张量
    ])
    hiding_data_dataset = CIFAR10(root='E:\code\deepsc_hiding_torch\dataset\CIFAR10', train=True, download=True,
                                  transform=hiding_data_transform)

    # hiding_data_dataset = datasets.ImageFolder(root=hiding_data_dir, transform=hiding_data_transform)
    hiding_data_loader = DataLoader(hiding_data_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_hiding_data_loader = DataLoader(hiding_data_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    epoch = 20000
    model = net(N).to(device)
    checkpoint_path = 'E:\code\deepsc_hiding_torch\checkpoints\image_hiding\\2024-06-03-10_17_33\checkpoint_473_0.pth'
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['net'])
    model.to(device)

    # fc_weights = model.invariance.state_dict()['fc.weight']
    # print("---------fc_weights---------")
    # print(fc_weights.shape)
    # print(fc_weights)
    # print("---------fc_weights---------done")
    optimizer_all = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer_en = torch.optim.Adam(model.en.parameters(), lr=learning_rate)

    lambda_1 = 1
    lambda_2 = 1
    lambda_3 = 1
    lambda_4 = 0.01

    loss_min = 879385

    # now = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    if not os.path.isdir("./checkpoints/image_hiding/" + now):
        os.mkdir("./checkpoints/image_hiding/" + now)

    for t in range(epoch):
        l = 0
        l1 = 0
        l2 = 0
        l3 = 0
        l4 = 0

        v = 0
        v1 = 0
        v2 = 0
        v3 = 0
        v4 = 0
        # print("Epoch: ", t)
        # 这是每一个epoch的loss，已经除以了多少个batch
        length = 1
        data_iter = iter(train_loader)
        hiding_data_iter = iter(hiding_data_loader)
        val_data_iter = iter(val_loader)
        val_hiding_data_iter = iter(val_hiding_data_loader)
        # 下面进入这个epoch的训练
        for batch in range(length):
            # print("batch:", batch)
            # 各取一个bs的数据进行组合
            data, _ = next(data_iter)
            hiding_data, _ = next(hiding_data_iter)
            data = data.to(device)
            hiding_data = hiding_data.to(device)
            loss, loss_1, loss_2, loss_3, loss_4 = train_all(t, model, batch_size, data, hiding_data, optimizer_all=optimizer_all, lambda_1=lambda_1, lambda_2=lambda_2, lambda_3=lambda_3, lambda_4=lambda_4)
            # loss_2, loss_3 = train_en(t, model, batch_size, data, hiding_data, optimizer_en=optimizer_en, lambda_1=lambda_1, lambda_2=lambda_2, lambda_3=lambda_3, lambda_4=lambda_4)
            l += loss
            l1 += loss_1
            l2 += loss_2
            l3 += loss_3
            l4 += loss_4
            # if batch % 500 == 0:
            #     print("epoch:", t)
            #     print("batch:", batch)
            #     print("loss:", loss + loss_2 + loss_3)
            #     checkpoint = {
            #         "net": model.state_dict(),
            #         'optimizer_all': optimizer_all.state_dict(),
            #         "optimizer_en": optimizer_en.state_dict(),
            #     }
            #     torch.save(checkpoint, './checkpoints/image_hiding/' + now + '/checkpoint_{}_{}.pth'.format(t, batch))
            #     print("model saved")
        total = (l1 + l2 + l3 + l4) / length
        print("train loss:", total)  # loss又经过了batch的平均

        writer.add_scalar("total_loss", total, t)
        writer.add_scalar("loss_1", l1 / length, t)
        writer.add_scalar("loss_2", l2 / length, t)
        writer.add_scalar("loss_3", l3 / length, t)
        writer.add_scalar("loss_4", l4 / length, t)

        # 下面是验证集
        v_total = 0
        for batch in range(length):
            data, _ = next(val_data_iter)
            hiding_data, _ = next(val_hiding_data_iter)
            data = data.to(device)
            hiding_data = hiding_data.to(device)
            loss_1, loss_2, loss_3, loss_4 = validate(model, batch_size, data, hiding_data, lambda_1, lambda_2, lambda_3, lambda_4)
            v_total += loss_1 + loss_2 + loss_3 + loss_4
            v1 += loss_1
            v2 += loss_2
            v3 += loss_3
            v4 += loss_4
            if batch % 500 == 0:
                print("epoch:", t)
                print("batch:", batch)
                checkpoint = {
                    "net": model.state_dict(),
                    'optimizer_all': optimizer_all.state_dict(),
                    # "optimizer_en": optimizer_en.state_dict(),
                }
                torch.save(checkpoint, './checkpoints/image_hiding/' + now + '/checkpoint_{}_{}.pth'.format(t, batch))
                print("model saved")
        v_total = v_total / length
        print("validate loss:", v_total)
        writer.add_scalar("val_total_loss", v_total, t)
        writer.add_scalar("val_loss_1", v1 / length, t)
        writer.add_scalar("val_loss_2", v2 / length, t)
        writer.add_scalar("val_loss_3", v3 / length, t)
        writer.add_scalar("val_loss_4", v4 / length, t)

        if v_total < loss_min:
            loss_min = v_total

            checkpoint = {
                "net": model.state_dict(),
                'optimizer_all': optimizer_all.state_dict(),
                # "optimizer_en": optimizer_en.state_dict(),
            }
            torch.save(checkpoint, './checkpoints/image_hiding/' + now + '/epoch_{}.pth'.format(t))
            print("epoch model saved")

    writer.close()
    print("total loss_min:", loss_min)