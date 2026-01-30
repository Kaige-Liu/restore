# -*- coding: utf-8 -*-
import os
import argparse
import time
import json
import torch
import random
import torch.nn as nn
import numpy as np
from utlis.tools import SNR_to_noise
from utlis.trainer import initNetParams, val_step, train_mi
from dataset.dataloader import return_iter
from models.transceiver import DeepSC, H_DeepSC
from models.mutual_info import Mine
from tqdm import tqdm

from utlis.trainer_new import train_step

parser = argparse.ArgumentParser()  # 创建一个命令行参数解释器
parser.add_argument('--vocab-file', default='.\\data\\vocab.json', type=str)
parser.add_argument('--checkpoint-path', default='.\\checkpoints\\deepsc-Rayleigh', type=str)
parser.add_argument('--channel', default='Rayleigh', type=str, help='Please choose AWGN, Rayleigh, and Rician')
parser.add_argument('--MAX-LENGTH', default=30, type=int)
parser.add_argument('--MIN-LENGTH', default=4, type=int)
parser.add_argument('--d-model', default=128, type=int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num-layers', default=4, type=int)
parser.add_argument('--num-heads', default=8, type=int)
parser.add_argument('--batch-size', default=128, type=int)  # 这里控制的是每次拿(从数据集中读取)多少张牌(个句子)
parser.add_argument('--epochs', default=80, type=int)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device: ", device)  # device:  cuda:0


def setup_seed(seed):  # 设置随机种子，根本没用
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 为当前GPU设置随机种子
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def validate(epoch, args, net):  # epoch表示正在验证的是第几轮
    test_iterator = return_iter(args, 'test')  # 从测试数据集中抓牌

    net.eval()  # 将模型设置为验证模式

    pbar = tqdm(test_iterator)

    total = 0  # loss的总和
    with torch.no_grad():  # 不需要计算梯度，看牌前的常规操作，不用管
        for sents in pbar:  # 其实就是for data in dataloader,这是[128, 31]的张量
            sents = sents.to(device)  # 将数据放到GPU上

            loss = val_step(net, sents, sents, 0.1, pad_idx, args.channel)

            total += loss
            pbar.set_description(  # 设置进度条的描述
                'Epoch: {};  Type: VAL; Loss: {:.5f}'.format(
                    epoch + 1, loss
                )
            )
        pbar.set_description(
            'Epoch: {};  Type: VAL; Loss: {:.5f}'.format(
                epoch, total / len(test_iterator)
            )
        )

    # 计算平均的loss
    return total / len(test_iterator)  # 后面的len其实就是batch的个数128，也就是拿了多少次牌（而不是每次拿了多少张牌）


def train(hiding, epoch, args, net, channel, n_var, mi_net=None):  # 当前训练的轮数，命令行参数，模型，互信息网络（默认是None，也就是不训互信息网络）
    train_iterator = return_iter(args, 'train')  # 从训练数据集中抓牌，得到的是一个dataloader类型的对象（其实就是dataloder 用法完全一样
    print("len", len(train_iterator))
    print("---------------------------train_iterator---------------------------")

    pbar = tqdm(train_iterator)  # 进度条

    total = 0
    noise_std = np.random.uniform(SNR_to_noise(5), SNR_to_noise(10), size=(1))  # 生成介于信噪比为5和10之间的随机的噪声标准差，
    # print("---------------------------noise_std---------------------------")
    # print("noise_std: ", noise_std)  # noise_std:  [0.37055991]
    # print("noise_std[0]:", noise_std[0])  # noise_std[0]: 0.3705599051835091
    # print("---------------------------noise_std---------------------------")

    for sents in pbar:  # 每个batch的数据
        sents = sents.to(device)

        loss = train_step(hiding, net, sents, sents, noise_std[0], pad_idx,
                          optimizer, args.channel)
        total += loss
        pbar.set_description(
            'Epoch: {};  Type: Train; Loss: {:.5f}'.format(
                epoch + 1, loss
            )
        )

    pbar.set_description(
        'Epoch: {};  Type: Train; Loss: {:.5f}'.format(
            epoch, total / len(train_iterator)
        )
    )


from models.text_hiding import Embedder_hiding, Extractor, CustomLinearLayer, net, Encoder_hiding

if __name__ == '__main__':
    # setup_seed(10)
    args = parser.parse_args()  # 将命令行参数存储到args中，用args.参数名来调用访问对应的参数
    # print(type(args))
    # print(args)
    # print("---------------------args---------------------")
    args.vocab_file = args.vocab_file

    """ preparing the dataset """
    vocab = json.load(open(args.vocab_file, 'rb'))
    token_to_idx = vocab['token_to_idx']
    num_vocab = len(token_to_idx)

    # 下面是三个比较特殊的单词，这里是将这三个单词的索引提取出来，也就是对应的数字
    pad_idx = token_to_idx["<PAD>"]  # 0
    start_idx = token_to_idx["<START>"]  # 1
    end_idx = token_to_idx["<END>"]  # 2

    N = 12  # 全连接层的输出通道数

    """ define optimizer and loss function """
    # 默认层数是4层，d_model是128，dff是512，num_heads是8
    # 源语言和目标语言的词汇量都是num_vocab，也就是字典里面的单词的个数
    # 源语言序列的最大长度也是num_vocab
    # d_model是隐藏层的维度，num_heads是多头注意力的头数，dff是前馈神经网络Feedforward层的维度
    # dropout是丢弃概率，为了防止过拟合
    # deepsc是一个transfomer模型，用来进行编码和解码
    H_deepsc = H_DeepSC(args.num_layers, num_vocab, num_vocab,
                        num_vocab, num_vocab, args.d_model, args.num_heads,
                        args.dff, 0.1).to(device)

    hiding = net(N).to(device)

    noise_std = np.random.uniform(SNR_to_noise(5), SNR_to_noise(10), size=(1))
    optimizer = torch.optim.Adam(H_deepsc.parameters(),
                                 lr=1e-4, betas=(0.9, 0.98), eps=1e-8, weight_decay=5e-4)
    # opt = NoamOpt(args.d_model, 1, 4000, optimizer)

    initNetParams(H_deepsc)  # 初始化deepsc模型的参数

    # 下面就是训练deepsc模型
    for epoch in range(args.epochs):  # 默认训练完整的数据集80轮
        start = time.time()  # 记录每轮开始时间（没用到）
        record_loss = 10  # 其实是loss，设置的大一点

        train(hiding, epoch, args, H_deepsc, args.channel, noise_std[0])  # 训练deepsc模型

        avg_loss = validate(epoch, args, H_deepsc)  # 验证deepsc模型，得到平均的loss

        if avg_loss < record_loss:  # 如果验证的loss小于之前的loss（性能更好了）
            if not os.path.exists(args.checkpoint_path):
                os.makedirs(args.checkpoint_path)  # 创建文件
            with open(args.checkpoint_path + '/checkpoint_{}.pth'.format(str(epoch + 1).zfill(2)), 'wb') as f:
                torch.save(H_deepsc.state_dict(), f)  # 保存模型到上面的文件中
            record_loss = avg_loss  # 更新最小的准确率
    record_loss = []
