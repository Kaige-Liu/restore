# -*- coding: utf-8 -*-
import os
import argparse
import pickle
import time
import json
import torch
import random
import torch.nn as nn
import numpy as np
from torch.nn.init import xavier_uniform_
from torch.utils.tensorboard import SummaryWriter

from models.transceiver import ChannelDecoder, Decoder, Encoder
from utlis.tools import SNR_to_noise
from utlis.trainer import initNetParams, train_mi
from dataset.dataloader import return_iter
# from models.transceiver import DeepSC
from models.mutual_info import Mine
from tqdm import tqdm

from utlis.trainer_new import train_step, val_step

parser = argparse.ArgumentParser()  # 创建一个命令行参数解释器
parser.add_argument('--vocab-file', default='.\\data\\vocab.json', type=str)
parser.add_argument('--checkpoint-path', default='.\\checkpoints\\deepsc_hiding', type=str)
parser.add_argument('--channel', default='Rayleigh', type=str, help='Please choose AWGN, Rayleigh, and Rician')
parser.add_argument('--MAX-LENGTH', default=30, type=int)
parser.add_argument('--MIN-LENGTH', default=4, type=int)
parser.add_argument('--d-model', default=128, type=int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num-layers', default=4, type=int)
parser.add_argument('--num-heads', default=8, type=int)
parser.add_argument('--batch-size', default=10, type=int)  # 这里控制的是每次拿(从数据集中读取)多少张牌(个句子)
parser.add_argument('--epochs', default=20000, type=int)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class tmp_full(nn.Module):  # 将1维度的展成48维度的 便于计算B2
    def __init__(self):
        super(tmp_full, self).__init__()
        self.fc = nn.Linear(31 * 128 * 1, 31 * 128 * 48)

    def forward(self, x):
        # 输入(batch_size, 1, 31, 1281)
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # 展平操作
        x = self.fc(x)  # 全连接层
        x = x.view(batch_size, 48, 31, 128)  # 恢复维度
        return x

class CustomConvBlock(nn.Module):  # 完成更新
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

        # self.relu = nn.ReLU(inplace=True)

    def forward(self, x):  # 这里需要改一下(改完了)
        # print("x shape:", x.shape)
        x1 = self.conv_1_1132(x)

        x21 = self.conv_2_1132(x)
        x2 = self.conv_2_3332(x21)

        x31 =self.conv_3_1132(x)
        x32 = self.conv_3_3332(x31)
        x3 = self.conv_3_3332_2(x32)

        b1 = torch.cat((x1, x2, x3), 1)  # 这里是将三个张量拼接起来，dim=1，也就是在第2个维度C上拼接
        b2 = self.conv111(b1)

        out = x + b2
        return b1, b2, out

class Encoder_hiding(nn.Module):  # 这个是信息隐藏的encoder，负责将二进制图片编码成一个32x32x48的隐藏文本图像的特征
    def __init__(self):
        super(Encoder_hiding, self).__init__()

        self.convBlock1 = CustomConvBlock(31, 128, 1)  # 卷积块
        self.convBlock2 = CustomConvBlock(31, 128, 1)  # 卷积块
        self.conv1124 = nn.Conv2d(1, 24, 1, stride=1, padding=0)
        # self.bn1124 = nn.BatchNorm2d(24)

        self.convBlock3 = CustomConvBlock(31, 128, 24)
        self.convBlock4 = CustomConvBlock(31, 128, 24)
        self.conv1148 = nn.Conv2d(24, 48, 1, stride=1, padding=0)
        # self.bn1148 = nn.BatchNorm2d(48)

        # self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        _, _, x = self.convBlock1(x)
        _, _, x = self.convBlock2(x)
        x = self.conv1124(x)
        _, _, x = self.convBlock3(x)
        _, _, x = self.convBlock4(x)
        x = self.conv1148(x)
        return x


class Decoder_hiding(nn.Module):  # 将特征（但是目前还缺少一个reshape 直接就当做是reshape之后的尺寸了，之后得改一下
    def __init__(self):
        super(Decoder_hiding, self).__init__()
        self.convBlock1 = CustomConvBlock(31, 128, 1)  # 卷积块
        self.convBlock2 = CustomConvBlock(31, 128, 1)  # 卷积块
        self.conv1124 = nn.Conv2d(1, 1, 1, stride=1, padding=0)
        # self.bn1124 = nn.BatchNorm2d(1)

        self.convBlock3 = CustomConvBlock(31, 128, 1)
        self.convBlock4 = CustomConvBlock(31, 128, 1)
        self.conv111 = nn.Conv2d(1, 1, 1, stride=1, padding=0)
        # self.bn111 = nn.BatchNorm2d(1)

        # self.relu = nn.ReLU(inplace=True)
        # self.sigmod = nn.Sigmoid()

    def forward(self, x, batch_size):
        _, _, x = self.convBlock1(x)
        _, _, x = self.convBlock2(x)
        # print("经过2个块之后的输出：", x.shape)
        x = self.conv1124(x)
        _, _, x = self.convBlock3(x)
        _, _, x = self.convBlock4(x)
        x = self.conv111(x)  # 最后输出不用relu 用sigmod
        return x

class Embedder_hiding(nn.Module):  # 嵌入层，负责将隐藏文本图像的特征注入到载体图像中，得到128x128x3的图像，要求很像载体图像
    def __init__(self):
        super(Embedder_hiding, self).__init__()
        self.convBlock_star = CustomConvBlock(31, 128, 48)  # 卷积块
        self.convBlock2 = CustomConvBlock(31, 128, 49)
        self.conv113 = nn.Conv2d(49, 1, 1, stride=1, padding=0)
        # self.bn113 = nn.BatchNorm2d(1)
        # self.sigmod = nn.Sigmoid()


    def forward(self, x, c):  # 接受的输入x是48x31x128的
        b1, b2, x = self.convBlock_star(x)
        # 下面将这个和另一个1x31x128的载体特征c拼接起来
        x = torch.cat((x, c), 1)
        _, _, x = self.convBlock2(x)
        x = self.conv113(x)  # 同样是输出，不用relu
        return x

class CustomLinearLayer(nn.Module):
    def __init__(self, num_channels, num_outputs):
        super(CustomLinearLayer, self).__init__()
        self.num_channels = num_channels
        self.num_outputs = num_outputs
        self.weights = nn.Parameter(torch.randn(num_outputs, num_channels))  # 定义权重矩阵，初始值为随机数
        xavier_uniform_(self.weights)
        # self.bn = nn.BatchNorm2d(num_outputs)
        # self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        x = x.view(batch_size, self.num_channels, -1)  # 将每个通道视为一个整体
        output = torch.matmul(self.weights, x)  # 矩阵乘法，得到输出
        output = output.view(batch_size, self.num_outputs, height, width)  # 调整输出形状
        # output = self.relu(self.bn(output))
        return output, self.weights


class Extractor(nn.Module):
    def __init__(self, N):
        super(Extractor, self).__init__()
        self.convBlock1 = CustomConvBlock(31, 128, N)  # 卷积块
        self.convBlock2 = CustomConvBlock(31, 128, N)  # 卷积块
        self.conv113 = nn.Conv2d(N, 1, 1, stride=1, padding=0)
        # self.bn113 = nn.BatchNorm2d(1)
        # self.sigmod = nn.Sigmoid()


    def forward(self, x):  # x是128x128xN的
        _, _, x = self.convBlock1(x)
        _, _, x = self.convBlock2(x)
        x = self.conv113(x)
        return x

class mac(nn.Module):
    def __init__(self, args):
        self.embedding = nn.Embedding(num_embeddings=args.vocab_size, embedding_dim=args.encoder_d_model)
    def forward(self, key):
        key_ebd = self.embedding(key)
        return key_ebd







class Hiding(nn.Module):
    def __init__(self, N):
        super(Hiding, self).__init__()
        self.encoder_hiding = Encoder_hiding()
        self.embedder_hiding = Embedder_hiding()
        self.invariance = CustomLinearLayer(1, N)
        self.extractor = Extractor(N)
        self.decoder_hiding = Decoder_hiding()
        self.tmp = tmp_full()


class H_DeepSC(nn.Module):
    def __init__(self, N, num_layers, src_vocab_size, trg_vocab_size, src_max_len,
                 trg_max_len, d_model, num_heads, dff, dropout=0.1):
        super(H_DeepSC, self).__init__()

        # self.deepsc = DeepSC(num_layers, src_vocab_size, trg_vocab_size, src_max_len,
        #                      trg_max_len, d_model, num_heads, dff, dropout)
        # self.hiding = Hiding(N)
        self.encoder = Encoder(num_layers, src_vocab_size, src_max_len,
                               d_model, num_heads, dff, dropout)  # 这个就是语义编码器

        self.channel_encoder = nn.Sequential(nn.Linear(d_model, 256),
                                             # nn.ELU(inplace=True),
                                             nn.ReLU(inplace=True),
                                             nn.Linear(256, 16))

        self.channel_decoder = ChannelDecoder(16, d_model, 512)

        self.decoder = Decoder(num_layers, trg_vocab_size, trg_max_len,
                               d_model, num_heads, dff, dropout)

        self.dense = nn.Linear(d_model, trg_vocab_size)  # 输出层
        self.encoder_hiding = Encoder_hiding()
        self.embedder_hiding = Embedder_hiding()
        self.invariance = CustomLinearLayer(1, N)
        self.extractor = Extractor(N)
        self.decoder_hiding = Decoder_hiding()
        self.tmp = tmp_full()

def setup_seed(seed):  # 设置随机种子，根本没用
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 为当前GPU设置随机种子
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def train(epoch, args, net, channel, n_var, lambda_1, lambda_2, lambda_3, lambda_4, mi_net=None):  # 当前训练的轮数，命令行参数，模型，互信息网络（默认是None，也就是不训互信息网络）
    hiding_train_iterator = return_iter(args, 'train')  # 从训练数据集中抓牌，得到的是一个dataloader类型的对象（其实就是dataloder 用法完全一样
    cover_train_iterator = return_iter(args, 'train')
    pbar = tqdm(hiding_train_iterator)  # 进度条

    total = 0
    total_1 = 0
    total_2 = 0
    noise_std = np.random.uniform(SNR_to_noise(5), SNR_to_noise(10), size=(1))  # 生成介于信噪比为5和10之间的随机的噪声标准差，
    # print("---------------------------noise_std---------------------------")
    # print("noise_std: ", noise_std)  # noise_std:  [0.37055991]
    # print("noise_std[0]:", noise_std[0])  # noise_std[0]: 0.3705599051835091
    # print("---------------------------noise_std---------------------------")
    cover_data_iter = iter(cover_train_iterator)  # 这里可能会改一下，目前是cover和hiding文本用的一个数据集

    for sents in pbar:  # 每个batch的数据
        sents = sents.to(device)
        cover_data = next(cover_data_iter)
        cover_data = cover_data.to(device)

        loss, loss_1, loss_2 = train_step(net, sents, sents, cover_data, cover_data, noise_std[0], pad_idx,
                          optimizer, args.channel, lambda_1, lambda_2, lambda_3, lambda_4)
        total += loss
        total_1 += loss_1
        total_2 += loss_2
        pbar.set_description(
            'Epoch: {};  Type: Train; Loss: {:.5f}'.format(
                epoch, loss
            )
        )

    pbar.set_description(
        'Epoch: {};  Type: Train; Loss: {:.5f}'.format(
            epoch, total / len(hiding_train_iterator)
        )
    )
    # 返回这个epoch的loss
    return total / len(hiding_train_iterator), total_1 / len(hiding_train_iterator), total_2 / len(hiding_train_iterator)

def validate(epoch, args, net, lambda_1, lambda_2, lambda_3, lambda_4):  # epoch表示正在验证的是第几轮
    hiding_test_iterator = return_iter(args, 'test')  # 从测试数据集中抓牌
    cover_test_iterator = return_iter(args, 'test')

    net.eval()  # 将模型设置为验证模式

    pbar = tqdm(hiding_test_iterator)
    cover_data_iter = iter(cover_test_iterator)  # 这里可能会改一下，目前是cover和hiding文本用的一个数据集
    total = 0  # loss的总和
    total_cover_part = 0
    total_hiding_part = 0

    with torch.no_grad():  # 不需要计算梯度，看牌前的常规操作，不用管
        for sents in pbar:  # 其实就是for data in dataloader,这是[128, 31]的张量
            sents = sents.to(device)  # 将数据放到GPU上
            cover_data = next(cover_data_iter)
            cover_data = cover_data.to(device)

            loss, loss_cover_part, loss_hiding_part = val_step(net, sents, sents, cover_data, cover_data, 0.1, pad_idx, args.channel, lambda_1, lambda_2, lambda_3, lambda_4)

            total += loss
            total_cover_part += loss_cover_part
            total_hiding_part += loss_hiding_part

            pbar.set_description(  # 设置进度条的描述
                'Epoch: {};  Type: VAL; Loss: {:.5f}'.format(
                    epoch, loss
                )
            )
        pbar.set_description(
            'Epoch: {};  Type: VAL; Loss: {:.5f}'.format(
                epoch, total / len(hiding_test_iterator)
            )
        )

    # 计算平均的loss 后面的len其实就是batch的个数128，也就是拿了多少次牌（而不是每次拿了多少张牌），即一个batch的平均loss
    # 但是后面两个是每个元素的loss，即每个单词的平均loss（然后再经过batch平均了）
    return total / len(hiding_test_iterator), total_cover_part / len(hiding_test_iterator), total_hiding_part / len(hiding_test_iterator)

if __name__ == '__main__':
    now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    # if not os.path.isdir("./checkpoints/deepsc_hiding/" + now):
    #     os.mkdir("./checkpoints/deepsc_hiding/" + now)
    if not os.path.isdir("F:\checkpoints\deepsc_hiding\\" + now):
        os.mkdir("F:\checkpoints\deepsc_hiding\\" + now)
    writer = SummaryWriter(log_dir="./logs/deepsc_hiding/" + now)

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

    N = 5  # 全连接层的输出通道数
    lambda_1 = 1
    lambda_2 = 1
    lambda_3 = 1
    lambda_4 = 0.01

    epoch_losses = []
    cover_part = []
    hiding_part = []
    hiding_net = []
    """ define optimizer and loss function """
    # 默认层数是4层，d_model是128，dff是512，num_heads是8
    # 源语言和目标语言的词汇量都是num_vocab，也就是字典里面的单词的个数
    # 源语言序列的最大长度也是num_vocab
    # d_model是隐藏层的维度，num_heads是多头注意力的头数，dff是前馈神经网络Feedforward层的维度
    # dropout是丢弃概率，为了防止过拟合
    # deepsc是一个transfomer模型，用来进行编码和解码
    H_deepsc = H_DeepSC(N, args.num_layers, num_vocab, num_vocab,
                    num_vocab, num_vocab, args.d_model, args.num_heads,
                    args.dff, 0.1).to(device)
    checkpoint_path = 'F:\checkpoints\deepsc_hiding\\2024-06-06-21_16_04\checkpoint_498.pth'
    checkpoint = torch.load(checkpoint_path)
    H_deepsc.load_state_dict(checkpoint['net'])
    H_deepsc.to(device)

    noise_std = np.random.uniform(SNR_to_noise(5), SNR_to_noise(10), size=(1))
    optimizer = torch.optim.Adam(H_deepsc.parameters(),
                                 lr=1e-4, betas=(0.9, 0.98), eps=1e-8, weight_decay=5e-4)

    # opt = NoamOpt(args.d_model, 1, 4000, optimizer)

    # initNetParams(H_deepsc)  # 初始化deepsc模型的参数

    # 下面就是训练deepsc模型
    for epoch in range(args.epochs):  # 默认训练完整的数据集80轮
        start = time.time()  # 记录每轮开始时间（没用到）
        record_loss = 100000  # 其实是loss，设置的大一点

        total, total_1, total_2 = train(epoch, args, H_deepsc, args.channel, noise_std[0], lambda_1, lambda_2, lambda_3, lambda_4)  # 训练deepsc模型
        avg_loss, avg_cover_part, avg_hiding_part = validate(epoch, args, H_deepsc, lambda_1, lambda_2, lambda_3,
                                                      lambda_4)  # 验证deepsc模型，得到平均的loss

        if avg_loss < record_loss:  # 如果验证的loss小于之前的loss（性能更好了）
            checkpoint = {
                "net": H_deepsc.state_dict(),
            }
            torch.save(checkpoint, 'F:\checkpoints\deepsc_hiding\\' + now + '/checkpoint_{}.pth'.format(epoch))
            print("model saved")
            record_loss = avg_loss  # 更新最小的准确率

        writer.add_scalar('Loss/train', total, epoch)
        writer.add_scalar('Loss/val', avg_loss, epoch)
        writer.add_scalar('Loss/train_cover', total_1, epoch)
        writer.add_scalar('Loss/val_cover', avg_cover_part, epoch)
        writer.add_scalar('Loss/train_hiding', total_2, epoch)
        writer.add_scalar('Loss/val_hiding', avg_hiding_part, epoch)



