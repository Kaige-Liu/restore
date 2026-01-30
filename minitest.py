# import torch
# import torch.nn as nn
# import torch.optim as optim
#
#
# # 定义网络结构
# class MAC_verify(nn.Module):
#     def __init__(self):
#         super(MAC_verify, self).__init__()
#         self.fc1 = nn.Linear(1 * 128, 128)  # 处理mac
#         self.fc2 = nn.Linear(31 * 128, 128)  # 处理f
#         self.fc3 = nn.Linear(10 * 128, 128)  # 处理key_bed
#         self.fc4 = nn.Linear(384, 128)  # 第三层全连接层
#         self.fc5 = nn.Linear(128, 1)  # 输出层
#
#     def forward(self, x1, x2, x3):  # mac,f,key_ebd 输出为[bs, 1] 即概率
#         x1 = x1.view(-1, 1 * 128)  # 展平
#         x2 = x2.view(-1, 31 * 128)  # 展平
#         x3 = x3.view(-1, 10 * 128)  # 展平
#
#         x1 = torch.relu(self.fc1(x1))  # 第一层激活函数
#         x2 = torch.relu(self.fc2(x2))  # 第二层激活函数
#         x3 = torch.relu(self.fc3(x3))  # 第二层激活函数
#
#         x = torch.cat((x1, x2, x3), 1)  # 将三个输入合并
#
#         x = torch.relu(self.fc4(x))  # 第三层激活函数
#         x = torch.sigmoid(self.fc5(x))  # 输出层激活函数，输出概率
#
#         return x
#
#
# # 初始化网络
# model = MAC_verify()
#
# # 定义损失函数和优化器
# criterion = nn.BCELoss()  # 二元交叉熵损失函数
# optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器
#
# # 模拟一些数据
# batch_size = 8
# x1 = torch.randn(batch_size, 1, 128)
# x2 = torch.randn(batch_size, 31, 128)
# x3 = torch.randn(batch_size, 10, 128)
# y = torch.ones(batch_size, 1).float()  # 全是1
#
# # 训练模型
# num_epochs = 5
# for epoch in range(num_epochs):
#     optimizer.zero_grad()  # 清空梯度
#     output = model(x1, x2, x3)  # 前向传播
#     loss = criterion(output, y)  # 计算损失
#     loss.backward()  # 反向传播
#     optimizer.step()  # 更新权重
#
#     print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
#
# # 测试模型
# with torch.no_grad():  # 测试时不计算梯度
#     predictions = model(x1, x2, x3)
#     predicted_classes = predictions.round()  # 将概率四舍五入到最近的整数
#     print(f'Predictions: {predictions}')
#     print(f'Predicted Classes: {predicted_classes}')
import argparse
import json
import pickle
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.dataloader import collate_data, return_iter, return_iter_eve, return_iter_10
from models.transceiver import DeepSC
from utlis.tools import SeqtoText, BleuScore


# 椒盐噪声（但是只能在图片上用）
def salt_pepper_noise(image, ratio):
    output = np.zeros(image.shape, np.uint8)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):

            rand = random.random()
            if rand < ratio:  # salt pepper noise
                if random.random() > 0.5:  # change the pixel to 255
                    output[i][j] = 255
                else:
                    output[i][j] = 0
            else:
                output[i][j] = image[i][j]

    return output

# 裁剪攻击
def crop_attack(image, ratio):
    output = np.zeros(image.shape, np.uint8)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):

            rand = random.random()
            if rand < ratio:  # crop
                output[i][j] = 0
            else:
                output[i][j] = image[i][j]

    return output


def SNR_to_noise(snr):  # 计算信噪比为snr时的 噪声标准差
    snr = 10 ** (snr / 10)
    noise_std = 1 / np.sqrt(2 * snr)

    return noise_std

noise_std = np.random.uniform(SNR_to_noise(5), SNR_to_noise(10), size=(1))  # 生成介于信噪比为5和10之间的随机的噪声标准差

# 生成[88, 31, 128]的张量
# bs = 88
# import torch
# import torch.nn as nn
# import torch.optim as optim
#
#
#
# # 定义网络结构
# class NoiseNet(nn.Module):
#     def __init__(self):
#         super(NoiseNet, self).__init__()
#         self.conv1 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
#         self.relu = nn.ReLU()
#         self.conv2 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.relu(x)
#         x = self.conv2(x)
#         return x
#
# # 实例化网络
# model = NoiseNet()
#
# # 定义损失函数
# criterion = nn.MSELoss()
#
# # 定义优化器
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#
# # 模拟一些数据
# batch_size = 128
# input_data = torch.randn(batch_size, 32, 16)
#
# # 训练网络
# num_epochs = 100
# for epoch in range(num_epochs):
#     optimizer.zero_grad()
#     output = model(input_data)
#
#     loss = criterion(output, input_data)
#     loss.backward()
#     optimizer.step()
#
#     if (epoch+1) % 10 == 0:
#         print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
#         print(input_data[0])
#         print(output[0])
#         print()
#
# # 测试网络
# test_output = model(input_data)
# test_loss = criterion(test_output, input_data)
# print(f'Test Loss: {test_loss.item():.4f}')


# loss0 = torch.tensor(0.)
# print(type(loss0.item()))

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# class NoiseNet(nn.Module):
#     def __init__(self):
#         super(NoiseNet, self).__init__()
#         self.conv1 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
#         self.relu = nn.ReLU()
#         self.conv2 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
#
#     def forward(self, x):
#         # 记录原始输入的序列长度
#         original_length = x.size(1)
#         # 如果x的第一维度不是32 就填充成32
#         if x.size(1) != 32:
#             x = F.pad(x, (0, 0, 0, 32 - x.size(1)))
#
#         # 应用卷积和激活函数
#         x = self.conv1(x)
#         x = self.relu(x)
#         x = self.conv2(x)
#
#         # 裁剪输出以匹配原始输入的序列长度
#         # 由于卷积层的padding=1，所以输出长度会比输入多2（每边多1）
#         # 我们需要裁剪掉两边的1个元素来保持长度一致
#         x = x[:, :original_length, :]
#
#         return x
#
# # 测试代码
# if __name__ == "__main__":
#     # 创建两个不同长度的输入
#     input1 = torch.randn(5, 32, 16)  # [bs, 32, 16]
#     input2 = torch.randn(5, 31, 16)  # [bs, 31, 16]
#
#     # 初始化网络
#     model = NoiseNet()
#
#     # 测试网络
#     output1 = model(input1)
#     output2 = model(input2)
#
#     # 打印输出形状
#     print("Output 1 shape:", output1.shape)  # 应为 [5, 32, 16]
#     print("Output 2 shape:", output2.shape)  # 应为 [5, 31, 16]

# bs = 4
# enc_output = torch.randn(bs, 3, 2)
# print(enc_output)
# # enc_output进行加性噪声
# noise = enc_output + torch.randn(bs, 3, 2) * noise_std
# for i in range(bs):
#     idx_1 = random.randint(0, 2)
#     idx_2 = random.randint(0, 1)
#     enc_output[i][idx_1][idx_2] = noise[i][idx_1][idx_2]
#
# print(enc_output)


# def return_iter_10(args, split='train'):  # 对数据集进行抓牌
#     data_eur = EurDataset_10(split)  # 得到train_data.pkl文件的内容（训练的数据集）
#     # 下面对数据集进行抓牌，每次抓取batch_size个数据（句子）
#     data_iter = DataLoader(data_eur, batch_size=args.batch_size, num_workers=0,
#                            pin_memory=True, collate_fn=collate_data, shuffle=True, drop_last=True)  #
#     return data_iter  # 返回一个迭代器dataloader,就是分好batch的数据集
#
# class EurDataset_10(Dataset):  # 数据集类，就是对pkl数据进行读取
#     def __init__(self, split='train'):
#         data_dir = '.\\data\\'
#         with open(data_dir + '{}_data_10.pkl'.format(split), 'rb') as f:  # 打开train_data.pkl文件
#             self.data = pickle.load(f)  # 读取文件内容（类似于一本书，很多个句子，每个句子又有很多单词，单词是数字）
#
#
#     def __getitem__(self, index):
#         sents = self.data[index]  # 取出第index个句子,list
#         return sents
#
#     def __len__(self):
#         return len(self.data)  # 返回这本书中句子的个数
#
# train_iterator = return_iter(args, 'train')

# # 加载数据集
# import pickle
# import numpy as np
# import torch
# from torch.utils.data import Dataset
# from torch.utils.data import DataLoader
# from dataset.dataloader import collate_data
# # 加载train_data.pkl文件
# data_dir = '.\\data\\'
# with open(data_dir + '{}_data.pkl'.format('train'), 'rb') as f:
#     data = pickle.load(f)
# # 输出数据集中数据的个数
# print(len(data))
# # 将data打乱 并保存前99%的数据以及剩下的数据
# np.random.shuffle(data)
# alice = data[:int(len(data) * 0.90)]
# print(len(alice))
# eve = data[int(len(data) * 0.90):]
# print(len(eve))
# # 保存在train_data_99.pkl文件中
# with open(data_dir + '{}_data_alice.pkl'.format('train'), 'wb') as f:
#     pickle.dump(alice, f)
# # 保存在test_data_1.pkl文件中
# with open(data_dir + '{}_data_eve.pkl'.format('train'), 'wb') as f:
#     pickle.dump(eve, f)



# # 加载train_data_little.pkl文件
# data_dir = '.\\data\\'
# with open(data_dir + '{}_data_10.pkl'.format('test'), 'rb') as f:
#     data = pickle.load(f)
# print(len(data))
#
# # 缩小数据集
# data = data[:len(data) // 15]
# # 保存在train_data_15.pkl文件中
# with open(data_dir + '{}_data_eve.pkl'.format('train'), 'wb') as f:
#     pickle.dump(data, f)
#
# data_dir = '.\\data\\'
# with open(data_dir + '{}_data_eve.pkl'.format('train'), 'rb') as f:
#     data = pickle.load(f)
# # 输出数据集中数据的个数
# print(len(data))




#
# indices = torch.randperm(bs)
# # 随机排列数据
# shuffled_tensor = enc_output[indices]
# # 如果enc_output和shuffled_tensor相同，则重新生成shuffled_tensor
# while torch.all(torch.eq(enc_output, shuffled_tensor)):
#     indices = torch.randperm(bs)
#     shuffled_tensor = enc_output[indices]
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# criterion_noise = nn.MSELoss().to(device)
# loss = -1 * criterion_noise(enc_output, shuffled_tensor)
# print(loss)
# print(enc_output)
# print(shuffled_tensor)


import torch
import torch.nn.functional as F
#
# # 计算余弦相似度
# def cosine_similarity(tensor1, tensor2):  # 形状为[batch_size, 1, 128] 即mac
#     sum = F.cosine_similarity(tensor1, tensor2, dim=2)
#     average_cosine_similarity = torch.mean(sum)
#     return average_cosine_similarity.item()
#
#
# # 假设tensor1和tensor2是两个形状为[batch_size, 1, 128]的张量
# tensor1 = torch.rand(3, 1, 128)  # 用随机数初始化张量，实际使用时替换为你的张量
# tensor2 = torch.ones(3, 1, 128)
#
# # x = cosine_similarity(tensor1, tensor2)
# # print(x)
#
# print(tensor1)
# print(torch.round(tensor1))

import torch


def r2_score(y_true, y_pred):
    # 计算每个样本的均值，保持最后一个维度不变
    y_mean = torch.mean(y_true, dim=-1, keepdim=True)

    # 计算残差平方和 (RSS)，按最后一个维度求和
    rss = torch.sum((y_true - y_pred) ** 2, dim=-1)

    # 计算总平方和 (TSS)，按最后一个维度求和
    tss = torch.sum((y_true - y_mean) ** 2, dim=-1)

    # 计算 R2
    r2 = 1 - rss / tss

    # 对整个 batch 的 R2 取平均值
    r2_mean = torch.mean(r2)

    return r2_mean


# # 示例数据，形状为 [bs, 1, 128]
# bs = 4  # 假设批次大小为4
# y_true = torch.randn(bs, 1, 128)  # 随机生成一些数据作为真实值
# y_pred = torch.randn(bs, 1, 128)  # 随机生成一些数据作为预测值
#
# # 计算R2
# r2 = r2_score(y_true, y_pred)
# print(f'R^2: {r2.item()}')


parser = argparse.ArgumentParser()  # 创建一个命令行参数解释器
parser.add_argument('--vocab-file', default='.\\data\\vocab.json', type=str)
parser.add_argument('--vocab_path', default='.\\data\\vocab.json', type=str)
parser.add_argument('--checkpoint-path', default='.\\checkpoints\\deepsc_MAC', type=str)
parser.add_argument('--channel', default='AWGN', type=str, help='Please choose AWGN, Rayleigh, and Rician')
parser.add_argument('--MAX-LENGTH', default=30, type=int)
parser.add_argument('--MIN-LENGTH', default=4, type=int)
parser.add_argument('--d-model', default=128, type=int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num-layers', default=4, type=int)
parser.add_argument('--num-heads', default=8, type=int)
parser.add_argument('--batch-size', default=88, type=int)  # 这里控制的是每次拿(从数据集中读取)多少张牌(个句子)
parser.add_argument('--epochs', default=3, type=int)

parser.add_argument('--encoder-num-layer', default=4, type=int, help='The number of encoder layers')
parser.add_argument('--encoder-d-model', default=128, type=int, help='The output dimension of attention')
parser.add_argument('--encoder-d-ff', default=512, type=int, help='The output dimension of ffn')
parser.add_argument('--encoder-num-heads', default=8, type=int, help='The number heads')
parser.add_argument('--encoder-dropout', default=0.1, type=float, help='The encoder dropout rate')

parser.add_argument('--decoder-num-layer', default=4, type=int, help='The number of decoder layers')
parser.add_argument('--decoder-d-model', default=128, type=int, help='The output dimension of decoder')
parser.add_argument('--decoder-d-ff', default=512, type=int, help='The output dimension of ffn')
parser.add_argument('--decoder-num-heads', default=8, type=int, help='The number heads')
parser.add_argument('--decoder-dropout', default=0.1, type=float, help='The decoder dropout rate')




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

args = parser.parse_args()
args.vocab_file = args.vocab_file
vocab = json.load(open(args.vocab_file, 'rb'))
token_to_idx = vocab['token_to_idx']
idx_to_token = dict(zip(token_to_idx.values(), token_to_idx.keys()))
num_vocab = len(token_to_idx)
pad_idx = token_to_idx["<PAD>"]
start_idx = token_to_idx["<START>"]
end_idx = token_to_idx["<END>"]

with open(args.vocab_path, 'r') as f:  # 使用'r'而不是'rb'，因为json.load默认读取文本
    vocab = json.load(f)
args.vocab_size = len(vocab['token_to_idx'])
token_to_idx = vocab['token_to_idx']
args.pad_idx = token_to_idx["<PAD>"]
args.start_idx = token_to_idx["<START>"]
args.end_idx = token_to_idx["<END>"]
vocab = json.load(open(args.vocab_file, 'rb'))
token_to_idx = vocab['token_to_idx']


StoT = SeqtoText(token_to_idx, args.end_idx)

deepsc = DeepSC(args.num_layers, num_vocab, num_vocab,
                        num_vocab, num_vocab, args.d_model, args.num_heads,
                        args.dff, 0.1).to(device)

class KnowledgeBase(nn.Module):
    def __init__(self):
        super(KnowledgeBase, self).__init__()  # 输入是[1, 128] 输出是[8, 128]
        # 第一个全连接层，输入特征为128，输出特征为256
        self.fc1 = nn.Linear(128, 256)
        # 第二个全连接层，输入特征为256，输出特征为8*128
        self.fc2 = nn.Linear(256, 8 * 128)

    def forward(self, x):
        # 应用第一个全连接层并使用ReLU激活函数
        x = F.relu(self.fc1(x))
        # 应用第二个全连接层
        x = self.fc2(x)
        # 重塑成[bs, 8, 128]
        x = x.view(8, 128)
        return x

class KB_Mapping(nn.Module):
    def __init__(self):
        super(KB_Mapping, self).__init__()
        # 全连接层，输入特征为8*128，输出特征为1024（中间层）
        self.fc1 = nn.Linear(8 * 128, 1024)
        # 全连接层，输入特征为1024，输出特征为8*128
        self.fc2 = nn.Linear(1024, 8 * 128)

    def forward(self, x):
        # 将输入的[8, 128]张量展平为[1, 8*128]
        x = x.view(1, -1)
        # 通过第一个全连接层并使用ReLU激活函数
        x = F.relu(self.fc1(x))
        # 通过第二个全连接层
        x = self.fc2(x)
        # 重新将[1, 8*128]张量重塑为[8, 128]
        x = x.view(8, 128)
        return x

# Alice_ID = torch.randn(1, args.d_model).to(device)
# Bob_ID = torch.randn(1, args.d_model).to(device)
#
# alice = KnowledgeBase().to(device)
# bob = KnowledgeBase().to(device)
#
# alice_map = KB_Mapping().to(device)
#
# alice_output = alice(Alice_ID)
# alice_map_output = alice_map(alice_output)
# Alice_kb_final = alice_map_output.repeat(10, 1, 1)

# print(Alice_kb_final)



# 将F:\checkpoints\deepsc_mac\good\checkpoint_2119.pth中的部分网路提取到deepsc网络中
# checkpoint = torch.load('C:\d\code\deepsc_mac\checkpoints\deepsc_mac\good\checkpoint_2119.pth')
# model_state_dict = checkpoint['deepsc']
# # key_state_dict = checkpoint['key_ab']
# # eve_state_dict = checkpoint['eve']
# # deepsc.encoder.load_state_dict(model_state_dict['encoder'])
# # deepsc.decoder.load_state_dict(model_state_dict['decoder'])
# # deepsc.channel_encoder.load_state_dict(model_state_dict['channel_encoder'])
# # deepsc.channel_decoder.load_state_dict(model_state_dict['channel_decoder'])
# # deepsc.dense.load_state_dict(model_state_dict['dense'])
# deepsc.load_state_dict(checkpoint, strict=False)
# # key_ab.load_state_dict(key_state_dict)
# # eve.load_state_dict(eve_state_dict)
# deepsc = deepsc.to(device)
#
# checkpoint_new = {
#     "deepsc": deepsc.state_dict(),  # 单独保存deepsc
# }
#
# torch.save(checkpoint_new, 'C:\d\code\deepsc_mac\checkpoints\deepsc_mac\good\\test.pth')

# import torch
# import torch.nn as nn
# import torch.optim as optim
#
# # 定义简单的神经网络
# class SimpleNet(nn.Module):
#     def __init__(self):
#         super(SimpleNet, self).__init__()
#         self.fc = nn.Linear(10, 1)  # 输入维度为10，输出为1
#
#     def forward(self, x):
#         x = self.fc(x)
#         return torch.sigmoid(x)  # 使用Sigmoid激活函数
#
# # 创建模型实例
# model = SimpleNet()
#
# # 定义损失函数和优化器
# criterion = nn.BCELoss()
# optimizer = optim.SGD(model.parameters(), lr=0.01)
#
# # 模拟一些输入数据和目标标签
# batch_size = 4
# input_data = torch.randn(batch_size, 10)  # 随机生成输入数据
# # targets = torch.tensor([[1.0], [0.0], [1.0], [0.0]])  # 真实标签
# targets = torch.tensor([1., 0., 1., 0.])  # 真实标签
#
# # 前向传播
# outputs = model(input_data)  # 得到模型输出（经过Sigmoid）
#
# # 计算损失
# loss = criterion(outputs, targets)
#
# # 反向传播和优化
# optimizer.zero_grad()  # 清零梯度
# loss.backward()  # 反向传播
# optimizer.step()  # 更新权重
#
# print("损失值:", loss.item())
# import torch
# x = torch.tensor([1], dtype=torch.float32, requires_grad=True)
# y1 = x ** 2
# y2 = x ** 3
# y3 = y1 + y2
# y3.backward()
# print(x.grad)
# x.grad.data.zero_()
# y3.backward()
# print(x.grad)


import torch
import torch.nn as nn
import torch.optim as optim

# # 创建一个可学习的参数，但不将其添加到任何模型中
# learnable_param = nn.Parameter(torch.randn(8, 128))
#
# print(learnable_param)
#
# # 创建一个优化器，直接包含这个参数
# optimizer = optim.Adam([learnable_param], lr=1e-4)
#
# # 假设有一个损失函数和输入数据
# criterion = nn.MSELoss()
# input_data = torch.randn(8, 128)
# target = torch.randn(8, 128)
#
# for i in range(100):
#     # 前向传播
#     output = input_data + learnable_param
#
#     # 计算损失
#     loss = criterion(output, target)
#
#     # 反向传播
#     optimizer.zero_grad()  # 清空梯度
#     loss.backward()  # 计算梯度
#     optimizer.step()  # 更新参数
#
#     # 检查参数是否更新
#     print(loss.item())




# train_iterator = return_iter(args, 'test')  # 从训练数据集中抓牌，得到的是一个dataloader类型的对象（其实就是dataloder 用法完全一样
# train_iterator_eve = return_iter_eve(args, 'test')
# print(len(train_iterator_eve))
#
#
# pbar = tqdm(train_iterator)  # 进度条
# print(type(pbar))
# pbar_eve = tqdm(train_iterator_eve)
# pbar_eve_iter = iter(pbar_eve)
# print(type(pbar_eve_iter))
# i = 1
# for sents in pbar:  # 每个batch的数据
#     # print("sents.shape: ", sents.shape)  # sents.shape:  torch.Size([128, 31])  说明一次拿了128张牌(句子)，每张牌(句子)31个数字(单词的索引)
#     sents = sents.to(device)
#     # 在pbar_eve中取一条数据
#     try:
#         sents_eve = next(pbar_eve_iter)  # 取一个 batch 的攻击者数据
#     except StopIteration:
#         # 如果 pbar_eve 已经遍历完，重新开始
#         pbar_eve_iter = iter(pbar_eve)
#         sents_eve = next(pbar_eve_iter)
#     print(sents)
#     print(sents_eve)
#     print(i)
#     i += 1

# def performance_bleu(args, SNR, deepsc, alice_bob_mac, key_ab, eve, Alice_KB, Bob_KB, Eve_KB, Alice_mapping, Bob_mapping, Eve_mapping):
#     # similarity = Similarity(args.bert_config_path, args.bert_checkpoint_path, args.bert_dict_path)
#     bleu_score_1gram = BleuScore(1, 0, 0, 0)
#
#     test_iterator = return_iter(args, 'test')
#     test_iterator_eve = return_iter_eve(args, 'test')
#     iter_eve = iter(test_iterator_eve)
#
#     StoT = SeqtoText(token_to_idx, end_idx)
#     score = []
#     score2 = []
#     cos_list = []
#     zheng_mac_list = []
#     eve_mac_0_list = []
#     tamper_0_list = []
#
#     deepsc.eval()
#     key_ab.eval()
#     alice_bob_mac.eval()
#     eve.eval()
#     Alice_KB.eval()
#     Bob_KB.eval()
#     Eve_KB.eval()
#     Alice_mapping.eval()
#     Bob_mapping.eval()
#     Eve_mapping.eval()
#
#
#     with torch.no_grad():
#         for epoch in range(args.epochs):
#             Tx_word = []
#             Rx_word = []
#             zheng_mac_list_tmp = []
#             eve_mac_0_list_tmp = []
#             tamper_0_list_tmp = []
#
#             for snr in tqdm(SNR):  # 对每个信噪比 所有的数据
#                 word = []
#                 target_word = []
#                 noise_std = SNR_to_noise(snr)
#
#                 total_zheng_mac = 0
#                 total_eve_mac_0 = 0
#                 total_tamper_0 = 0
#                 for sents in test_iterator:
#                     sents = sents.to(device)
#                     try:
#                         sents_eve = next(iter_eve).to(device)
#                     except:
#                         iter_eve = iter(test_iterator_eve)
#                         sents_eve = next(iter_eve).to(device)
#                     # src = batch.src.transpose(0, 1)[:1]
#                     target = sents
#
#                     # out, zheng_mac_accuracy, eve_mac_0, tamper_0, key_0 = greedy_decode(args, deepsc, alice_bob_mac, key_ab, eve, Alice_KB, Bob_KB, Alice_ID, Bob_ID, sents, noise_std, args.MAX_LENGTH, pad_idx,
#                     #                     start_idx, args.channel)
#                     # , zheng_mac_accuracy, eve_mac_0, tamper_0
#                     out, zheng_mac_accuracy, eve_mac_0, tamper_0 = greedy_decode(args, deepsc, alice_bob_mac, key_ab, eve, Alice_KB, Bob_KB, Eve_KB, Alice_mapping, Bob_mapping, Eve_mapping,
#                                                                                         sents, sents_eve,
#                                                                                         noise_std, args.MAX_LENGTH,
#                                                                                         pad_idx,
#                                                                                         start_idx, args.channel)
#                     total_zheng_mac += zheng_mac_accuracy
#                     total_eve_mac_0 += eve_mac_0
#                     total_tamper_0 += tamper_0
#                     # total_key_0 += key_0
#
#                     # 下面是将数字句子转换为字符串句子
#                     sentences = out.cpu().numpy().tolist()  # list bs长度 每个元素是一个句子，句子也是一个List,用数字表示
#                     result_string = list(map(StoT.sequence_to_text, sentences))  # list 每个元素是一个字符串句子
#                     word = word + result_string  # list 数据集的所有预测句子全加进来
#                     # print(result_string)
#
#                     target_sent = target.cpu().numpy().tolist()
#                     result_string = list(map(StoT.sequence_to_text, target_sent))
#                     target_word = target_word + result_string  # list 数据集的所有原始句子全加进来
#                     # print(result_string, end='\n\n')
#
#                 Tx_word.append(word)  # list 长度7 每个元素是list 即Tx_word[0][0]是第一个信噪比下的第一个字符串句子
#                 Rx_word.append(target_word)
#                 # average_cos = total_cos / len(test_iterator)
#                 average_zheng_mac = total_zheng_mac / len(test_iterator)  # 当前信噪比下的平均准确率(一个数)
#                 average_eve_mac_0 = total_eve_mac_0 / len(test_iterator)
#                 average_tamper_0 = total_tamper_0 / len(test_iterator)
#
#                 # cos_list_tmp.append(average_cos)
#                 zheng_mac_list_tmp.append(average_zheng_mac)
#                 eve_mac_0_list_tmp.append(average_eve_mac_0)
#                 tamper_0_list_tmp.append(average_tamper_0)
#
#             bleu_score = []
#             sim_score = []
#             # cos_list.append(cos_list_tmp)
#             zheng_mac_list.append(zheng_mac_list_tmp)
#             eve_mac_0_list.append(eve_mac_0_list_tmp)
#             tamper_0_list.append(tamper_0_list_tmp)
#
#             for sent1, sent2 in zip(Tx_word, Rx_word):  # sent1是第一个信噪比下的所有句子
#                 # 1-gram
#                 bleu_score.append(bleu_score_1gram.compute_score(sent1, sent2))  # 每个元素是list,bleu_score[0][0]是第一个信噪比下的第一个句子的BLEU分数,这样计算了所有的句子
#                 # sim_score.append(similarity.compute_similarity(sent1, sent2)) # 7*num_sent
#             bleu_score = np.array(bleu_score)  # 尺寸为7 * 句子数
#             bleu_score = np.mean(bleu_score, axis=1)  # 每个信噪比下的所有句子的平均BLEU分数
#             score.append(bleu_score)  # 存储到当前epoch中
#
#             # sim_score = np.array(sim_score)
#             # sim_score = np.mean(sim_score, axis=1)
#             # score2.append(sim_score)
#
#     score1 = np.mean(np.array(score), axis=0)  # 每个信噪比下的所有句子的平均BLEU分数(按照epoch平均)
#     # score2 = np.mean(np.array(score2), axis=0)
#     zheng_mac_score = np.mean(np.array(zheng_mac_list), axis=0)
#     eve_mac_0_score = np.mean(np.array(eve_mac_0_list), axis=0)
#     tamper_0_score = np.mean(np.array(tamper_0_list), axis=0)
#
#     # return score1, zheng_mac_score, eve_mac_0_score, tamper_0_score, key_0_score  # , score2
#     return score1, zheng_mac_score, eve_mac_0_score, tamper_0_score


class PowerPreservingNet(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=32, seq_len=32):  # 添加seq_len参数
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len  # 保存序列长度
        self.fc1 = nn.Linear(input_dim * seq_len, hidden_dim)  # 使用seq_len
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, input_dim * seq_len)  # 使用seq_len

    def forward(self, x):
        input_power = self.calculate_power(x)
        batch_size = x.size(0)

        # 展平输入（使用self.seq_len）
        flat_x = x.view(batch_size, self.input_dim * self.seq_len)

        # 通过网络
        hidden = self.relu(self.fc1(flat_x))
        output_flat = self.fc2(hidden)

        # 恢复形状（使用self.seq_len）
        output = output_flat.view(batch_size, self.seq_len, self.input_dim)

        # 功率匹配
        output_power = self.calculate_power(output)
        scale = torch.sqrt(input_power / (output_power + 1e-8)).unsqueeze(1).unsqueeze(2)
        return output * scale

    @staticmethod
    def calculate_power(signals: torch.Tensor) -> torch.Tensor:
        """计算每个信号的全局功率"""
        squared = signals.pow(2)
        return squared.mean(dim=[1, 2])

# # 测试代码
# p = PowerPreservingNet()
# x = torch.randn(88, 32, 16)  # 输入张量
# output = p(x)  # 前向传播
#
# # 批量计算所有样本的功率
# input_powers = PowerPreservingNet.calculate_power(x)
# output_powers = PowerPreservingNet.calculate_power(output + x)
#
# # 打印结果
# for i, (in_pow, out_pow) in enumerate(zip(input_powers, output_powers)):
#     print(f"信号 {i+1}: 输入功率={in_pow:.6f}, 输出功率={out_pow:.6f}, 比率={out_pow/in_pow:.6f}")




