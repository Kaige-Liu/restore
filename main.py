# -*- coding: utf-8 -*-
import os
import argparse
import time
import json
import torch
import random
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from models.transceiver import DeepSC, Key_net, Attacker, MAC
from utlis.tools import SNR_to_noise, SeqtoText
from utlis.trainer import initNetParams, train_step, val_step, train_mi
from dataset.dataloader import return_iter, return_iter_10, return_iter_eve
from models.transceiver import DeepSC, KnowledgeBase, KB_Mapping, BLEU_predictor
from models.mutual_info import Mine
from tqdm import tqdm

parser = argparse.ArgumentParser()  # 创建一个命令行参数解释器
parser.add_argument('--vocab-file', default='./data/vocab.json', type=str)
parser.add_argument('--vocab_path', default='./data/vocab.json', type=str)
parser.add_argument('--checkpoint-path', default='./checkpoints/deepsc_mac', type=str)
parser.add_argument('--channel', default='Rayleigh', type=str, help='Please choose AWGN, Rayleigh, and Rician')
parser.add_argument('--MAX-LENGTH', default=30, type=int)
parser.add_argument('--MIN-LENGTH', default=4, type=int)
parser.add_argument('--d-model', default=128, type=int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num-layers', default=4, type=int)
parser.add_argument('--num-heads', default=8, type=int)
parser.add_argument('--batch-size', default=88, type=int)  # 这里控制的是每次拿(从数据集中读取)多少张牌(个句子)
parser.add_argument('--epochs', default=400, type=int)

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

def setup_seed(seed):  # 设置随机种子，根本没用
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 为当前GPU设置随机种子
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def train(epoch, args, net, alice_bob_mac, key_ab, eve, Alice_KB, Bob_KB, Eve_KB, Alice_mapping, Bob_mapping, Eve_mapping, mi_net=None):  # 当前训练的轮数，命令行参数，模型，互信息网络（默认是None，也就是不训互信息网络）
    train_iterator = return_iter(args, 'train')  # 从训练数据集中抓牌，得到的是一个dataloader类型的对象（其实就是dataloder 用法完全一样）
    print("len", len(train_iterator))
    train_iterator_eve = return_iter_eve(args, 'train')
    print("len", len(train_iterator_eve))

    pbar = tqdm(train_iterator)  # 进度条
    pbar_eve = tqdm(train_iterator_eve, leave=False)
    pbar_eve_iter = iter(train_iterator_eve)
    # print("---------------------------pbar---------------------------")
    # print("pbar: ", pbar)  # 输出就是一个静态的一直是0的进度条

    batch = 0
    total_deepsc = 0
    total_normal = 0
    total_bleu = 0
    total_eve_mac_0 = 0
    total_eve_mac_1 = 0
    total_tamper_0 = 0
    total_tamper_1 = 0
    total_key_0 = 0

    noise_std = np.random.uniform(SNR_to_noise(3), SNR_to_noise(10), size=(1))  # 生成介于信噪比为5和10之间的随机的噪声标准差，
    # print("---------------------------noise_std---------------------------")
    # print("noise_std: ", noise_std)  # noise_std:  [0.37055991]
    # print("noise_std[0]:", noise_std[0])  # noise_std[0]: 0.3705599051835091
    # print("---------------------------noise_std---------------------------")

    for sents in pbar:  # 每个batch的数据
        # print("sents.shape: ", sents.shape)  # sents.shape:  torch.Size([128, 31])  说明一次拿了128张牌(句子)，每张牌(句子)31个数字(单词的索引)
        sents = sents.to(device)
        # 在pbar_eve中取一条数据
        try:
            sents_eve = next(pbar_eve_iter).to(device)  # 取一个 batch 的攻击者数据
        except StopIteration:
            # 如果 pbar_eve 已经遍历完，重新开始
            pbar_eve_iter = iter(pbar_eve)
            sents_eve = next(pbar_eve_iter).to(device)

        if mi_net is not None:
            mi = train_mi(net, mi_net, sents, 0.1, pad_idx, mi_opt, args.channel)
            loss = train_step(net, sents, sents, 0.1, pad_idx,
                              optimizer, args.channel, mi_net)  # 这是训练的一个batch的loss
            total0 += loss
            pbar.set_description(
                'Epoch: {};  Type: Train; Loss: {:.5f}; MI {:.5f}'.format(
                    epoch + 1, loss, mi
                )
            )
        else:
            # loss0, loss1, loss2, loss3, loss4, loss5, loss_deepsc, loss_avalanche, loss_normal = (
            #             train_step(args, batch, net, key_ab, eve, sents, sents, noise_std[0], pad_idx,
            #                   optimizer_joint, args.channel))


            # loss_normal, eve_mac_0, eve_mac_1, tamper, key_0 = train_step(args, epoch, batch, net, alice_bob_mac, key_ab, eve, Alice_KB, Bob_KB, Alice_ID, Bob_ID, sents, sents, noise_std[0], pad_idx,
            #                       optimizer_joint, args.channel)
            # loss_deepsc = train_step(args, epoch, batch, net, alice_bob_mac,
            #                                                               key_ab, eve, Alice_KB, Bob_KB, Alice_mapping, Bob_mapping, sents, sents, noise_std[0], pad_idx,
            #                                                               optimizer_joint, args.channel)
            # loss_deepsc, loss_normal, eve_mac_0, eve_mac_1, tamper_0, tamper_1 = train_step(args, epoch, batch, net, alice_bob_mac, key_ab, eve, Alice_KB, Bob_KB, Eve_KB, Alice_mapping, Bob_mapping, Eve_mapping, sents, sents, noise_std[0], pad_idx,
            #                         optimizer_joint, args.channel)
            # 注意：下面的loss_deepsc和loss_bleu都是在第三类攻击下的了
            loss_deepsc, loss_normal, eve_mac_0, eve_mac_1, tamper_0, tamper_1 = train_step(args, epoch, batch, net,
                                                                                            alice_bob_mac, key_ab, eve,
                                                                                            Alice_KB, Bob_KB, Eve_KB,
                                                                                            Alice_mapping, Bob_mapping,
                                                                                            Eve_mapping,
                                                                                            sents, sents, sents_eve,
                                                                                            noise_std[0], pad_idx,
                                                                                            optimizer_joint,
                                                                                            args.channel)


            # loss_deepsc = train_step(args, epoch, batch, net, alice_bob_mac, key_ab, eve, sents, sents, noise_std[0], pad_idx,
            #                         optimizer_joint, args.channel)
            total_deepsc += loss_deepsc
            total_normal += loss_normal
            # total_bleu += loss_bleu
            total_eve_mac_0 += eve_mac_0
            total_eve_mac_1 += eve_mac_1
            total_tamper_0 += tamper_0
            total_tamper_1 += tamper_1
            # total_key_0 += key_0

        batch += 1

    print("================train======================")
    print("epoch: ", epoch)
    # print("loss_deepsc: ", total_deepsc / len(train_iterator))
    # print("loss_normal: ", total_normal / len(train_iterator))
    print("================train======================")

    # return total0 / len(train_iterator), total1 / len(train_iterator), total2 / len(train_iterator), total3 / len(train_iterator),total4 / len(train_iterator), total5 / len(train_iterator), total_deepsc / len(train_iterator), total_avalanche / len(train_iterator), total_normal / len(train_iterator)
    # return total_normal / len(train_iterator), total_eve_mac_0 / len(train_iterator), total_eve_mac_1 / len(train_iterator), total_tamper / len(train_iterator), total_key_0 / len(train_iterator)
    # return total_deepsc / len(train_iterator)
    return total_deepsc / len(train_iterator), total_normal / len(train_iterator), total_eve_mac_0 / len(train_iterator), total_eve_mac_1 / len(train_iterator), total_tamper_0 / len(train_iterator), total_tamper_1 / len(train_iterator)

def validate(epoch, args, net, alice_bob_mac, key_ab, eve, Alice_KB, Bob_KB, Eve_KB, Alice_mapping, Bob_mapping, Eve_mapping):  # epoch表示正在验证的是第几轮
    test_iterator = return_iter(args, 'test')  # 从测试数据集中抓牌
    test_iterator_eve = return_iter_eve(args, 'test')

    net.eval()  # 将模型设置为验证模式
    alice_bob_mac.eval()
    key_ab.eval()
    eve.eval()
    Alice_KB.eval()
    Bob_KB.eval()
    Eve_KB.eval()
    Alice_mapping.eval()
    Bob_mapping.eval()
    Eve_mapping.eval()

    pbar = tqdm(test_iterator)
    pbar_eve = tqdm(test_iterator_eve, leave=False)
    pbar_eve_iter = iter(test_iterator_eve)
    batch = 0
    total_deepsc = 0
    # total_deepsc_burst = 0
    total_normal = 0
    # total_bleu = 0
    total_eve_mac_0 = 0
    total_eve_mac_1 = 0
    total_tamper_0 = 0
    total_tamper_1 = 0
    total_key_0 = 0

    mac_total = 0  # 这里的loss是一般的loss，不是互信息量
    deepsc_total = 0
    with torch.no_grad():  # 不需要计算梯度，看牌前的常规操作，不用管
        for sents in pbar:  # 其实就是for data in dataloader,这是[128, 31]的张量
            sents = sents.to(device)  # 将数据放到GPU上
            try:
                sents_eve = next(pbar_eve_iter).to(device)  # 取一个 batch 的攻击者数据
            except StopIteration:
                # 如果 pbar_eve 已经遍历完，重新开始
                pbar_eve_iter = iter(pbar_eve)
                sents_eve = next(pbar_eve_iter).to(device)

            # loss0, loss1, loss2, loss3, loss4, loss5, loss_deepsc, loss_avalanche, loss_normal = (
            #             val_step(args, batch, net, key_ab, eve, sents, sents, 0.1, pad_idx, args.channel))
            # loss_deepsc, loss_normal, eve_mac_0, eve_mac_1, tamper, key_0 = val_step(args, batch, net, alice_bob_mac, key_ab, eve, Alice_KB, Bob_KB, Alice_ID, Bob_ID, sents, sents, 0.1, pad_idx, args.channel)
            # loss_deepsc = val_step(args, batch, net, alice_bob_mac, key_ab, eve, Alice_KB, Bob_KB, Alice_mapping, Bob_mapping, sents, sents, 0.1, pad_idx, args.channel)
            # loss_deepsc, loss_normal, eve_mac_0, eve_mac_1, tamper_0, tamper_1 = val_step(args, batch, net, alice_bob_mac, key_ab, eve, Alice_KB, Bob_KB, Eve_KB, Alice_mapping, Bob_mapping, Eve_mapping, sents, sents, 0.1, pad_idx, args.channel)
            loss_deepsc, loss_normal, eve_mac_0, eve_mac_1, tamper_0, tamper_1 = val_step(args, batch, net, alice_bob_mac, key_ab, eve, Alice_KB, Bob_KB, Eve_KB, Alice_mapping, Bob_mapping, Eve_mapping,
                                                                                          sents, sents, sents_eve, 0.1, pad_idx, args.channel)

            # loss_deepsc = val_step(args, batch, net, alice_bob_mac, key_ab, eve, sents, sents, 0.1, pad_idx, args.channel)

            total_deepsc += loss_deepsc
            # total_deepsc_burst += loss_deepsc_burst
            total_normal += loss_normal
            # total_bleu += loss_bleu
            total_eve_mac_0 += eve_mac_0
            total_eve_mac_1 += eve_mac_1
            total_tamper_0 += tamper_0
            total_tamper_1 += tamper_1
            # total_key_0 += key_0
            # pbar.set_description(  # 设置进度条的描述
            #     'Epoch: {};  Type: VAL; Loss: {:.5f}'.format(
            #         epoch + 1, loss_total
            #     )
            # )
            batch += 1
        # pbar.set_description(
        #     'Epoch: {};  Type: VAL; Loss: {:.5f}'.format(
        #         epoch, total / len(test_iterator)
        #     )
        # )
    print("================validate======================")
    print("epoch: ", epoch)
    print("loss_deepsc: ", total_deepsc / len(test_iterator))
    # print("loss_deepsc_burst: ", total_deepsc_burst / len(test_iterator))
    print("loss_normal: ", total_normal / len(test_iterator))
    print("================validate======================")

    # 计算平均的loss
    # 后面的len其实就是batch的个数128，也就是拿了多少次牌（而不是每次拿了多少张牌）
    # return total0 / len(test_iterator), total1 / len(test_iterator), total2 / len(test_iterator), total3 / len(test_iterator), total4 / len(test_iterator), total5 / len(test_iterator), total_deepsc / len(test_iterator), total_avalanche / len(test_iterator), total_normal / len(test_iterator)
    # return total_deepsc / len(test_iterator), total_normal / len(test_iterator), total_eve_mac_0 / len(test_iterator), total_eve_mac_1 / len(test_iterator), total_tamper / len(test_iterator), total_key_0 / len(test_iterator)
    # return total_deepsc / len(test_iterator)
    return total_deepsc / len(test_iterator), total_normal / len(test_iterator), total_eve_mac_0 / len(test_iterator), total_eve_mac_1 / len(test_iterator), total_tamper_0 / len(test_iterator), total_tamper_1 / len(test_iterator)

if __name__ == '__main__':
    now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    # if not os.path.isdir("./checkpoints/deepsc_hiding/" + now):
    #     os.mkdir("./checkpoints/deepsc_hiding/" + now)
    os.mkdir("./checkpoints/deepsc_mac/" + now)
    writer = SummaryWriter(log_dir="./logs/deepsc_mac/" + now)

    """ preparing the dataset """
    train_start_time = time.time()
    torch.manual_seed(5)  # 等价于tf.random.set_seed(5)，确保随机性一致
    args = parser.parse_args()  # 将命令行参数存储到args中，用args.参数名来调用访问对应的参数

    # 使用with语句确保文件正确关闭
    with open(args.vocab_path, 'r') as f:  # 使用'r'而不是'rb'，因为json.load默认读取文本
        vocab = json.load(f)
    args.vocab_size = len(vocab['token_to_idx'])
    token_to_idx = vocab['token_to_idx']
    args.pad_idx = token_to_idx["<PAD>"]
    args.start_idx = token_to_idx["<START>"]
    args.end_idx = token_to_idx["<END>"]
    vocab = json.load(open(args.vocab_file, 'rb'))
    token_to_idx = vocab['token_to_idx']
    num_vocab = len(token_to_idx)
    # 下面是三个比较特殊的单词，这里是将这三个单词的索引提取出来，也就是对应的数字
    pad_idx = token_to_idx["<PAD>"]  # 0
    start_idx = token_to_idx["<START>"]  # 1
    end_idx = token_to_idx["<END>"]  # 2

    StoT = SeqtoText(token_to_idx, args.end_idx)

    """ define optimizer and loss function """
    # 默认层数是4层，d_model是128，dff是512，num_heads是8
    # 源语言和目标语言的词汇量都是num_vocab，也就是字典里面的单词的个数
    # 源语言序列的最大长度也是num_vocab
    # d_model是隐藏层的维度，num_heads是多头注意力的头数，dff是前馈神经网络Feedforward层的维度
    # dropout是丢弃概率，为了防止过拟合
    # deepsc是一个transfomer模型，用来进行编码和解码
    deepsc = DeepSC(args.num_layers, num_vocab, num_vocab,
                    num_vocab, num_vocab, args.d_model, args.num_heads,
                    args.dff, 0.1).to(device)
    # 加载模型
    # checkpoint = torch.load('F:\checkpoints\deepsc_mac\\2024-09-04-16_52_50\\checkpoint_3445.pth')
    # model_state_dict = checkpoint
    # deepsc.load_state_dict(model_state_dict)
    # deepsc = deepsc.to(device)

    mi_net = Mine().to(device)  # 计算通信网络的互信息量的 其实没啥用 可以全都当做None来训

    alice_bob_mac = MAC().to(device)
    key_ab = Key_net(args).to(device)
    eve = Attacker().to(device)
    # bleu_predictor = BLEU_predictor().to(device)

    Alice_KB = KnowledgeBase().to(device)
    Bob_KB = KnowledgeBase().to(device)
    Eve_KB = KnowledgeBase().to(device)

    Alice_mapping = KB_Mapping().to(device)
    Bob_mapping = KB_Mapping().to(device)
    Eve_mapping = KB_Mapping().to(device)

    # initNetParams(deepsc)
    # initNetParams(alice_bob_mac)
    # initNetParams(key_ab)
    # initNetParams(eve)
    # initNetParams(bleu_predictor)
    # initNetParams(Alice_KB)
    # initNetParams(Bob_KB)
    # initNetParams(Alice_mapping)  # 没训练的初始化
    # initNetParams(Bob_mapping)
    # initNetParams(Eve_KB)
    # initNetParams(Eve_mapping)
    # checkpoint = torch.load('C:\d\code\deepsc_mac\checkpoints\deepsc_mac\good\checkpoint_2119.pth')
    # checkpoint = torch.load('C:\d\code\deepsc_mac\checkpoints\deepsc_mac\\2024-09-27-11_30_12\checkpoint_192.pth')  # 这个是预训练好的deepsc 正向 第一类攻击
    # checkpoint = torch.load('C:\d\code\deepsc_mac\checkpoints\deepsc_mac\\2024-10-10-11_17_52\checkpoint_326.pth')
    # checkpoint = torch.load('C:\d\code\deepsc_mac\checkpoints\deepsc_mac\\2024-12-12-11_18_09\checkpoint_244.pth')
    # checkpoint = torch.load('C:\d\code\deepsc_mac\checkpoints\deepsc_mac\\2024-12-16-17_37_47\checkpoint_2257.pth')
    # checkpoint = torch.load('C:\d\code\deepsc_mac\checkpoints\deepsc_mac\\2025-01-09-20_33_09\checkpoint_2752.pth')
    # checkpoint = torch.load('C:\d\code\deepsc_mac\checkpoints\deepsc_mac\\2025-01-17-18_29_38\checkpoint_289.pth')
    # checkpoint = torch.load('C:\d\code\deepsc_mac\checkpoints\deepsc_mac\\2025-02-24-19_27_26\checkpoint_1618.pth')
    # checkpoint = torch.load(r'C:\d\code\deepsc_mac\checkpoints\deepsc_mac\2025-03-11-23_49_38\checkpoint_247.pth')
    # checkpoint = torch.load('C:\d\code\deepsc_mac\checkpoints\deepsc_mac\\2025-05-14-03_54_30\checkpoint_70.pth')
    # checkpoint = torch.load(r'/root/autodl-tmp/deepsc_mac/checkpoints/deepsc_mac/2025-12-13-05_11_21/checkpoint_195.pth')
    # checkpoint = torch.load(r'/root/autodl-tmp/deepsc_mac/checkpoints/deepsc_mac/2025-12-13-15_40_57/checkpoint_23.pth')
    # checkpoint = torch.load(r'/root/autodl-tmp/deepsc_mac/checkpoints/deepsc_mac/2025-12-13-17_37_22/checkpoint_17.pth')
    checkpoint = torch.load(r'/root/autodl-tmp/deepsc_mac/checkpoints/deepsc_mac/2025-12-13-19_33_31/checkpoint_109.pth')  # 之前三个检测率都最好
    # checkpoint = torch.load(r'/root/autodl-tmp/deepsc_mac/checkpoints/deepsc_mac/2025-12-14-05_11_08/checkpoint_103.pth')  # 目前三个检测率都最好
    checkpoint_deepsc = torch.load(r'/root/autodl-tmp/deepsc_mac/checkpoints/deepsc_mac/checkpoint_247.pth')  # 去年训的不错的


    model_state_dict = checkpoint_deepsc['deepsc']
    alice_bob_mac_state_dict = checkpoint['alice_bob_mac']
    key_state_dict = checkpoint['key_ab']
    eve_state_dict = checkpoint['eve']
    Alice_KB_state_dict = checkpoint['Alice_KB']
    Bob_KB_state_dict = checkpoint['Bob_KB']
    Eve_KB_state_dict = checkpoint['Eve_KB']
    Alice_mapping_state_dict = checkpoint['Alice_mapping']
    Bob_mapping_state_dict = checkpoint['Bob_mapping']
    Eve_mapping_state_dict = checkpoint['Eve_mapping']
    # bleu_predictor_state_dict = checkpoint['bleu_predictor']

    deepsc.load_state_dict(model_state_dict)
    alice_bob_mac.load_state_dict(alice_bob_mac_state_dict)
    # initNetParams(alice_bob_mac)
    key_ab.load_state_dict(key_state_dict)
    eve.load_state_dict(eve_state_dict)
    # initNetParams(eve.burst)  # 这个是新的网络
    Alice_KB.load_state_dict(Alice_KB_state_dict)
    Bob_KB.load_state_dict(Bob_KB_state_dict)
    Eve_KB.load_state_dict(Eve_KB_state_dict)
    Alice_mapping.load_state_dict(Alice_mapping_state_dict)
    Bob_mapping.load_state_dict(Bob_mapping_state_dict)
    Eve_mapping.load_state_dict(Eve_mapping_state_dict)
    # bleu_predictor.load_state_dict(bleu_predictor_state_dict)

    deepsc = deepsc.to(device)
    alice_bob_mac = alice_bob_mac.to(device)
    key_ab = key_ab.to(device)
    eve = eve.to(device)
    Alice_KB = Alice_KB.to(device)
    Bob_KB = Bob_KB.to(device)
    Eve_KB = Eve_KB.to(device)
    Alice_mapping = Alice_mapping.to(device)
    Bob_mapping = Bob_mapping.to(device)
    Eve_mapping = Eve_mapping.to(device)
    # bleu_predictor = bleu_predictor.to(device)

    optimizer = torch.optim.Adam(deepsc.parameters(),
                                 lr=1e-5, betas=(0.9, 0.98), eps=1e-8, weight_decay=5e-4)
    mi_opt = torch.optim.Adam(mi_net.parameters(), lr=1e-3)
    # opt = NoamOpt(args.d_model, 1, 4000, optimizer)

    # 联合训练的优化器
    optimizer_joint = torch.optim.Adam(
        list(deepsc.parameters()) +
        list(alice_bob_mac.parameters()) +
        list(key_ab.parameters()) +
        list(eve.parameters()) +
        list(Alice_KB.parameters()) +
        list(Bob_KB.parameters()) +
        list(Eve_KB.parameters()) +
        list(Alice_mapping.parameters()) +
        list(Bob_mapping.parameters()) +
        list(Eve_mapping.parameters()),
        # list(bleu_predictor.parameters()),
        lr=1e-4, betas=(0.9, 0.98), eps=1e-8, weight_decay=5e-4)
    # 模型的设置可能有点问题
    # 只优化deepsc的优化器
    # optimizer_joint = torch.optim.Adam(deepsc.parameters(),
    #                              lr=1e-4, betas=(0.9, 0.98), eps=1e-8, weight_decay=5e-4)

    # 下面就是训练deepsc模型
    for epoch in range(args.epochs):  # 默认训练完整的数据集80轮
        start = time.time()  # 记录每轮开始时间（没用到）
        record_loss = 1000  # 其实是loss，设置的大一点

        # loss0, loss1, loss2, loss3, loss4, loss5, loss_deepsc, loss_avalanche, loss_normal = (
        #                                         train(epoch, args, deepsc, key_ab, eve))  # 训练deepsc模型
        # loss_normal, eve_mac_0, eve_mac_1, tamper, key_0 = train(epoch, args, deepsc, alice_bob_mac, key_ab, eve, Alice_KB, Bob_KB, Alice_ID, Bob_ID)  # 训练deepsc模型
        # loss_deepsc, loss_normal, eve_mac_0, eve_mac_1, tamper_0, tamper_1 = train(epoch, args, deepsc, alice_bob_mac, key_ab, eve, Alice_KB, Bob_KB, Eve_KB, Alice_mapping, Bob_mapping, Eve_mapping)  # 单独预训练deepsc
        loss_deepsc, loss_normal, eve_mac_0, eve_mac_1, tamper_0, tamper_1 = train(epoch, args, deepsc, alice_bob_mac, key_ab, eve, Alice_KB, Bob_KB, Eve_KB, Alice_mapping, Bob_mapping, Eve_mapping)  # 单独预训练deepsc


        # (loss0_test, loss1_test, loss2_test, loss3_test,
        #  loss4_test, loss5_test, loss_deepsc_test,
        #  loss_avalanche_test, loss_normal_test) = validate(epoch, args, deepsc, key_ab, eve)  # 验证deepsc模型，得到平均的loss
        #
        # loss_deepsc_test, loss_normal_test, eve_mac_0_test, eve_mac_1_test, tamper_test, key_0_test = validate(epoch, args, deepsc, alice_bob_mac, key_ab, eve, Alice_KB, Bob_KB, Alice_ID, Bob_ID)
        # loss_deepsc_test, loss_normal_test, eve_mac_0_test, eve_mac_1_test, tamper_0_test, tamper_1_test = validate(epoch, args, deepsc, alice_bob_mac, key_ab, eve, Alice_KB, Bob_KB, Eve_KB, Alice_mapping, Bob_mapping, Eve_mapping)
        loss_deepsc_test, loss_normal_test, eve_mac_0_test, eve_mac_1_test, tamper_0_test, tamper_1_test = validate(epoch, args, deepsc, alice_bob_mac, key_ab, eve, Alice_KB, Bob_KB, Eve_KB, Alice_mapping, Bob_mapping, Eve_mapping)

        # loss_deepsc_test = validate(epoch, args, deepsc, alice_bob_mac, key_ab, eve)  # 验证deepsc模型，得到平均的loss


        if loss_deepsc_test < record_loss:  # 如果验证的loss小于之前的loss（性能更好了）
            # if not os.path.exists(args.checkpoint_path):
            #     os.makedirs(args.checkpoint_path)  # 创建文件
            # with open("F:\checkpoints\deepsc_mac\\" + now + '/checkpoint_{}.pth'.format(str(epoch + 1).zfill(2)), 'wb') as f:
            #     torch.save(deepsc.state_dict(), f)  # 保存模型到上面的文件中
            checkpoint = {
                "deepsc": deepsc.state_dict(),
                "alice_bob_mac": alice_bob_mac.state_dict(),
                "key_ab": key_ab.state_dict(),
                "eve": eve.state_dict(),
                "Alice_KB": Alice_KB.state_dict(),
                "Bob_KB": Bob_KB.state_dict(),
                "Eve_KB": Eve_KB.state_dict(),
                "Alice_mapping": Alice_mapping.state_dict(),
                "Bob_mapping": Bob_mapping.state_dict(),
                "Eve_mapping": Eve_mapping.state_dict(),
                # "bleu_predictor": bleu_predictor.state_dict(),
            }
            # checkpoint = {
            #     "deepsc": deepsc.state_dict(),  # 单独保存deepsc
            # }
            torch.save(checkpoint, './checkpoints/deepsc_mac/' + now + '/checkpoint_{}.pth'.format(epoch))
            record_loss = loss_deepsc_test  # 更新最小的准确率

        writer.add_scalar('Loss_deepsc', loss_deepsc, epoch)
        writer.add_scalar('Loss_normal', loss_normal, epoch)
        # writer.add_scalar('Loss_bleu', loss_bleu, epoch)
        writer.add_scalar('eve_mac_0', eve_mac_0, epoch)
        writer.add_scalar('eve_mac_1', eve_mac_1, epoch)
        writer.add_scalar('tamper_0', tamper_0, epoch)
        writer.add_scalar('tamper_1', tamper_1, epoch)
        # writer.add_scalar('key_0', key_0, epoch)

        writer.add_scalar('Loss_deepsc_test', loss_deepsc_test, epoch)
        # writer.add_scalar('Loss_deepsc_burst_test', loss_deepsc_burst_test, epoch)
        writer.add_scalar('Loss_normal_test', loss_normal_test, epoch)
        # writer.add_scalar('Loss_bleu_test', loss_bleu_test, epoch)
        writer.add_scalar('eve_mac_0_test', eve_mac_0_test, epoch)
        writer.add_scalar('eve_mac_1_test', eve_mac_1_test, epoch)
        writer.add_scalar('tamper_0_test', tamper_0_test, epoch)
        writer.add_scalar('tamper_1_test', tamper_1_test, epoch)
        # writer.add_scalar('key_0_test', key_0_test, epoch)

