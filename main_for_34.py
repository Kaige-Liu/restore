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

from models.transceiver import Key_net, Attacker, MAC, CAEM_Fig2_SNR_1D, FeatureMapSelectionModule_SNR_AllC, \
    VerificationDiscriminatorLN, DiffusionSchedule, ConditionalDenoiser
from utlis.tools import SNR_to_noise, SeqtoText, BleuScore
from utlis.trainer_for_34 import initNetParams, train_step, val_step, train_mi, greedy_decode
from dataset.dataloader import return_iter, return_iter_10, return_iter_eve
from models.transceiver import DeepSC, KnowledgeBase, KB_Mapping
from models.mutual_info import Mine
from tqdm import tqdm

parser = argparse.ArgumentParser()  # 创建一个命令行参数解释器
parser.add_argument('--vocab-file', default='./data/vocab.json', type=str)
parser.add_argument('--vocab_path', default='./data/vocab.json', type=str)
parser.add_argument('--checkpoint-path', default='./checkpoints/34', type=str)
parser.add_argument('--channel', default='Rayleigh', type=str, help='Please choose AWGN, Rayleigh, and Rician')
parser.add_argument('--MAX-LENGTH', default=30, type=int)
parser.add_argument('--MIN-LENGTH', default=4, type=int)
parser.add_argument('--d-model', default=128, type=int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num-layers', default=4, type=int)
parser.add_argument('--num-heads', default=8, type=int)
parser.add_argument('--batch-size', default=88, type=int)  # 这里控制的是每次拿(从数据集中读取)多少张牌(个句子)
parser.add_argument('--epochs', default=600, type=int)

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

def train(schedule, cdmodel, epoch, args, net, alice_bob_mac, key_ab, eve, Alice_KB, Bob_KB, Eve_KB, Alice_mapping, Bob_mapping, Eve_mapping, mi_net=None):  # 当前训练的轮数，命令行参数，模型，互信息网络（默认是None，也就是不训互信息网络）
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
    total_eps = 0

    noise_std = np.random.uniform(SNR_to_noise(3), SNR_to_noise(10), size=(1))  # 这里其实没用 但是保留吧
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
            loss_eps = train_step(schedule, cdmodel,
                                    args, epoch, batch, net,
                                    alice_bob_mac, key_ab, eve,
                                    Alice_KB, Bob_KB, Eve_KB,
                                    Alice_mapping, Bob_mapping,
                                    Eve_mapping,
                                    sents, sents, sents_eve,
                                    noise_std[0], pad_idx,
                                    optimizer_joint,
                                    args.channel)

            total_eps += loss_eps

        batch += 1

    print("================train======================")
    print("epoch: ", epoch)
    print("loss_eps: ", total_eps / len(train_iterator))
    print("================train======================")

    return total_eps / len(train_iterator)

def validate(schedule, cdmodel, epoch, args, net, alice_bob_mac, key_ab, eve, Alice_KB, Bob_KB, Eve_KB, Alice_mapping, Bob_mapping, Eve_mapping):  # epoch表示正在验证的是第几轮
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
    cdmodel.eval()

    pbar = tqdm(test_iterator)
    pbar_eve = tqdm(test_iterator_eve, leave=False)
    pbar_eve_iter = iter(test_iterator_eve)
    batch = 0
    total_eps = 0

    with torch.no_grad():  # 不需要计算梯度，看牌前的常规操作，不用管
        for sents in pbar:  # 其实就是for data in dataloader,这是[128, 31]的张量
            sents = sents.to(device)  # 将数据放到GPU上
            try:
                sents_eve = next(pbar_eve_iter).to(device)  # 取一个 batch 的攻击者数据
            except StopIteration:
                # 如果 pbar_eve 已经遍历完，重新开始
                pbar_eve_iter = iter(pbar_eve)
                sents_eve = next(pbar_eve_iter).to(device)

            loss_eps = val_step(schedule, cdmodel,
                                            args, batch, net, alice_bob_mac, key_ab, eve, Alice_KB, Bob_KB, Eve_KB, Alice_mapping, Bob_mapping, Eve_mapping,
                                                                                          sents, sents, sents_eve, 0.1, pad_idx, args.channel)

            total_eps += loss_eps
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
    print("loss_eps_test: ", total_eps / len(test_iterator))
    print("================validate======================")

    return total_eps / len(test_iterator)



def performance(schedule, cdmodel, args, SNR, deepsc, alice_bob_mac, key_ab, eve, Alice_KB, Bob_KB, Eve_KB, Alice_mapping, Bob_mapping, Eve_mapping):
    # similarity = Similarity(args.bert_config_path, args.bert_checkpoint_path, args.bert_dict_path)
    bleu_score_1gram = BleuScore(1, 0, 0, 0)

    test_iterator = return_iter_10(args, 'test')
    test_iterator_eve = return_iter_eve(args, 'test')
    iter_eve = iter(test_iterator_eve)

    StoT = SeqtoText(token_to_idx, end_idx)
    score = []

    deepsc.eval()
    key_ab.eval()
    alice_bob_mac.eval()
    eve.eval()
    Alice_KB.eval()
    Bob_KB.eval()
    Eve_KB.eval()
    Alice_mapping.eval()
    Bob_mapping.eval()
    Eve_mapping.eval()
    cdmodel.eval()


    with torch.no_grad():
        for epoch in range(1):
            Tx_word = []
            Rx_word = []

            for snr in tqdm(SNR):  # 对每个信噪比 所有的数据
                word = []
                target_word = []
                noise_std = SNR_to_noise(snr)

                for sents in test_iterator:
                    sents = sents.to(device)
                    try:
                        sents_eve = next(iter_eve).to(device)
                    except:
                        iter_eve = iter(test_iterator_eve)
                        sents_eve = next(iter_eve).to(device)
                    # src = batch.src.transpose(0, 1)[:1]
                    target = sents

                    out = greedy_decode(schedule, cdmodel, args, deepsc, alice_bob_mac, key_ab, eve, Alice_KB, Bob_KB, Eve_KB, Alice_mapping, Bob_mapping, Eve_mapping,
                                                                                        sents, sents_eve,
                                                                                        noise_std, args.MAX_LENGTH,
                                                                                        pad_idx,
                                                                                        start_idx, args.channel)

                    # 下面是将数字句子转换为字符串句子
                    sentences = out.cpu().numpy().tolist()  # list bs长度 每个元素是一个句子，句子也是一个List,用数字表示
                    result_string = list(map(StoT.sequence_to_text, sentences))  # list 每个元素是一个字符串句子
                    word = word + result_string  # list 数据集的所有预测句子全加进来
                    # print(result_string)

                    target_sent = target.cpu().numpy().tolist()
                    result_string = list(map(StoT.sequence_to_text, target_sent))
                    target_word = target_word + result_string  # list 数据集的所有原始句子全加进来
                    # print(result_string, end='\n\n')

                Tx_word.append(word)  # list 长度7 每个元素是list 即Tx_word[0][0]是第一个信噪比下的第一个字符串句子
                Rx_word.append(target_word)

            bleu_score = []
            sim_score = []

            for sent1, sent2 in zip(Tx_word, Rx_word):  # sent1是第一个信噪比下的所有句子
                # 1-gram
                bleu_score.append(bleu_score_1gram.compute_score(sent1, sent2))  # 每个元素是list,bleu_score[0][0]是第一个信噪比下的第一个句子的BLEU分数,这样计算了所有的句子
                # sim_score.append(similarity.compute_similarity(sent1, sent2)) # 7*num_sent
            bleu_score = np.array(bleu_score)  # 尺寸为7 * 句子数
            bleu_score = np.mean(bleu_score, axis=1)  # 每个信噪比下的所有句子的平均BLEU分数
            score.append(bleu_score)  # 存储到当前epoch中

            # sim_score = np.array(sim_score)
            # sim_score = np.mean(sim_score, axis=1)
            # score2.append(sim_score)

    score1 = np.mean(np.array(score), axis=0)  # 每个信噪比下的所有句子的平均BLEU分数(按照epoch平均)
    # score2 = np.mean(np.array(score2), axis=0)

    # return score1, zheng_mac_score, eve_mac_0_score, tamper_0_score, key_0_score  # , score2
    return score1





if __name__ == '__main__':
    now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    # if not os.path.isdir("./checkpoints/deepsc_hiding/" + now):
    #     os.mkdir("./checkpoints/deepsc_hiding/" + now)
    os.mkdir("./checkpoints/34/" + now)
    writer = SummaryWriter(log_dir="./logs/34/" + now)

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

    Alice_KB = KnowledgeBase().to(device)
    Bob_KB = KnowledgeBase().to(device)
    Eve_KB = KnowledgeBase().to(device)

    Alice_mapping = KB_Mapping().to(device)
    Bob_mapping = KB_Mapping().to(device)
    Eve_mapping = KB_Mapping().to(device)

    T = 200
    schedule = DiffusionSchedule(T=T, device=device)
    cdmodel = ConditionalDenoiser(
        feature_dim=128,
        model_dim=256,
        num_layers=4,
        num_heads=8,
    ).to(device)


    checkpoint = torch.load(r'/root/autodl-tmp/restore/checkpoints/checkpoint_109.pth')  # 之前三个检测率都最好
    model_state_dict = checkpoint['deepsc']
    alice_bob_mac_state_dict = checkpoint['alice_bob_mac']
    key_state_dict = checkpoint['key_ab']
    eve_state_dict = checkpoint['eve']
    Alice_KB_state_dict = checkpoint['Alice_KB']
    Bob_KB_state_dict = checkpoint['Bob_KB']
    Eve_KB_state_dict = checkpoint['Eve_KB']
    Alice_mapping_state_dict = checkpoint['Alice_mapping']
    Bob_mapping_state_dict = checkpoint['Bob_mapping']
    Eve_mapping_state_dict = checkpoint['Eve_mapping']

    deepsc.load_state_dict(model_state_dict)
    alice_bob_mac.load_state_dict(alice_bob_mac_state_dict)
    key_ab.load_state_dict(key_state_dict)
    eve.load_state_dict(eve_state_dict)
    Alice_KB.load_state_dict(Alice_KB_state_dict)
    Bob_KB.load_state_dict(Bob_KB_state_dict)
    Eve_KB.load_state_dict(Eve_KB_state_dict)
    Alice_mapping.load_state_dict(Alice_mapping_state_dict)
    Bob_mapping.load_state_dict(Bob_mapping_state_dict)
    Eve_mapping.load_state_dict(Eve_mapping_state_dict)

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

    initNetParams(cdmodel)

    optimizer = torch.optim.Adam(deepsc.parameters(),
                                 lr=1e-5, betas=(0.9, 0.98), eps=1e-8, weight_decay=5e-4)
    mi_opt = torch.optim.Adam(mi_net.parameters(), lr=1e-3)
    # opt = NoamOpt(args.d_model, 1, 4000, optimizer)

    # 联合训练的优化器
    optimizer_joint = torch.optim.Adam(
        list(cdmodel.parameters()),
        lr=1e-4, betas=(0.9, 0.98), eps=1e-8, weight_decay=5e-4)
    # 模型的设置可能有点问题
    # 只优化deepsc的优化器
    # optimizer_joint = torch.optim.Adam(deepsc.parameters(),
    #                              lr=1e-4, betas=(0.9, 0.98), eps=1e-8, weight_decay=5e-4)

    # 下面就是训练deepsc模型
    SNR = [0]  # 其实根本不用 因为都不过信道
    for epoch in range(args.epochs):  # 默认训练完整的数据集80轮
        start = time.time()  # 记录每轮开始时间（没用到）
        record_loss = 1000  # 其实是loss，设置的大一点

        loss_eps = train(schedule, cdmodel, epoch, args, deepsc, alice_bob_mac, key_ab, eve, Alice_KB, Bob_KB, Eve_KB, Alice_mapping, Bob_mapping, Eve_mapping)  # 单独预训练deepsc

        loss_eps_test = validate(schedule, cdmodel, epoch, args, deepsc, alice_bob_mac, key_ab, eve, Alice_KB, Bob_KB, Eve_KB, Alice_mapping, Bob_mapping, Eve_mapping)

        bleu_score = performance(schedule, cdmodel, args, SNR, deepsc, alice_bob_mac, key_ab, eve, Alice_KB, Bob_KB, Eve_KB, Alice_mapping, Bob_mapping, Eve_mapping)
        print("bleu_score: ", bleu_score)

        if loss_eps_test < record_loss:  # 如果验证的loss小于之前的loss（性能更好了）
            checkpoint = {
                "cdmodel": cdmodel.state_dict(),
            }
            torch.save(checkpoint, './checkpoints/34/' + now + '/checkpoint_{}'.format(epoch) + '_{}'.format(str(loss_eps_test)[:6]) + '_{}.pth'.format(str(bleu_score)[1:7]))  # 保存模型
            record_loss = loss_eps_test  # 更新最小的准确率

        writer.add_scalar('Loss_eps', loss_eps, epoch)
        writer.add_scalar('Loss_eps_test', loss_eps_test, epoch)
