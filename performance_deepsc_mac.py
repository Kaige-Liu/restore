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
from utlis.tools import SNR_to_noise, SeqtoText, BleuScore
from utlis.trainer import train_step, val_step, train_mi, greedy_decode
from dataset.dataloader import return_iter, return_iter_10
from models.transceiver import DeepSC
from models.mutual_info import Mine
from tqdm import tqdm

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


def performance(args, SNR, deepsc, alice_bob_mac, key_ab, eve):
    # similarity = Similarity(args.bert_config_path, args.bert_checkpoint_path, args.bert_dict_path)
    bleu_score_1gram = BleuScore(1, 0, 0, 0)

    test_iterator = return_iter(args, 'test')

    StoT = SeqtoText(token_to_idx, end_idx)
    score = []
    score2 = []
    cos_list = []
    zheng_mac_list = []
    eve_mac_0_list = []
    tamper_0_list = []
    key_0_list = []

    deepsc.eval()
    key_ab.eval()
    alice_bob_mac.eval()
    eve.eval()
    with torch.no_grad():
        for epoch in range(args.epochs):
            Tx_word = []
            Rx_word = []
            cos_list_tmp = []
            zheng_mac_list_tmp = []
            eve_mac_0_list_tmp = []
            tamper_0_list_tmp = []
            key_0_list_tmp = []

            for snr in tqdm(SNR):  # 对每个信噪比 所有的数据
                word = []
                target_word = []
                noise_std = SNR_to_noise(snr)

                total_cos = 0
                total_zheng_mac = 0
                total_eve_mac_0 = 0
                total_tamper_0 = 0
                total_key_0 = 0
                for sents in test_iterator:
                    sents = sents.to(device)
                    # src = batch.src.transpose(0, 1)[:1]
                    target = sents

                    out, zheng_mac_accuracy, eve_mac_0, tamper_0, key_0 = greedy_decode(args, deepsc, alice_bob_mac, key_ab, eve, sents, noise_std, args.MAX_LENGTH, pad_idx,
                                        start_idx, args.channel)
                    # out = greedy_decode(args, deepsc, alice_bob_mac, key_ab, eve, sents, noise_std, args.MAX_LENGTH, pad_idx,
                    #                     start_idx, args.channel)
                    total_zheng_mac += zheng_mac_accuracy
                    total_eve_mac_0 += eve_mac_0
                    total_tamper_0 += tamper_0
                    total_key_0 += key_0

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
                # average_cos = total_cos / len(test_iterator)
                average_zheng_mac = total_zheng_mac / len(test_iterator)  # 当前信噪比下的平均准确率(一个数)
                average_eve_mac_0 = total_eve_mac_0 / len(test_iterator)
                average_tamper_0 = total_tamper_0 / len(test_iterator)
                average_key_0 = total_key_0 / len(test_iterator)

                # cos_list_tmp.append(average_cos)
                zheng_mac_list_tmp.append(average_zheng_mac)
                eve_mac_0_list_tmp.append(average_eve_mac_0)
                tamper_0_list_tmp.append(average_tamper_0)
                key_0_list_tmp.append(average_key_0)

            bleu_score = []
            sim_score = []
            # cos_list.append(cos_list_tmp)
            zheng_mac_list.append(zheng_mac_list_tmp)
            eve_mac_0_list.append(eve_mac_0_list_tmp)
            tamper_0_list.append(tamper_0_list_tmp)
            key_0_list.append(key_0_list_tmp)

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
    # cos_score = np.mean(np.array(cos_list), axis=0)
    zheng_mac_score = np.mean(np.array(zheng_mac_list), axis=0)
    eve_mac_0_score = np.mean(np.array(eve_mac_0_list), axis=0)
    tamper_0_score = np.mean(np.array(tamper_0_list), axis=0)
    key_0_score = np.mean(np.array(key_0_list), axis=0)

    return score1, zheng_mac_score, eve_mac_0_score, tamper_0_score, key_0_score  # , score2
    # return score1

if __name__ == '__main__':
    args = parser.parse_args()
    SNR = [0, 3, 6, 9, 12, 15, 18]
    # SNR = [15, 18]

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


    """ define optimizer and loss function """
    deepsc = DeepSC(args.num_layers, num_vocab, num_vocab,
                        num_vocab, num_vocab, args.d_model, args.num_heads,
                        args.dff, 0.1).to(device)
    alice_bob_mac = MAC().to(device)
    key_ab = Key_net(args).to(device)
    eve = Attacker(args).to(device)

    # checkpoint = torch.load('C:\d\code\deepsc_mac\checkpoints\deepsc_mac\\2024-09-23-11_16_42\checkpoint_3013.pth')
    # checkpoint = torch.load('C:\d\code\deepsc_mac\checkpoints\deepsc_mac\\2024-09-27-11_30_12\checkpoint_373.pth')
    # checkpoint = torch.load('C:\d\code\deepsc_mac\checkpoints\deepsc_mac\\2024-09-29-17_09_37\checkpoint_576.pth')
    # checkpoint = torch.load('C:\d\code\deepsc_mac\checkpoints\deepsc_mac\\2024-10-09-10_39_50\checkpoint_99.pth')
    # checkpoint = torch.load('C:\d\code\deepsc_mac\checkpoints\deepsc_mac\\2024-10-09-20_19_15\checkpoint_306.pth')
    # checkpoint = torch.load('C:\d\code\deepsc_mac\checkpoints\deepsc_mac\\2024-10-10-11_17_52\checkpoint_432.pth')
    # checkpoint = torch.load('C:\d\code\deepsc_mac\checkpoints\deepsc_mac\\2024-10-10-16_54_17\checkpoint_501.pth')
    # checkpoint = torch.load('C:\d\code\deepsc_mac\checkpoints\deepsc_mac\\2024-10-14-17_28_10\checkpoint_250.pth')
    checkpoint = torch.load('C:\d\code\deepsc_mac\checkpoints\deepsc_mac\\2024-10-17-10_30_58\checkpoint_243.pth')
    model_state_dict = checkpoint['deepsc']
    alice_bob_mac_state_dict = checkpoint['alice_bob_mac']
    key_state_dict = checkpoint['key_ab']
    eve_state_dict = checkpoint['eve']
    deepsc.load_state_dict(model_state_dict)
    alice_bob_mac.load_state_dict(alice_bob_mac_state_dict)
    key_ab.load_state_dict(key_state_dict)
    eve.load_state_dict(eve_state_dict)
    deepsc = deepsc.to(device)
    alice_bob_mac = alice_bob_mac.to(device)
    key_ab = key_ab.to(device)
    eve = eve.to(device)

    bleu_score, zheng_mac_score, eve_mac_0_score, tamper_0_score, key_0_score = performance(args, SNR, deepsc, alice_bob_mac, key_ab, eve)
    # bleu_score = performance(args, SNR, deepsc, alice_bob_mac, key_ab, eve)
    print("bleu_score:")
    print(bleu_score)  # 输出的结果是，七个信噪比下的BLEU平均分数，每个平均分数是测试集中所有的句子的平均分数
    print("正向检测准确率：")
    print(zheng_mac_score)
    print("1类攻击检测准确率:")
    print(eve_mac_0_score)
    print("2类攻击检测准确率:")
    print(tamper_0_score)
    print("错误密钥检测准确率：")
    print(key_0_score)
    # similarity.compute_similarity(sent1, real)

