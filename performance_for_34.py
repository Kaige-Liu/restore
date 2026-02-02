import os
import argparse
import time
import json
import torch
import random
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from models.transceiver import DeepSC, Key_net, Attacker, MAC, CAEM_Fig2_SNR_1D, FeatureMapSelectionModule_SNR_AllC, \
    VerificationDiscriminatorLN, DiffusionSchedule, ConditionalDenoiser
from utlis.tools import SNR_to_noise, SeqtoText, BleuScore
from utlis.trainer_for_34 import train_step, val_step, train_mi, greedy_decode, initNetParams
from dataset.dataloader import return_iter, return_iter_10, return_iter_eve
from models.transceiver import DeepSC, Key_net, Attacker, MAC, KnowledgeBase, KB_Mapping
from models.mutual_info import Mine
from tqdm import tqdm

parser = argparse.ArgumentParser()
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
parser.add_argument('--epochs', default=1, type=int)

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


# 需要对解码数据进行BLEU分数计算
# 输入是解码结果、原文、以及
def performance(schedule, cdmodel, args, SNR, deepsc, alice_bob_mac, key_ab, eve, Alice_KB, Bob_KB, Eve_KB, Alice_mapping, Bob_mapping, Eve_mapping):
    # similarity = Similarity(args.bert_config_path, args.bert_checkpoint_path, args.bert_dict_path)
    bleu_score_1gram = BleuScore(1, 0, 0, 0)

    test_iterator = return_iter_10(args, 'train')
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
    args = parser.parse_args()
    SNR = [20]

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

    checkpoint = torch.load(r'/root/autodl-tmp/restore/checkpoints/checkpoint_109.pth')
    checkpoint_34 = torch.load(r'/root/autodl-tmp/restore/checkpoints/34/2026-02-02-06_37_46/checkpoint_399_0.8405_0.0004.pth')  # 12部分的那三个网络
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
    cdmodel_state_dict = checkpoint_34['cdmodel']


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
    cdmodel.load_state_dict(cdmodel_state_dict)

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
    cdmodel = cdmodel.to(device)

    bleu_score = performance(schedule, cdmodel, args, SNR, deepsc, alice_bob_mac, key_ab, eve, Alice_KB, Bob_KB, Eve_KB, Alice_mapping, Bob_mapping, Eve_mapping)
    print(bleu_score)

