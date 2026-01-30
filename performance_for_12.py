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
    VerificationDiscriminatorLN
from utlis.tools import SNR_to_noise, SeqtoText, BleuScore
from utlis.trainer_for_next_work import train_step, val_step, train_mi, greedy_decode, initNetParams
from dataset.dataloader import return_iter, return_iter_10, return_iter_eve
from models.transceiver import DeepSC, Key_net, Attacker, MAC, KnowledgeBase, KB_Mapping
from models.mutual_info import Mine
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--vocab-file', default='./data/vocab.json', type=str)
parser.add_argument('--vocab_path', default='./data/vocab.json', type=str)
parser.add_argument('--checkpoint-path', default='./checkpoints/12', type=str)
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
def performance(CAEM_with_SNR, fms, alice_verifier, args, SNR, deepsc, alice_bob_mac, key_ab, eve, Alice_KB, Bob_KB, Eve_KB, Alice_mapping, Bob_mapping, Eve_mapping):
    test_iterator = return_iter(args, 'test')
    test_iterator_eve = return_iter_eve(args, 'test')
    iter_eve = iter(test_iterator_eve)

    StoT = SeqtoText(token_to_idx, end_idx)
    score = []
    alice_list = []
    eve_list = []

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
    CAEM_with_SNR.eval()
    fms.eval()
    alice_verifier.eval()


    with torch.no_grad():
        for epoch in range(args.epochs):
            alice_list_tmp = []
            eve_list_tmp = []

            for snr in tqdm(SNR):  # 对每个信噪比 所有的数据
                # snr就是一个数
                noise_std = SNR_to_noise(snr)

                total_alice = 0
                total_eve = 0
                for sents in test_iterator:
                    sents = sents.to(device)
                    try:
                        sents_eve = next(iter_eve).to(device)
                    except:
                        iter_eve = iter(test_iterator_eve)
                        sents_eve = next(iter_eve).to(device)
                    alice_1, eve_0 = greedy_decode(CAEM_with_SNR, fms, alice_verifier, args, deepsc, alice_bob_mac, key_ab, eve, Alice_KB, Bob_KB, Eve_KB, Alice_mapping, Bob_mapping, Eve_mapping,
                                                                                        sents, sents_eve,
                                                                                        noise_std, args.MAX_LENGTH,
                                                                                        pad_idx,
                                                                                        start_idx, args.channel)
                    total_alice += alice_1
                    total_eve += eve_0

                average_alice = total_alice / len(test_iterator)  # 当前信噪比下的平均准确率(一个数)
                average_eve = total_eve / len(test_iterator)

                alice_list_tmp.append(average_alice)
                eve_list_tmp.append(average_eve)

            alice_list.append(alice_list_tmp)
            eve_list.append(eve_list_tmp)

    alice_score = np.mean(np.array(alice_list), axis=0)
    eve_score = np.mean(np.array(eve_list), axis=0)

    return alice_score, eve_score




if __name__ == '__main__':
    args = parser.parse_args()
    SNR = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

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

    CAEM_with_SNR = CAEM_Fig2_SNR_1D(C_in=31, C_out=16, use_resnet=True).to(device)  # 加入SNR考量的映射 alice和bob共享
    fms = FeatureMapSelectionModule_SNR_AllC(C=16, hidden=64).to(device)  # 对上面的映射进行特征选择 只有bob用 16就是上面的输出通道数 注意这里筛选完也是16通道 只不过有的被置零了
    alice_verifier = VerificationDiscriminatorLN(C=16, L=128, output_logits=True).to(device)  # alice的验证器 128是特征长度

    checkpoint = torch.load(r'/root/autodl-tmp/for_work_12/checkpoints/checkpoint_109.pth')
    checkpoint_12 = torch.load(r'/root/autodl-tmp/for_work_12/checkpoints/12/2026-01-29-17_55_16/checkpoint_399_0.9968_0.9851.pth')  # 12部分的那三个网络
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
    CAEM_with_SNR_state_dict = checkpoint_12['CAEM_with_SNR']
    fms_state_dict = checkpoint_12['fms']
    alice_verifier_state_dict = checkpoint_12['alice_verifier']


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
    CAEM_with_SNR.load_state_dict(CAEM_with_SNR_state_dict)
    fms.load_state_dict(fms_state_dict)
    alice_verifier.load_state_dict(alice_verifier_state_dict)

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
    CAEM_with_SNR = CAEM_with_SNR.to(device)
    fms = fms.to(device)
    alice_verifier = alice_verifier.to(device)

    alice_score, eve_score = performance(CAEM_with_SNR, fms, alice_verifier, args, SNR, deepsc, alice_bob_mac, key_ab, eve, Alice_KB, Bob_KB, Eve_KB, Alice_mapping, Bob_mapping, Eve_mapping)
    print("alice检测准确率：")
    print(alice_score)
    print("eve检测准确率：")
    print(eve_score)

