import os
import argparse
import time
import json
import torch
import random
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from models.diffusion_dit import FeatureRestorationDiT
from models.tranceiver_for_34 import SNR_Net
from models.transceiver import DeepSC, Key_net, Attacker, MAC, CAEM_Fig2_SNR_1D, FeatureMapSelectionModule_SNR_AllC, \
    VerificationDiscriminatorLN, DiffusionSchedule, ConditionalDenoiser
from utlis.tools import SNR_to_noise, SeqtoText, BleuScore
from utlis.trainer_for_34 import train_step, val_step, greedy_decode, initNetParams, DDIMScheduler
from dataset.dataloader import return_iter, return_iter_10, return_iter_eve
from models.tranceiver_for_34 import DeepSC, Key_net, Attacker, MAC, KnowledgeBase, KB_Mapping
from models.mutual_info import Mine
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--vocab-file', default='./data/vocab.json', type=str)
parser.add_argument('--vocab_path', default='./data/vocab.json', type=str)
parser.add_argument('--checkpoint-path', default='./checkpoints/34', type=str)
parser.add_argument('--channel', default='AWGN', type=str, help='Please choose AWGN, Rayleigh, and Rician')
parser.add_argument('--MAX-LENGTH', default=30, type=int)
parser.add_argument('--MIN-LENGTH', default=4, type=int)
parser.add_argument('--d-model', default=128, type=int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num-layers', default=4, type=int)
parser.add_argument('--num-heads', default=8, type=int)
parser.add_argument('--batch-size', default=512, type=int)  # 这里控制的是每次拿(从数据集中读取)多少张牌(个句子)
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


# 将 cdmodel 和 ddim_scheduler 加入参数列表
def performance(snr_net_alice, snr_net_bob, args, SNR, deepsc, alice_bob_mac, key_ab, Alice_KB, Bob_KB, Alice_mapping, Bob_mapping, cdmodel,
                ddim_scheduler):
    bleu_score_1gram = BleuScore(1, 0, 0, 0)
    bleu_score_2gram = BleuScore(0.5, 0.5, 0, 0)  # 主要关注这几个bleu分数
    bleu_score_4gram = BleuScore(0.25, 0.25, 0.25, 0.25)

    test_iterator = return_iter(args, 'test')

    StoT = SeqtoText(token_to_idx, end_idx)
    score = []

    deepsc.eval()
    key_ab.eval()
    alice_bob_mac.eval()
    Alice_KB.eval()
    Bob_KB.eval()
    Alice_mapping.eval()
    Bob_mapping.eval()
    cdmodel.eval()
    snr_net_alice.eval()
    snr_net_bob.eval()

    with torch.no_grad():
        for epoch in range(1):  # 测试的时候跑三次
            Tx_word = []
            Rx_word = []

            for snr in tqdm(SNR):  # 对每个信噪比 所有的数据
                word = []
                target_word = []
                noise_std = SNR_to_noise(snr)

                for sents in test_iterator:
                    sents = sents.to(device)
                    # src = batch.src.transpose(0, 1)[:1]
                    target = sents

                    # 把 cdmodel, ddim_scheduler 和 current_snr 传给 greedy_decode
                    out = greedy_decode(snr_net_alice, snr_net_bob, args, deepsc, alice_bob_mac, key_ab, Alice_KB, Bob_KB, Alice_mapping,
                                        Bob_mapping,
                                        sents,
                                        noise_std, args.MAX_LENGTH,
                                        pad_idx,
                                        start_idx, args.channel,
                                        cdmodel=cdmodel, ddim_scheduler=ddim_scheduler, current_snr=snr)
                    # 下面是将数字句子转换为字符串句子
                    sentences = out.cpu().numpy().tolist()  # list bs长度 每个元素是一个句子，句子也是一个List,用数字表示
                    result_string = list(map(StoT.sequence_to_text, sentences))  # list 每个元素是一个字符串句子
                    word = word + result_string  # list 数据集的所有预测句子全加进来

                    target_sent = target.cpu().numpy().tolist()
                    result_string = list(map(StoT.sequence_to_text, target_sent))
                    target_word = target_word + result_string  # list 数据集的所有原始句子全加进来

                Tx_word.append(word)  # list 长度7 每个元素是list 即Tx_word[0][0]是第一个信噪比下的第一个字符串句子
                Rx_word.append(target_word)

            bleu_score = []

            for sent1, sent2 in zip(Tx_word, Rx_word):  # sent1是第一个信噪比下的所有句子
                # 1-gram
                bleu_score.append(bleu_score_1gram.compute_score(sent1,
                                                                 sent2))  # 每个元素是list,bleu_score[0][0]是第一个信噪比下的第一个句子的BLEU分数,这样计算了所有的句子
                # sim_score.append(similarity.compute_similarity(sent1, sent2)) # 7*num_sent
            bleu_score = np.array(bleu_score)  # 尺寸为7 * 句子数
            bleu_score = np.mean(bleu_score, axis=1)  # 每个信噪比下的所有句子的平均BLEU分数
            score.append(bleu_score)  # 存储到当前epoch中
    for i in range(len(sent1)):
        if len(sent1[i]) >= 100 and len(sent2[i]) >= 100:
            print(sent2[i])
            print(sent1[i])
            print()
    score1 = np.mean(np.array(score), axis=0)  # 每个信噪比下的所有句子的平均BLEU分数(按照epoch平均)
    return score1




if __name__ == '__main__':
    args = parser.parse_args()
    SNR = [0]

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

    snr_net_alice = SNR_Net(args.d_model).to(device)
    snr_net_bob = SNR_Net(args.d_model).to(device)

    cdmodel = FeatureRestorationDiT(
        feature_dim=16,
        seq_len=32,
        hidden_size=256,
        depth=4,
        num_heads=8
    ).to(device)

    ddim_scheduler = DDIMScheduler(device=device)

    checkpoint = torch.load(r'/root/autodl-tmp/restore/checkpoints/checkpoint_109.pth')
    # checkpoint_34 = torch.load(
    #     r'/root/autodl-tmp/restore/checkpoints/34/2026-03-07-05_46_40/checkpoint_073_0.6073.pth')
    checkpoint_34 = torch.load(
        r'/root/autodl-tmp/restore/checkpoints/34/2026-03-07-22_30_59/checkpoint_048_0.6266.pth')
    checkpoint_deepsc_snr = torch.load(
        r'/root/autodl-tmp/restore/checkpoints/34/2026-03-07-21_28_55/checkpoint_084_0.5835.pth'  # 新保存的deepsc模型
    )
    # checkpoint_snr = torch.load(
    #     r'/root/autodl-tmp/restore/checkpoints/34/2026-03-07-21_10_48/checkpoint_045_0.5454.pth'  # snr网络
    # )
    model_state_dict = checkpoint_deepsc_snr['deepsc']
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
    snr_net_alice_state_dict = checkpoint_deepsc_snr['snr_net_alice']
    snr_net_bob_state_dict = checkpoint_deepsc_snr['snr_net_bob']

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
    snr_net_alice.load_state_dict(snr_net_alice_state_dict)
    snr_net_bob.load_state_dict(snr_net_bob_state_dict)

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
    snr_net_alice = snr_net_alice.to(device)
    snr_net_bob = snr_net_bob.to(device)

    bleu_score = performance(snr_net_alice, snr_net_bob, args, SNR, deepsc, alice_bob_mac, key_ab, Alice_KB, Bob_KB, Alice_mapping, Bob_mapping,
                             cdmodel, ddim_scheduler)

    print(bleu_score)

