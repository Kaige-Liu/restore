# !usr/bin/env python
# -*- coding:utf-8 _*-
import os
import json
import torch
import argparse
import numpy as np
from dataset.dataloader import return_iter, return_iter_10
from deepsc_hiding import H_DeepSC
from utlis.tools import BleuScore, SNR_to_noise, SeqtoText
from utlis.trainer_new import greedy_decode
from tqdm import tqdm
from sklearn.preprocessing import normalize
from w3lib.html import remove_tags

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='.\\data\\train_data.pkl', type=str)
parser.add_argument('--vocab-file', default='.\\data\\vocab.json', type=str)
parser.add_argument('--checkpoint-path', default='checkpoints\\deepsc_hiding', type=str)
parser.add_argument('--channel', default='Rayleigh', type=str)
parser.add_argument('--MAX-LENGTH', default=30, type=int)
parser.add_argument('--MIN-LENGTH', default=4, type=int)
parser.add_argument('--d-model', default=128, type=int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num-layers', default=4, type=int)
parser.add_argument('--num-heads', default=8, type=int)
parser.add_argument('--batch-size', default=10, type=int)  # 训练的时候就是32
parser.add_argument('--epochs', default=2, type=int)
parser.add_argument('--bert-config-path', default='bert/cased_L-12_H-768_A-12/bert_config.json', type=str)
parser.add_argument('--bert-checkpoint-path', default='bert/cased_L-12_H-768_A-12/bert_model.ckpt', type=str)
parser.add_argument('--bert-dict-path', default='bert/cased_L-12_H-768_A-12/vocab.txt', type=str)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def performance(args, SNR, net):
    # similarity = Similarity(args.bert_config_path, args.bert_checkpoint_path, args.bert_dict_path)
    bleu_score_1gram = BleuScore(1, 0, 0, 0)  # 计算的是BLEU-1分数

    hiding_test_iterator = return_iter_10(args, 'test')
    cover_test_iterator = return_iter(args, 'test')

    StoT = SeqtoText(token_to_idx, end_idx)
    hiding_score = []
    cover_score = []
    score2 = []
    net.eval()
    with torch.no_grad():
        for epoch in range(args.epochs):  # 一共就2次
            hiding_Tx_word = []  # 预测的
            hiding_Rx_word = []  # 正确的
            cover_Tx_word = []
            cover_Rx_word = []

            for snr in tqdm(SNR):
                hiding_word = []
                hiding_target_word = []
                cover_word = []
                cover_target_word = []

                noise_std = SNR_to_noise(snr)
                cover_data_iter = iter(cover_test_iterator)

                for hiding_data in hiding_test_iterator:
                    hiding_data = hiding_data.to(device)
                    cover_data = next(cover_data_iter)
                    cover_data = cover_data.to(device)

                    # src = batch.src.transpose(0, 1)[:1]
                    hiding_target = hiding_data
                    cover_target = cover_data

                    # 这里应该返回两个结果，分别是hidden_data和cover_data的预测结果
                    hiding_out, cover_out = greedy_decode(net, hiding_data, cover_data, noise_std, args.MAX_LENGTH, pad_idx,
                                        start_idx, args.channel)

                    hiding_sentences_list = hiding_out.cpu().numpy().tolist()  # list bs长度 每个元素是一个句子，句子也是一个List,用数字表示
                    hiding_result_string = list(map(StoT.sequence_to_text, hiding_sentences_list))  # list 每个元素是一个字符串句子
                    hiding_word = hiding_word + hiding_result_string  # list 数据集的所有预测句子全加进来
                    # print(len(hiding_result_string[0].split()))
                    print(hiding_result_string[0])

                    hiding_target_list = hiding_target.cpu().numpy().tolist()
                    hiding_result_string = list(map(StoT.sequence_to_text, hiding_target_list))
                    hiding_target_word = hiding_target_word + hiding_result_string  # list 数据集的所有原始句子全加进来
                    # print(len(hiding_result_string[0].split()))
                    print(hiding_result_string[0], end='\n\n')


                    cover_sentences_list = cover_out.cpu().numpy().tolist()  # list bs长度 每个元素是一个句子，句子也是一个List,用数字表示
                    cover_result_string = list(map(StoT.sequence_to_text, cover_sentences_list))  # list 每个元素是一个字符串句子
                    cover_word = cover_word + cover_result_string  # list 数据集的所有预测句子全加进来
                    # print(len(cover_sentences_list[0]))
                    print(cover_result_string[0])

                    cover_target_list = cover_target.cpu().numpy().tolist()
                    cover_result_string = list(map(StoT.sequence_to_text, cover_target_list))
                    cover_target_word = cover_target_word + cover_result_string  # list 数据集的所有原始句子全加进来
                    print(cover_result_string[0], end='\n\n')


                hiding_Tx_word.append(hiding_word)  # list 长度7 每个元素是list 即Tx_word[0][0]是第一个信噪比下的第一个字符串句子
                hiding_Rx_word.append(hiding_target_word)

                cover_Tx_word.append(cover_word)
                cover_Rx_word.append(cover_target_word)

            hiding_bleu_score = []
            cover_bleu_score = []
            sim_score = []
            # 计算hiding_data的BLEU分数
            for sent1, sent2 in zip(hiding_Tx_word, hiding_Rx_word):  # sent1是第一个信噪比下的所有句子
                # 按照sent2[i]的单词长度截取sent1[i]的单词长度
                # 注意是单词长度，不是字母
                for i in range(len(sent2)):
                    word1 = sent1[i].split()  # 类型是list
                    word2 = sent2[i].split()
                    len_word = len(word2)
                    word1 = word1[:len_word]
                    sent1[i] = ' '.join(word1)  # 类型是str



                # 1-gram
                hiding_bleu_score.append(bleu_score_1gram.compute_score(sent1, sent2))  # 每个元素是list,bleu_score[0][0]是第一个信噪比下的第一个句子的BLEU分数,这样计算了所有的句子
                # sim_score.append(similarity.compute_similarity(sent1, sent2)) # 7*num_sent
            hiding_bleu_score = np.array(hiding_bleu_score)  # 尺寸为7 * 句子数
            print(hiding_bleu_score.shape)
            hiding_bleu_score = np.mean(hiding_bleu_score, axis=1)  # 每个信噪比下的所有句子的平均BLEU分数
            hiding_score.append(hiding_bleu_score)  # 存储到当前epoch中

            # 计算cover_data的BLEU分数
            for sent1, sent2 in zip(cover_Tx_word, cover_Rx_word):
                for i in range(len(sent2)):
                    word1 = sent1[i].split()  # 类型是list
                    word2 = sent2[i].split()
                    len_word = len(word2)
                    word1 = word1[:len_word]
                    sent1[i] = ' '.join(word1)  # 类型是str
                cover_bleu_score.append(bleu_score_1gram.compute_score(sent1, sent2))
            cover_bleu_score = np.array(cover_bleu_score)
            print(cover_bleu_score.shape)
            cover_bleu_score = np.mean(cover_bleu_score, axis=1)
            cover_score.append(cover_bleu_score)

            # sim_score = np.array(sim_score)
            # sim_score = np.mean(sim_score, axis=1)
            # score2.append(sim_score)

    hiding_score1 = np.mean(np.array(hiding_score), axis=0)  # 每个信噪比下的所有句子的平均BLEU分数(按照epoch平均)
    cover_score1 = np.mean(np.array(cover_score), axis=0)
    # score2 = np.mean(np.array(score2), axis=0)

    return hiding_score1, cover_score1  # , score2


if __name__ == '__main__':
    args = parser.parse_args()
    SNR = [0, 3, 6, 9, 12, 15, 18]
    # SNR = [15]
    N = 5

    args.vocab_file = args.vocab_file
    vocab = json.load(open(args.vocab_file, 'rb'))
    token_to_idx = vocab['token_to_idx']
    idx_to_token = dict(zip(token_to_idx.values(), token_to_idx.keys()))
    num_vocab = len(token_to_idx)
    pad_idx = token_to_idx["<PAD>"]
    start_idx = token_to_idx["<START>"]
    end_idx = token_to_idx["<END>"]

    """ define optimizer and loss function """
    # deepsc = DeepSC(args.num_layers, num_vocab, num_vocab,
    #                 num_vocab, num_vocab, args.d_model, args.num_heads,
    #                 args.dff, 0.1).to(device)

    # model_paths = []
    # for fn in os.listdir(args.checkpoint_path):
    #     if not fn.endswith('.pth'): continue
    #     idx = int(os.path.splitext(fn)[0].split('_')[-1])  # read the idx of image
    #     model_paths.append((os.path.join(args.checkpoint_path, fn), idx))
    #
    # model_paths.sort(key=lambda x: x[1])  # sort the image by the idx
    #
    # model_path, _ = model_paths[-1]
    # checkpoint = torch.load(model_path)
    # deepsc.load_state_dict(checkpoint)
    # print('model load!')
    H_deepsc = H_DeepSC(N, args.num_layers, num_vocab, num_vocab,
                        num_vocab, num_vocab, args.d_model, args.num_heads,
                        args.dff, 0.1).to(device)

    checkpoint = torch.load('F:\checkpoints\deepsc_hiding\\2024-06-06-21_16_04\checkpoint_498.pth')
    model_state_dict = checkpoint['net']
    H_deepsc.load_state_dict(model_state_dict)
    H_deepsc = H_deepsc.to(device)

    hiding_bleu_score, cover_bleu_score = performance(args, SNR, H_deepsc)
    print(hiding_bleu_score)  # 输出的结果是，七个信噪比下的BLEU平均分数，每个平均分数是测试集中所有的句子的平均分数
    print(cover_bleu_score)
    # similarity.compute_similarity(sent1, real)
