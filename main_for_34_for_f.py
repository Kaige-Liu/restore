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

from models.tranceiver_for_34 import Key_net, Attacker, MAC
from utlis.tools import SNR_to_noise, SeqtoText, BleuScore

from utlis.trainer_for_34_for_f import initNetParams, train_step, val_step, greedy_decode, DDIMScheduler
from dataset.dataloader import return_iter, return_iter_10, return_iter_eve
from models.tranceiver_for_34 import DeepSC, KnowledgeBase, KB_Mapping
from models.mutual_info import Mine
from tqdm import tqdm

from models.diffusion_dit_for_f import FeatureRestorationDiT

parser = argparse.ArgumentParser()  # 创建一个命令行参数解释器
parser.add_argument('--vocab-file', default='./data/vocab.json', type=str)
parser.add_argument('--vocab_path', default='./data/vocab.json', type=str)
parser.add_argument('--checkpoint-path', default='./checkpoints/f', type=str)
parser.add_argument('--channel', default='Rayleigh', type=str, help='Please choose AWGN, Rayleigh, and Rician')
parser.add_argument('--MAX-LENGTH', default=30, type=int)
parser.add_argument('--MIN-LENGTH', default=4, type=int)
parser.add_argument('--d-model', default=128, type=int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num-layers', default=4, type=int)
parser.add_argument('--num-heads', default=8, type=int)
parser.add_argument('--batch-size', default=1024, type=int) # 大 Batch Size 有利于扩散模型收敛
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


def train(epoch, args, net, alice_bob_mac, key_ab, Alice_KB, Bob_KB, Alice_mapping, Bob_mapping, cdmodel,
          ddim_scheduler, optimizer_joint, mi_net=None):  # 当前训练的轮数，命令行参数，模型，互信息网络（默认是None，也就是不训互信息网络）
    train_iterator = return_iter(args, 'train')  # 一个dataloader类型的对象（其实就是dataloder 用法完全一样）

    pbar = tqdm(train_iterator)  # 进度条

    batch = 0
    total_loss = 0 # 变量名改成了 total_loss，因为它现在包含了 MSE 和 CE 两种 loss

    noise_std = np.random.uniform(SNR_to_noise(3), SNR_to_noise(10),
                                  size=(1))  # 这里就是训练的时候的噪声标准差，我开始是这么设置的，您根据您的思路来，看看怎么设置效果会好

    for sents in pbar:  # 每个batch的数据
        sents = sents.to(device)

        # 下面这个if完全不用管 因为绝对绝对不会用到
        if mi_net is not None:  # 完全没用
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
            # 这里的 loss 已经是联合了 CE 和 MSE 的 loss_total 了
            loss_total_batch = train_step(args, epoch, batch, net,
                                  alice_bob_mac, key_ab,
                                  Alice_KB, Bob_KB,
                                  Alice_mapping, Bob_mapping,
                                  sents, sents,
                                  noise_std[0], pad_idx,
                                  optimizer_joint,
                                  args.channel,
                                  cdmodel, ddim_scheduler)

            total_loss += loss_total_batch

        batch += 1

    print("================train======================")
    print("epoch: ", epoch)
    print("loss_total (MSE + CE): ", total_loss / len(train_iterator))
    print("================train======================")

    return total_loss / len(train_iterator)


# 将 cdmodel 和 ddim_scheduler 加入参数列表
def validate(epoch, args, net, alice_bob_mac, key_ab, Alice_KB, Bob_KB, Alice_mapping, Bob_mapping, cdmodel,
             ddim_scheduler):  # epoch表示正在验证的是第几轮
    test_iterator = return_iter(args, 'test')  # 从测试数据集中抓牌

    net.eval()
    alice_bob_mac.eval()
    key_ab.eval()
    Alice_KB.eval()
    Bob_KB.eval()
    Alice_mapping.eval()
    Bob_mapping.eval()

    pbar = tqdm(test_iterator)
    batch = 0
    total_loss = 0

    with torch.no_grad():
        for sents in pbar:
            sents = sents.to(device)

            # cdmodel 和 ddim_scheduler 传给 val_step
            loss_total_batch = val_step(args, batch, net, alice_bob_mac, key_ab, Alice_KB, Bob_KB, Alice_mapping, Bob_mapping,
                                sents, sents, 0.1, pad_idx, args.channel, cdmodel, ddim_scheduler)

            total_loss += loss_total_batch

            batch += 1

    print("================validate======================")
    print("epoch: ", epoch)
    print("loss_total_test (MSE + CE): ", total_loss / len(test_iterator))
    print("================validate======================")

    return total_loss / len(test_iterator)


# 将 cdmodel 和 ddim_scheduler 加入参数列表
def performance(args, SNR, deepsc, alice_bob_mac, key_ab, Alice_KB, Bob_KB, Alice_mapping, Bob_mapping, cdmodel,
                ddim_scheduler):
    bleu_score_1gram = BleuScore(1, 0, 0, 0)
    # bleu_score_2gram = BleuScore(0.5, 0.5, 0, 0)  # 主要关注这几个bleu分数
    # bleu_score_4gram = BleuScore(0.25, 0.25, 0.25, 0.25)

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

    with torch.no_grad():
        for epoch in range(1):  # 训练过程中的测试，为了省时间，依然只跑1次。真正定稿画图时再去跑 performance_for_34.py 里的 5 次。
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
                    out = greedy_decode(args, deepsc, alice_bob_mac, key_ab, Alice_KB, Bob_KB, Alice_mapping,
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

    score1 = np.mean(np.array(score), axis=0)  # 每个信噪比下的所有句子的平均BLEU分数(按照epoch平均)
    return score1


if __name__ == '__main__':
    now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    os.mkdir("./checkpoints/f/" + now)
    writer = SummaryWriter(log_dir="./logs/f/" + now)

    """ preparing the dataset """
    train_start_time = time.time()
    torch.manual_seed(5)
    args = parser.parse_args()

    with open(args.vocab_path, 'r') as f:
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
    deepsc = DeepSC(args.num_layers, num_vocab, num_vocab,
                    num_vocab, num_vocab, args.d_model, args.num_heads,
                    args.dff, 0.1).to(device)

    mi_net = Mine().to(device)  # 计算通信网络的互信息量的 完全没用

    alice_bob_mac = MAC().to(device)  # 我的那个IBSID组件 跟着流程走就行了 完全不用管
    key_ab = Key_net(args).to(device)  # 密钥的一个网络 作为IBSID生成和验证的一个输入 也是完全不用管

    Alice_KB = KnowledgeBase().to(device)  # 我的那个知识库组件 跟着流程走就行了 完全不用管
    Bob_KB = KnowledgeBase().to(device)

    Alice_mapping = KB_Mapping().to(device)  # 我的那个知识库映射组件 跟着流程走就行了 完全不用管
    Bob_mapping = KB_Mapping().to(device)

    # 实例化全新的带 AdaLN 的 cdmodel
    cdmodel = FeatureRestorationDiT(
        feature_dim=128,
        seq_len=32,
        hidden_size=256,
        depth=4,
        num_heads=8
    ).to(device)

    ddim_scheduler = DDIMScheduler(device=device)

    # 1. 加载 109 基线权重 
    OLD_CHECKPOINT_PATH = r'/root/autodl-tmp/restore/checkpoints/checkpoint_109.pth'
    checkpoint_old = torch.load(OLD_CHECKPOINT_PATH) 
    
    deepsc.load_state_dict(checkpoint_old['deepsc'])
    alice_bob_mac.load_state_dict(checkpoint_old['alice_bob_mac'])
    key_ab.load_state_dict(checkpoint_old['key_ab'])
    Alice_KB.load_state_dict(checkpoint_old['Alice_KB'])
    Bob_KB.load_state_dict(checkpoint_old['Bob_KB'])
    Alice_mapping.load_state_dict(checkpoint_old['Alice_mapping'])
    Bob_mapping.load_state_dict(checkpoint_old['Bob_mapping'])
    
  
    # 高分的 122权重路径
    LATEST_CDMODEL_PATH = r'/root/autodl-tmp/restore/checkpoints/f/checkpoint_122_0.9457.pth'
    checkpoint_latest = torch.load(LATEST_CDMODEL_PATH)
    cdmodel.load_state_dict(checkpoint_latest['cdmodel'])

    deepsc = deepsc.to(device)
    alice_bob_mac = alice_bob_mac.to(device)
    key_ab = key_ab.to(device)
    Alice_KB = Alice_KB.to(device)
    Bob_KB = Bob_KB.to(device)
    Alice_mapping = Alice_mapping.to(device)
    Bob_mapping = Bob_mapping.to(device)
    cdmodel = cdmodel.to(device)

    optimizer = torch.optim.Adam(deepsc.parameters(),
                                 lr=1e-5, betas=(0.9, 0.98), eps=1e-8, weight_decay=5e-4)
    mi_opt = torch.optim.Adam(mi_net.parameters(), lr=1e-3)
    # opt = NoamOpt(args.d_model, 1, 4000, optimizer)

    # 依然只训练 cdmodel，不要去动 deepsc 的参数 
    # 联合训练的梯度会通过冻结的 deepsc.decoder 传回 cdmodel
    # 学习率降低到 5e-5，进行平滑的联合微调
    optimizer_joint = torch.optim.Adam(
        list(cdmodel.parameters()),  
        lr=5e-5, betas=(0.9, 0.98), eps=1e-8, weight_decay=5e-4)

    # 训练deepsc模型
    SNR = [0]  # 训练过程中的快速验证，只看 0dB
    
    for epoch in range(args.epochs):
        record_loss = 1000  # 其实是loss，设置的大一点

        # 把 cdmodel, ddim_scheduler 和 optimizer_joint 传进 train
        loss_total_train = train(epoch, args, deepsc, alice_bob_mac, key_ab, Alice_KB, Bob_KB, Alice_mapping, Bob_mapping,
                          cdmodel, ddim_scheduler, optimizer_joint)

        # 把 cdmodel 和 ddim_scheduler 传进 validate
        loss_total_test = validate(epoch, args, deepsc, alice_bob_mac, key_ab, Alice_KB, Bob_KB, Alice_mapping,
                                 Bob_mapping, cdmodel, ddim_scheduler)

        # 把 cdmodel 和 ddim_scheduler 传进 performance
        bleu_score = performance(args, SNR, deepsc, alice_bob_mac, key_ab, Alice_KB, Bob_KB, Alice_mapping, Bob_mapping,
                                 cdmodel, ddim_scheduler)
        print("bleu_score: ", bleu_score)

        if loss_total_test < record_loss:  # 如果验证的联合 loss 小于之前的loss（性能更好了）
            checkpoint = {
                "deepsc": deepsc.state_dict(),
                "cdmodel": cdmodel.state_dict(),  
            }
            torch.save(checkpoint, './checkpoints/f/' + now + '/checkpoint_{}'.format(str(epoch).zfill(3)) + '_{}.pth'.format(
                str(bleu_score)[1:7]))  # 保存模型 这个您随意保存
            record_loss = loss_total_test  # 更新最小的 loss

        # 这里的日志记录也同步改了名字
        writer.add_scalar('Loss_Total_Train', loss_total_train, epoch)
        writer.add_scalar('Loss_Total_Test', loss_total_test, epoch)
        writer.add_scalar('BLEU_score', bleu_score[0], epoch)