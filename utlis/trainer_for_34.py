# 9.4更新 将碰撞的Loss删除了(和雪崩的有点重复) 将所有的loss加在一起统一更新参数 把mod全撤了 全部直接训练
import json
import math
import random

import torch
import torch.nn as nn
import numpy as np
from models.mutual_info import sample_batch, mutual_information
import torch.nn.functional as F

from utlis.tools import SeqtoText, BleuScore

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# def cosine_similarity(tensor1, tensor2):  # 形状为[batch_size, 1, 128] 即mac
#     sum = F.cosine_similarity(tensor1, tensor2, dim=2)
#     average_cosine_similarity = torch.mean(sum)
#     return average_cosine_similarity.item()

# def forward_with_codebook(self, x, memory, trg_padding_mask, m_feature_ratio, sub_vectors_expanded, args):
#     x_embed = self.embedding(x)
#     x_embed_pos = self.pos_encoding(x_embed)
#     x = x_embed_pos
#
#     look_ahead_mask_expanded = torch.repeat_interleave(subsequent_mask(x.size(-2)), x.shape[0], dim=0).to(x.device)
#
#     trg_padding_mask_trim = apply_m_feature_ratio_to_mask(trg_padding_mask, m_feature_ratio)
#     # memory = apply_m_feature_ratio_to_m_feature(memory, trg_padding_mask, m_feature_ratio)
#
#     codebook_memory = torch.concat([memory, sub_vectors_expanded], dim=1)
#     # codebook_memory = torch.nn.functional.normalize(codebook_memory, dim=-1, p=2)
#     codebook_mask = torch.zeros((sub_vectors_expanded.size(0), 1, sub_vectors_expanded.size(1))).to(
#         sub_vectors_expanded.device)
#
#     combined_mask = torch.concat((trg_padding_mask_trim, codebook_mask), dim=-1)
#
#     codebook_memory_pos = codebook_memory
#     for dec_layer in self.dec_layers:
#         x = dec_layer(x, codebook_memory_pos, look_ahead_mask_expanded, combined_mask)
#
#     return x

def greedy_decode_bleu_predict(args, deepsc, alice_bob_mac, key_ab, eve, Alice_KB, Bob_KB, Eve_KB, Alice_mapping, Bob_mapping, Eve_mapping, src, noise_std, max_len, padding_idx, start_symbol, channel):
    """
    这里采用贪婪解码器，如果需要更好的性能情况下，可以使用beam search decode
    """
    freeze_net(key_ab, False)
    freeze_net(alice_bob_mac, False)
    freeze_net(eve, False)
    freeze_net(deepsc, False)
    freeze_net(Alice_KB, False)
    freeze_net(Bob_KB, False)
    freeze_net(Eve_KB, False)
    freeze_net(Alice_mapping, False)
    freeze_net(Bob_mapping, False)
    freeze_net(Eve_mapping, False)
    # 加一句bleu预测网络也是false

    bs = src.size(0)
    key = generate_key(args, src.shape)
    key_wrong = generate_key(args, src.shape)
    while torch.equal(key, key_wrong):
        key_wrong = generate_key(args, src.shape)

    Alice_ID = torch.randn(1, args.d_model).to(device)
    Bob_ID = torch.randn(1, args.d_model).to(device)
    Alice_tmp = Alice_KB(Alice_ID)  # 这就是知识库 不断的更新 只有最开始的一轮才是真正的ID，形状是[8, 128]
    Bob_tmp = Bob_KB(Bob_ID)
    Alice_mapping_tmp = Alice_mapping(Alice_tmp)  # 形状是[8, 128]
    Bob_mapping_tmp = Bob_mapping(Bob_tmp)
    Alice_kb_final = Alice_tmp.repeat(bs, 1, 1)  # 进行复制
    Bob_kb_final = Bob_tmp.repeat(bs, 1, 1)

    Alice_mapping_final = Alice_mapping_tmp.repeat(bs, 1, 1)
    Bob_mapping_final = Bob_mapping_tmp.repeat(bs, 1, 1)

    # create src_mask
    channels = Channels()
    src_mask = (src == padding_idx).unsqueeze(-2).type(torch.FloatTensor).to(device)  # [batch, 1, seq_len]

    key_ebd = key_ab(key)
    enc_output = deepsc.encoder(src, src_mask, Alice_kb_final, Bob_mapping_final)
    enc_output = enc_output[:, :31, :]  # 只前三十个通道
    mac = alice_bob_mac.mac_encoder(key_ebd, enc_output, Alice_kb_final, Bob_mapping_final)
    semantic_mac = torch.cat([enc_output, mac], dim=1)
    channel_enc_output = deepsc.channel_encoder(semantic_mac)
    Tx_sig = PowerNormalize(channel_enc_output)

    if channel == 'AWGN':
        Rx_sig = channels.AWGN(Tx_sig, noise_std)  # 这个noise_std也是一个数
    elif channel == 'Rayleigh':
        Rx_sig = channels.Rayleigh(Tx_sig, noise_std)
    elif channel == 'Rician':
        Rx_sig = channels.Rician(Tx_sig, noise_std)
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")
    memory = deepsc.channel_decoder(Rx_sig)
    f_p = memory[:, :31, :]  # 前31个通道
    mac_p = memory[:, 31:, :]  # 后31个通道
    outputs = torch.ones(src.size(0), 1).fill_(start_symbol).type_as(src.data)

    for i in range(max_len - 1):  # 下面就是解码
        # create the decode mask
        trg_mask = (outputs == padding_idx).unsqueeze(-2).type(torch.FloatTensor).to(device)  # [batch, 1, seq_len]
        look_ahead_mask = subsequent_mask(outputs.size(1)).type(torch.FloatTensor).to(device)
        # print(look_ahead_mask)
        combined_mask = torch.max(trg_mask, look_ahead_mask)
        combined_mask = combined_mask.to(device)

        # decode the received signal
        dec_output = deepsc.decoder(outputs, f_p, combined_mask, src_mask, Alice_mapping_final, Bob_kb_final, mac_p)
        pred = deepsc.dense(dec_output)

        # predict the word
        prob = pred[:, -1:, :]  # (batch_size, 1, vocab_size), 取最后一个单词的预测概率
        #         # prob = prob.squeeze()

        # return the max-prob index
        _, next_word = torch.max(prob, dim=-1)
        # next_word = next_word.unsqueeze(1)

        # next_word = next_word.data[0]
        outputs = torch.cat([outputs, next_word], dim=1)  # [bs, 30]
    return outputs


# 计算mac判别准确度
def mac_accuracy(result, target): # 返回的是平均到batch_size后的准确率
    # result和target的形状都是[batch_size, 1, 128]
    ct = 0
    for i in range(result.shape[0]):
        if torch.equal(result[i], target[i]):
            ct += 1
    return ct / result.shape[0]

def generate_key(args, data_size):  # 输入的data_size=[bs, 31] 输出形状是[bs, 10]
    k_range = [6, 8]
    # 使用torch.randint生成均匀分布的随机整数
    key = torch.randint(high=k_range[1], low=k_range[0], size=(data_size[0], 8), dtype=torch.int32)
    # 创建起始和结束索引的列
    start_column = torch.full((data_size[0], 1), args.start_idx, dtype=torch.int32)
    end_column = torch.full((data_size[0], 1), args.end_idx, dtype=torch.int32)
    # 沿着列方向拼接
    key = torch.cat([start_column, key, end_column], dim=1)
    return key.to(device)


criterion_mac = nn.BCELoss().to(device)  # 二元交叉熵损失函数 mac验证的时候计算loss用
# criterion_mac = nn.CrossEntropyLoss().to(device)  # 交叉熵损失函数 mac验证的时候计算loss用
criterion_bcelogits = nn.BCEWithLogitsLoss()  # 注意alice判别的损失函数要用这个


def SNR_to_noise(snr):  # 计算信噪比为snr时的 噪声标准差  将DB转换成线性的
    snr = 10 ** (snr / 10)
    noise_std = 1 / np.sqrt(snr)
    return noise_std

# 拉普拉斯噪声(不准备弄了 因为很难说清楚为什么用拉普拉斯噪声)
def laplace_noise(tensor, noise_std):
    noise = torch.distributions.laplace.Laplace(0, noise_std).sample(tensor.size())  # 0表示均值，noise_std表示标准差
    return tensor + noise.to(device)


def freeze_net(net, is_requires_grad):
    for param in net.parameters():
        param.requires_grad = is_requires_grad
    if is_requires_grad:
        net.train()
    else:
        net.eval()

criterion_noise = nn.MSELoss().to(device)

# loss_alice = torch.tensor(0.)
# loss_eve = torch.tensor(0.)
#
# loss_alice_test = torch.tensor(0.)
# loss_eve_test = torch.tensor(0.)


def train_step(schedule, cdmodel, args, epoch, batch, model, alice_bob_mac, key_ab, eve, Alice_KB, Bob_KB, Eve_KB, Alice_mapping, Bob_mapping, Eve_mapping, src, trg, src_eve, n_var, pad, opt_joint, channel, mi_net=None):  # 模型，发送的128个句子，发送的128个句子，噪声标准差(类型数字)，数字0，deepsc优化器，信道类型
    torch.autograd.set_detect_anomaly(True)  # 检测梯度异常

    trg_inp = trg[:, :-1]  # 把每个句子的最后一个单词(填充的PAD0或END2)去掉
    trg_real = trg[:, 1:]  # 把每个句子的第一个单词(开始的START1)去掉
    trg_inp_eve = src_eve[:, :-1]  # 把每个句子的最后一个单词(填充的PAD0或END2)去掉
    trg_real_eve = src_eve[:, 1:]  # 把每个句子的第一个单词(开始的START1)去掉

    src_mask, look_ahead_mask = create_masks(src, trg_inp, pad)
    src_mask_eve, look_ahead_mask_eve = create_masks(src_eve, trg_inp_eve, pad)


    channels = Channels()
    bs = args.batch_size
    snr_min, snr_max = -5.0, 20.0  # 学习的信噪比区间 不用转换成线性的 线性的反而不好学 因为跨度太大
    noise_std = np.random.uniform(SNR_to_noise(snr_min), SNR_to_noise(snr_max), size=(1))[0]  # 不好的环境
    snr_lin = 1.0 / (noise_std ** 2)
    snr_db = 10 * torch.log10(torch.tensor(snr_lin, device=device))
    snr = snr_db.expand(bs).float()  # 输入到snr网络中的snr 单位是db

    key = generate_key(args, src.shape)

    freeze_net(key_ab, False)
    freeze_net(alice_bob_mac, False)
    freeze_net(eve, False)
    freeze_net(Alice_KB, False)
    freeze_net(Bob_KB, False)
    freeze_net(Eve_KB, False)
    freeze_net(Alice_mapping, False)
    freeze_net(Bob_mapping, False)
    freeze_net(Eve_mapping, False)
    freeze_net(model, False)
    freeze_net(cdmodel, True)


    # 知识库 生成随机的[bs, 1， 128]的张量
    # 这里错了 其实应该生成[1, 128]的张量，然后通过网络生成一个[8, 128]形状的张量，然后复制成[bs, 8, 128]的张量即可（保证每个batch的句子都使用同一个知识库）
    # 已改
    Alice_ID = torch.randn(1, args.d_model).to(device)
    Bob_ID = torch.randn(1, args.d_model).to(device)
    Eve_ID = torch.randn(1, args.d_model).to(device)
    Alice_tmp = Alice_KB(Alice_ID)  # 这就是知识库 不断的更新 只有最开始的一轮才是真正的ID，形状是[8, 128]
    Bob_tmp = Bob_KB(Bob_ID)
    Eve_tmp = Eve_KB(Eve_ID)
    Alice_mapping_tmp = Alice_mapping(Alice_tmp)  # 形状是[8, 128]
    Bob_mapping_tmp = Bob_mapping(Bob_tmp)
    Eve_mapping_tmp = Eve_mapping(Eve_tmp)
    Alice_kb_final = Alice_tmp.repeat(bs, 1, 1)  # 进行复制
    Bob_kb_final = Bob_tmp.repeat(bs, 1, 1)
    Eve_kb_final = Eve_tmp.repeat(bs, 1, 1)
    Alice_mapping_final = Alice_mapping_tmp.repeat(bs, 1, 1)
    Bob_mapping_final = Bob_mapping_tmp.repeat(bs, 1, 1)
    Eve_mapping_final = Eve_mapping_tmp.repeat(bs, 1, 1)

    key_ebd = key_ab(key)  # 生成密钥

    enc_output = model.encoder(src, src_mask, Alice_kb_final, Bob_mapping_final)  # f
    enc_output = enc_output[:, :31, :]  # 只前31个通道 f


    t = schedule.sample_timesteps(bs)  # [bs]
    eps = torch.randn_like(enc_output)  # [bs, L, D]
    f_t = schedule.q_sample(f0=enc_output, t=t, eps=eps)  # [bs, L, D]

    eps_pred = cdmodel(f_t=f_t, t=t, hat_f=enc_output)  # [bs, L, D]
    loss = F.mse_loss(eps_pred, eps)

    opt_joint.zero_grad(set_to_none=True)
    loss.backward()
    opt_joint.step()

    return loss.item()


def val_step(schedule, cdmodel, args, batch, model, alice_bob_mac, key_ab, eve, Alice_KB, Bob_KB, Eve_KB, Alice_mapping, Bob_mapping, Eve_mapping, src, trg, src_eve, n_var, pad, channel):  # 参数模型，发送的128个句子，发送的128个句子，噪声标准差(数字0.1)，数字0，信道类型
    trg_inp = trg[:, :-1]  # 把每个句子的最后一个单词(填充的PAD0或END2)去掉
    trg_real = trg[:, 1:]  # 把每个句子的第一个单词(开始的START1)去掉
    trg_inp_eve = src_eve[:, :-1]  # 把每个句子的最后一个单词(填充的PAD0或END2)去掉
    trg_real_eve = src_eve[:, 1:]  # 把每个句子的第一个单词(开始的START1)去掉

    src_mask, look_ahead_mask = create_masks(src, trg_inp, pad)
    src_mask_eve, look_ahead_mask_eve = create_masks(src_eve, trg_inp_eve, pad)

    channels = Channels()
    bs = src.size(0)
    snr_min, snr_max = -5.0, 20.0  # 学习的信噪比区间 不用转换成线性的 线性的反而不好学 因为跨度太大
    noise_std = np.random.uniform(SNR_to_noise(snr_min), SNR_to_noise(snr_max), size=(1))[0]  # 不好的环境
    snr_lin = 1.0 / (noise_std ** 2)
    snr_db = 10 * torch.log10(torch.tensor(snr_lin, device=device))
    snr = snr_db.expand(bs).float()  # 输入到snr网络中的snr 单位是db

    key = generate_key(args, src.shape)

    freeze_net(key_ab, False)
    freeze_net(alice_bob_mac, False)
    freeze_net(eve, False)
    freeze_net(model, False)
    freeze_net(Alice_KB, False)
    freeze_net(Bob_KB, False)
    freeze_net(Eve_KB, False)
    freeze_net(Alice_mapping, False)
    freeze_net(Bob_mapping, False)
    freeze_net(Eve_mapping, False)
    freeze_net(cdmodel, False)

    args.vocab_file = args.vocab_file
    vocab = json.load(open(args.vocab_file, 'rb'))
    token_to_idx = vocab['token_to_idx']
    pad_idx = token_to_idx["<PAD>"]
    start_idx = token_to_idx["<START>"]
    end_idx = token_to_idx["<END>"]

    Alice_ID = torch.randn(1, args.d_model).to(device)
    Bob_ID = torch.randn(1, args.d_model).to(device)
    Eve_ID = torch.randn(1, args.d_model).to(device)
    Alice_tmp = Alice_KB(Alice_ID)  # 这就是知识库 不断的更新 只有最开始的一轮才是真正的ID，形状是[8, 128]
    Bob_tmp = Bob_KB(Bob_ID)
    Eve_tmp = Eve_KB(Eve_ID)
    Alice_mapping_tmp = Alice_mapping(Alice_tmp)  # 形状是[8, 128]
    Bob_mapping_tmp = Bob_mapping(Bob_tmp)
    Eve_mapping_tmp = Eve_mapping(Eve_tmp)
    Alice_kb_final = Alice_tmp.repeat(bs, 1, 1)  # 进行复制
    Bob_kb_final = Bob_tmp.repeat(bs, 1, 1)
    Eve_kb_final = Eve_tmp.repeat(bs, 1, 1)
    Alice_mapping_final = Alice_mapping_tmp.repeat(bs, 1, 1)
    Bob_mapping_final = Bob_mapping_tmp.repeat(bs, 1, 1)
    Eve_mapping_final = Eve_mapping_tmp.repeat(bs, 1, 1)

    key_ebd = key_ab(key)  # 生成密钥

    enc_output = model.encoder(src, src_mask, Alice_kb_final, Bob_mapping_final)  # f
    enc_output = enc_output[:, :31, :]  # 只前31个通道 f

    t = schedule.sample_timesteps(bs)  # [bs]
    eps = torch.randn_like(enc_output)  # [bs, L, D]
    f_t = schedule.q_sample(f0=enc_output, t=t, eps=eps)  # [bs, L, D]

    eps_pred = cdmodel(f_t=f_t, t=t, hat_f=enc_output)  # [bs, L, D]
    loss = F.mse_loss(eps_pred, eps)

    return loss.item()


def mac_accuracy_all(normal, eve1, eve2): # 返回的是检测成功率
    target1 = torch.ones(normal.size(0), 1).float().to(device)
    target0 = torch.zeros(eve1.size(0), 1).float().to(device)
    # normal和target的形状都是[batch_size, 1, 128]
    ct_normal = 0
    for i in range(normal.shape[0]):
        if torch.equal(normal[i], target1[i]):
            ct_normal += 1

    ct_eve1 = 0
    for i in range(eve1.shape[0]):
        if torch.equal(eve1[i], target0[i]):
            ct_eve1 += 1

    ct_eve2 = 0
    for i in range(eve2.shape[0]):
        if torch.equal(eve2[i], target0[i]):
            ct_eve2 += 1

    return (ct_normal + ct_eve1 + ct_eve2) / (normal.size(0) + eve1.size(0) + eve2.size(0))  # 返回总的准确率

def greedy_decode(CAEM_with_SNR, fms, alice_verifier, args, deepsc, alice_bob_mac, key_ab, eve, Alice_KB, Bob_KB, Eve_KB, Alice_mapping, Bob_mapping, Eve_mapping, src, src_eve, noise_std, max_len, pad, start_symbol, channel):
    trg_inp = src[:, :-1]  # 把每个句子的最后一个单词(填充的PAD0或END2)去掉
    trg_real = src[:, 1:]  # 把每个句子的第一个单词(开始的START1)去掉
    trg_inp_eve = src_eve[:, :-1]  # 把每个句子的最后一个单词(填充的PAD0或END2)去掉
    trg_real_eve = src_eve[:, 1:]  # 把每个句子的第一个单词(开始的START1)去掉

    bs = args.batch_size
    src_mask, look_ahead_mask = create_masks(src, trg_inp, pad)
    src_mask_eve, look_ahead_mask_eve = create_masks(src_eve, trg_inp_eve, pad)

    perm = torch.randperm(bs, device=src.device)
    src_neg = src[perm]
    trg_inp_neg = src_neg[:, :-1]
    trg_real_neg = src_neg[:, 1:]
    src_mask_neg, look_ahead_mask_neg = create_masks(src_neg, trg_inp_neg, pad)


    channels = Channels()
    bs = src.size(0)
    snr_lin = 1.0 / (noise_std ** 2)
    snr_db = 10 * torch.log10(torch.tensor(snr_lin, device=device))
    snr = snr_db.expand(bs).float()  # 输入到snr网络中的snr 单位是db

    key = generate_key(args, src.shape)

    freeze_net(key_ab, False)
    freeze_net(alice_bob_mac, False)
    freeze_net(eve, False)
    freeze_net(deepsc, False)
    freeze_net(Alice_KB, False)
    freeze_net(Bob_KB, False)
    freeze_net(Eve_KB, False)
    freeze_net(Alice_mapping, False)
    freeze_net(Bob_mapping, False)
    freeze_net(Eve_mapping, False)
    freeze_net(CAEM_with_SNR, False)
    freeze_net(fms, False)
    freeze_net(alice_verifier, False)

    args.vocab_file = args.vocab_file
    vocab = json.load(open(args.vocab_file, 'rb'))
    token_to_idx = vocab['token_to_idx']
    pad_idx = token_to_idx["<PAD>"]
    start_idx = token_to_idx["<START>"]
    end_idx = token_to_idx["<END>"]

    Alice_ID = torch.randn(1, args.d_model).to(device)
    Bob_ID = torch.randn(1, args.d_model).to(device)
    Eve_ID = torch.randn(1, args.d_model).to(device)
    Alice_tmp = Alice_KB(Alice_ID)  # 这就是知识库 不断的更新 只有最开始的一轮才是真正的ID，形状是[8, 128]
    Bob_tmp = Bob_KB(Bob_ID)
    Eve_tmp = Eve_KB(Eve_ID)
    Alice_mapping_tmp = Alice_mapping(Alice_tmp)  # 形状是[8, 128]
    Bob_mapping_tmp = Bob_mapping(Bob_tmp)
    Eve_mapping_tmp = Eve_mapping(Eve_tmp)
    Alice_kb_final = Alice_tmp.repeat(bs, 1, 1)  # 进行复制
    Bob_kb_final = Bob_tmp.repeat(bs, 1, 1)
    Eve_kb_final = Eve_tmp.repeat(bs, 1, 1)
    Alice_mapping_final = Alice_mapping_tmp.repeat(bs, 1, 1)
    Bob_mapping_final = Bob_mapping_tmp.repeat(bs, 1, 1)
    Eve_mapping_final = Eve_mapping_tmp.repeat(bs, 1, 1)

    # 先测一下deepsc的性能
    key_ebd = key_ab(key)  # 生成密钥

    enc_output = deepsc.encoder(src, src_mask, Alice_kb_final, Bob_mapping_final)  # f
    enc_output = enc_output[:, :31, :]  # 只前31个通道 f
    mac = alice_bob_mac.mac_encoder(key_ebd, enc_output, Alice_kb_final, Bob_mapping_final)
    semantic_mac = torch.cat([enc_output, mac], dim=1)
    g = CAEM_with_SNR(enc_output, snr)  # alice得到的g

    enc_output_eve = deepsc.encoder(src_eve, src_mask_eve, Eve_kb_final, Bob_mapping_final)  # f
    enc_output_eve = enc_output_eve[:, :31, :]  # 只前31个通道
    mac_eve = eve.mac_encoder(enc_output_eve, Eve_kb_final, Bob_mapping_final)
    semantic_mac_eve = torch.cat([enc_output_eve, mac_eve], dim=1)

    # enc_output_neg = deepsc.encoder(src_neg, src_mask_neg, Alice_kb_final, Bob_mapping_final)
    # enc_output_neg = enc_output_neg[:, :31, :]  # 只前31个通道
    # mac_neg = alice_bob_mac.mac_encoder(key_ebd, enc_output_neg, Alice_kb_final, Bob_mapping_final)
    # semantic_mac_neg = torch.cat([enc_output_neg, mac_neg], dim=1)

    channel_enc_output = deepsc.channel_encoder(semantic_mac)
    channel_enc_output_eve = deepsc.channel_encoder(semantic_mac_eve)
    # channel_enc_output_neg = deepsc.channel_encoder(semantic_mac_neg)
    Tx_sig = PowerNormalize(channel_enc_output)
    Tx_sig_eve = PowerNormalize(channel_enc_output_eve)
    # Tx_sig_neg = PowerNormalize(channel_enc_output_neg)

    if channel == 'AWGN':
        Rx_sig = channels.AWGN(Tx_sig, noise_std)
        Rx_sig_eve = channels.AWGN(Tx_sig_eve, noise_std)
    elif channel == 'Rayleigh':
        Rx_sig = channels.Rayleigh(Tx_sig, noise_std)
        Rx_sig_eve = channels.Rayleigh(Tx_sig_eve, noise_std)
        # Rx_sig_neg = channels.Rayleigh(Tx_sig_neg, noise_std)
    elif channel == 'Rician':
        Rx_sig = channels.Rician(Tx_sig, noise_std)
        Rx_sig_eve = channels.Rician(Tx_sig_eve, noise_std)
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")
    channel_dec_output = deepsc.channel_decoder(Rx_sig)
    channel_dec_output_eve = deepsc.channel_decoder(Rx_sig_eve)
    # channel_dec_output_neg = deepsc.channel_decoder(Rx_sig_neg)
    f_p = channel_dec_output[:, :31, :]  # 前31个通道 发送的时候也是
    f_eve_p = channel_dec_output_eve[:, :31, :]  # 前31个通道 发送的时候也是
    # f_neg_p = channel_dec_output_neg[:, :31, :]  # 前31个通道 发送的时候也是

    # 下面就是那个映射
    g_p = CAEM_with_SNR(f_p, snr)  # bob得到的g'
    g_pp, Mk, ent, probs, onehot = fms(g_p, snr, tau=0.7, hard=True)  # 经过筛选的g'

    g_eve_p = CAEM_with_SNR(f_eve_p, snr)  # bob得到的g'
    g_eve_pp, Mk_eve, ent_eve, probs_eve, onehot_eve = fms(g_eve_p, snr, tau=0.7, hard=True)  # 经过筛选的g_eve'

    # g_neg_p = CAEM_with_SNR(f_neg_p, snr)  # bob得到的g'
    # g_neg_pp, Mk_neg, ent_neg, probs_neg, onehot_neg = fms(g_neg_p, snr, tau=0.7, hard=True)  # 经过筛选的g_neg'

    # 然后进行判别
    logits = alice_verifier(g, g_pp)  # 判别结果
    logits_eve = alice_verifier(g, g_eve_pp)  # 判别eve结果
    # logits_neg = alice_verifier(g, g_neg_pp)  # 判别neg结果

    pred_pos = (logits >= 0).float()  # [bs,1]
    alice_1 = pred_pos.mean().item()  # 正样本正确率 = 预测为1的比例

    pred_neg = (logits_eve >= 0).float()  # [bs,1]
    eve_0 = (1.0 - pred_neg).mean().item()  # 负样本正确率 = 预测为0的比例

    # pred_neg_neg = (logits_neg >= 0).float()  # [bs,1]
    # neg_0 = (1.0 - pred_neg_neg).mean().item()  # 负样本正确率 = 预测为0的比例

    return alice_1, eve_0

def train_mi(model, mi_net, src, n_var, padding_idx, opt, channel):
    mi_net.train()
    opt.zero_grad()
    channels = Channels()
    src_mask = (src == padding_idx).unsqueeze(-2).type(torch.FloatTensor).to(device)  # [batch, 1, seq_len]
    enc_output = model.encoder(src, src_mask)
    channel_enc_output = model.channel_encoder(enc_output)
    Tx_sig = PowerNormalize(channel_enc_output)

    if channel == 'AWGN':
        Rx_sig = channels.AWGN(Tx_sig, n_var)
    elif channel == 'Rayleigh':
        Rx_sig = channels.Rayleigh(Tx_sig, n_var)
    elif channel == 'Rician':
        Rx_sig = channels.Rician(Tx_sig, n_var)
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")

    joint, marginal = sample_batch(Tx_sig, Rx_sig)
    mi_lb, _, _ = mutual_information(joint, marginal, mi_net)
    loss_mine = -mi_lb

    loss_mine.backward()
    torch.nn.utils.clip_grad_norm_(mi_net.parameters(), 10.0)
    opt.step()

    return loss_mine.item()


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        # 将数组全部填充为某一个值
        true_dist.fill_(self.smoothing / (self.size - 2))
        # 按照index将input重新排列
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        # 第一行加入了<strat> 符号，不需要加入计算
        true_dist[:, self.padding_idx] = 0  #
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist)


class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        self._weight_decay = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        weight_decay = self.weight_decay()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
            p['weight_decay'] = weight_decay
        self._rate = rate
        self._weight_decay = weight_decay
        # update weights
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step

        # if step <= 3000 :
        #     lr = 1e-3

        # if step > 3000 and step <=9000:
        #     lr = 1e-4

        # if step>9000:
        #     lr = 1e-5

        lr = self.factor * \
             (self.model_size ** (-0.5) *
              min(step ** (-0.5), step * self.warmup ** (-1.5)))

        return lr

        # return lr

    def weight_decay(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step

        if step <= 3000:
            weight_decay = 1e-3

        if step > 3000 and step <= 9000:
            weight_decay = 0.0005

        if step > 9000:
            weight_decay = 1e-4

        weight_decay = 0
        return weight_decay


class Channels():

    def AWGN(self, Tx_sig, n_var):  # 参数分别是发送信号和噪声方差
        Rx_sig = Tx_sig + torch.normal(0.0, n_var, size=Tx_sig.shape).to(device)
        return Rx_sig  # 返回接收信号

    def Rayleigh(self, Tx_sig, n_var):
        shape = Tx_sig.shape
        # H_real = torch.normal(0, math.sqrt(1 / 2), size=[1]).to(device)
        # H_imag = torch.normal(0, math.sqrt(1 / 2), size=[1]).to(device)
        H_real = np.random.normal(0, math.sqrt(1 / 2))
        H_imag = np.random.normal(0, math.sqrt(1 / 2))
        H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(device)
        Tx_sig = torch.matmul(Tx_sig.view(shape[0], -1, 2), H)
        Rx_sig = self.AWGN(Tx_sig, n_var)
        # Channel estimation
        Rx_sig = torch.matmul(Rx_sig, torch.inverse(H)).view(shape)

        return Rx_sig

    def Rician(self, Tx_sig, n_var, K=1):
        shape = Tx_sig.shape
        mean = math.sqrt(K / (K + 1))
        std = math.sqrt(1 / (K + 1))
        H_real = torch.normal(mean, std, size=[1]).to(device)
        H_imag = torch.normal(mean, std, size=[1]).to(device)
        H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(device)
        Tx_sig = torch.matmul(Tx_sig.view(shape[0], -1, 2), H)
        Rx_sig = self.AWGN(Tx_sig, n_var)
        # Channel estimation
        Rx_sig = torch.matmul(Rx_sig, torch.inverse(H)).view(shape)

        return Rx_sig


def initNetParams(model):  # 初始化网络参数
    '''Init net parameters.'''
    for p in model.parameters():
        if p.dim() > 1:  # 只对于维度大于1的参数进行初始化
            nn.init.xavier_uniform_(p)  # Xavier初始化，训练的时候收敛更快更好
    return model


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    # 产生下三角矩阵
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask)


def create_masks(src, trg, padding_idx):  # 输入的128个句子，输入的128个去掉最后一个单词的句子，数字0
    # print("src: ", src)
    # print("trg: ", trg)
    src_mask = (src == padding_idx).unsqueeze(-2).type(torch.FloatTensor)  # [batch, 1, seq_len]，
    # print("src_mask: ", src_mask)  # 128x1x31，就是将src中各个句子 本来是一个数组[1,1,4,5,...]，现在变成了[[1,1,4,5,...]](加了一维)，且pad位置变成1，其余为0
    trg_mask = (trg == padding_idx).unsqueeze(-2).type(torch.FloatTensor)  # [batch, 1, seq_len]
    # print("trg_mask: ", trg_mask)  # 128x1x30，同上
    look_ahead_mask = subsequent_mask(trg.size(-1)).type_as(trg_mask.data)
    # print("look_ahead_mask: ", look_ahead_mask)  # 1x30x30，由一个30x30的矩阵构成，一个[上三角矩阵]，矩阵是2维，这里在最外边加了一个[]，矩阵第一行0,1,1,1,...;第二行0,0,1,1...;最后一行000000
    combined_mask = torch.max(trg_mask, look_ahead_mask)  # 既不能看到填充的部分 也不能看到未来的部分
    # print("combined_mask: ", combined_mask)  # 1x30x30，由30个30x30的矩阵构成

    return src_mask.to(device), combined_mask.to(device)



criterion = nn.CrossEntropyLoss(reduction='none').to(device)

# 定义损失函数
def loss_function(x, trg, padding_idx):
    loss = criterion(x, trg)  # x与预期的交叉熵
    mask = (trg != padding_idx).type_as(loss.data)  # mask去掉padding的部分
    loss *= mask  # 将padding的部分的loss置为0，因为我们通常会使用填充标记来对齐不同长度的序列，但是这些填充部分不应该对损失产生影响

    return loss.mean()  # 返回loss的平均值


def PowerNormalize(x):
    x_square = torch.mul(x, x)
    power = torch.mean(x_square).sqrt()
    x = torch.div(x, power)

    return x