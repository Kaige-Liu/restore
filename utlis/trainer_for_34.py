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


class DDIMScheduler:
    def __init__(self, num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02, device="cuda:0"):
        self.num_train_timesteps = num_train_timesteps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, original_samples, noise, timesteps):
        alphas_cumprod_t = self.alphas_cumprod[timesteps].view(-1, 1, 1)
        sqrt_alpha_prod = torch.sqrt(alphas_cumprod_t)
        sqrt_one_minus_alpha_prod = torch.sqrt(1 - alphas_cumprod_t)
        return sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise

    @torch.no_grad()
    def ddim_sample(self, model, f_cond, snr_tensor, num_inference_steps=20, guidance_scale=2.5, eta=0.0):
        model.eval()
        bs, seq_len, feature_dim = f_cond.shape
        device = f_cond.device

        step_ratio = self.num_train_timesteps // num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        timesteps = torch.from_numpy(timesteps).to(device)

        x_t = torch.randn((bs, seq_len, feature_dim), device=device)

        context_mask_uncond = torch.ones(bs, dtype=torch.bool, device=device)
        context_mask_cond = torch.zeros(bs, dtype=torch.bool, device=device)

        for i, t in enumerate(timesteps):
            t_tensor = torch.full((bs,), t, device=device, dtype=torch.long)

            noise_pred_uncond = model(x_t, f_cond, t_tensor, snr_tensor, context_mask=context_mask_uncond)
            noise_pred_cond = model(x_t, f_cond, t_tensor, snr_tensor, context_mask=context_mask_cond)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            alpha_prod_t = self.alphas_cumprod[t]
            alpha_prod_t_prev = self.alphas_cumprod[timesteps[i + 1]] if i < len(timesteps) - 1 else torch.tensor(1.0,
                                                                                                                  device=device)

            pred_original_sample = (x_t - torch.sqrt(1 - alpha_prod_t) * noise_pred) / torch.sqrt(alpha_prod_t)
            variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
            std_dev_t = eta * torch.sqrt(variance)

            pred_sample_direction = torch.sqrt(1 - alpha_prod_t_prev - std_dev_t ** 2) * noise_pred
            x_t = torch.sqrt(alpha_prod_t_prev) * pred_original_sample + pred_sample_direction

            if eta > 0:
                x_t += std_dev_t * torch.randn_like(x_t)
        return x_t


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


def freeze_net(net, is_requires_grad):
    for param in net.parameters():
        param.requires_grad = is_requires_grad
    if is_requires_grad:
        net.train()
    else:
        net.eval()


def train_step(args, epoch, batch, model, alice_bob_mac, key_ab, Alice_KB, Bob_KB, Alice_mapping, Bob_mapping, src, trg,
               n_var, pad, opt_joint, channel, cdmodel, ddim_scheduler, mi_net=None):
    torch.autograd.set_detect_anomaly(True)  # 检测梯度异常

    trg_inp = trg[:, :-1]  # 把每个句子的最后一个单词(填充的PAD0或END2)去掉
    trg_real = trg[:, 1:]  # 把每个句子的第一个单词(开始的START1)去掉

    src_mask, look_ahead_mask = create_masks(src, trg_inp, pad)  # 计算掩码

    channels = Channels()
    bs = args.batch_size
    snr_min, snr_max = 20.0, 20.0  # 这里我就是为了恢复在20db下的语义特征
    noise_std = np.random.uniform(SNR_to_noise(snr_min), SNR_to_noise(snr_max), size=(1))[0]

    # 修改：Condition 必须是随机的低 SNR (-9 ~ +18)
    current_cond_snr = np.random.uniform(3, 10)
    noise_std_condition = SNR_to_noise(current_cond_snr)
    snr_tensor = torch.full((bs,), current_cond_snr, device=device, dtype=torch.float32)

    key = generate_key(args, src.shape)

    freeze_net(model, False)
    freeze_net(alice_bob_mac, False)
    freeze_net(key_ab, False)
    freeze_net(Alice_KB, False)
    freeze_net(Bob_KB, False)
    freeze_net(Alice_mapping, False)
    freeze_net(Bob_mapping, False)
    freeze_net(cdmodel, True)

    Alice_ID = torch.randn(1, args.d_model).to(device)
    Bob_ID = torch.randn(1, args.d_model).to(device)
    Alice_tmp = Alice_KB(Alice_ID)
    Bob_tmp = Bob_KB(Bob_ID)
    Alice_mapping_tmp = Alice_mapping(Alice_tmp)  # 形状是[8, 128]
    Bob_mapping_tmp = Bob_mapping(Bob_tmp)
    Alice_kb_final = Alice_tmp.repeat(bs, 1, 1)  # 进行复制
    Bob_kb_final = Bob_tmp.repeat(bs, 1, 1)
    Alice_mapping_final = Alice_mapping_tmp.repeat(bs, 1, 1)
    Bob_mapping_final = Bob_mapping_tmp.repeat(bs, 1, 1)

    key_ebd = key_ab(key)  # 生成密钥

    enc_output = model.encoder(src, src_mask, Alice_kb_final, Bob_mapping_final)
    enc_output = enc_output[:, :31, :]  # 只前31个通道 f
    mac = alice_bob_mac.mac_encoder(key_ebd, enc_output, Alice_kb_final, Bob_mapping_final)

    semantic_mac = torch.cat([enc_output, mac], dim=1)

    channel_enc_output = model.channel_encoder(semantic_mac)
    Tx_sig = PowerNormalize(channel_enc_output)  # 功率归一化

    if channel == 'AWGN':
        Rx_sig = channels.AWGN(Tx_sig, noise_std)
        Rx_sig_condition = channels.AWGN(Tx_sig, noise_std_condition)
    elif channel == 'Rayleigh':  # 训练的时候 一般都是走瑞丽信道
        Rx_sig = channels.Rayleigh(Tx_sig, noise_std)
        Rx_sig_condition = channels.Rayleigh(Tx_sig, noise_std_condition)
    elif channel == 'Rician':
        Rx_sig = channels.Rician(Tx_sig, noise_std)
        Rx_sig_condition = channels.Rician(Tx_sig, noise_std_condition)
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")

    memory = model.channel_decoder(Rx_sig)
    memory_condition = model.channel_decoder(Rx_sig_condition)

    # f_p = memory[:, :31, :]  # 前31个通道 发送的时候也是
    # mac_p = memory[:, 31:, :]
    # dec_output = model.decoder(trg_inp, f_p, look_ahead_mask, src_mask, Alice_mapping_final, Bob_kb_final, mac_p)
    # pred = model.dense(dec_output)
    #
    # f_p_condition = memory_condition[:, :31, :]
    # mac_p_condition = memory_condition[:, 31:, :]
    # dec_output_condition = model.decoder(trg_inp, f_p_condition, look_ahead_mask, src_mask, Alice_mapping_final, Bob_kb_final, mac_p_condition)
    # pred_condition = model.dense(dec_output_condition)
    # ntokens = pred.size(-1)

    # 此时 memory 是过 20dB 的干净特征，memory_condition 是过随机低 SNR 的受损特征
    # 它们都包含了 [f_p, mac_p] 的拼接，直接传给扩散模型

    # 训练 DiT 模型
    # cdmodel.train()
    opt_joint.zero_grad()

    timesteps = torch.randint(0, ddim_scheduler.num_train_timesteps, (bs,), device=device).long()
    noise = torch.randn_like(Tx_sig)

    # 给 最开始的Tx_sig加噪
    x_t = ddim_scheduler.add_noise(Tx_sig, noise, timesteps)

    # 10% 的概率丢弃条件 (无分类器引导)
    context_mask = torch.rand(bs, device=device) < 0.1

    # 模型预测噪声
    noise_pred = cdmodel(x_t, Rx_sig_condition, timesteps, snr_tensor, context_mask=context_mask)

    # 计算 MSE Loss 并反向传播
    loss_eps = F.mse_loss(noise_pred, noise)
    loss_eps.backward()
    torch.nn.utils.clip_grad_norm_(cdmodel.parameters(), 1.0)
    opt_joint.step()

    return loss_eps.item()


def val_step(args, batch, model, alice_bob_mac, key_ab, Alice_KB, Bob_KB, Alice_mapping, Bob_mapping, src, trg, n_var,
             pad, channel, cdmodel, ddim_scheduler):
    trg_inp = trg[:, :-1]
    trg_real = trg[:, 1:]

    src_mask, look_ahead_mask = create_masks(src, trg_inp, pad)

    channels = Channels()
    bs = src.size(0)
    snr_min, snr_max = 20.0, 20.0
    noise_std = np.random.uniform(SNR_to_noise(snr_min), SNR_to_noise(snr_max), size=(1))[0]

    current_cond_snr = np.random.uniform(3, 10)
    noise_std_condition = SNR_to_noise(current_cond_snr)
    snr_tensor = torch.full((bs,), current_cond_snr, device=device, dtype=torch.float32)

    key = generate_key(args, src.shape)

    freeze_net(key_ab, False)
    freeze_net(alice_bob_mac, False)
    freeze_net(model, False)
    freeze_net(Alice_KB, False)
    freeze_net(Bob_KB, False)
    freeze_net(Alice_mapping, False)
    freeze_net(Bob_mapping, False)
    # cdmodel.eval()
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
    Alice_tmp = Alice_KB(Alice_ID)
    Bob_tmp = Bob_KB(Bob_ID)
    Alice_mapping_tmp = Alice_mapping(Alice_tmp)
    Bob_mapping_tmp = Bob_mapping(Bob_tmp)
    Alice_kb_final = Alice_tmp.repeat(bs, 1, 1)
    Bob_kb_final = Bob_tmp.repeat(bs, 1, 1)
    Alice_mapping_final = Alice_mapping_tmp.repeat(bs, 1, 1)
    Bob_mapping_final = Bob_mapping_tmp.repeat(bs, 1, 1)

    key_ebd = key_ab(key)

    enc_output = model.encoder(src, src_mask, Alice_kb_final, Bob_mapping_final)
    enc_output = enc_output[:, :31, :]
    mac = alice_bob_mac.mac_encoder(key_ebd, enc_output, Alice_kb_final, Bob_mapping_final)

    semantic_mac = torch.cat([enc_output, mac], dim=1)

    channel_enc_output = model.channel_encoder(semantic_mac)
    Tx_sig = PowerNormalize(channel_enc_output)

    if channel == 'AWGN':
        Rx_sig = channels.AWGN(Tx_sig, noise_std)
        Rx_sig_condition = channels.AWGN(Tx_sig, noise_std_condition)
    elif channel == 'Rayleigh':
        Rx_sig = channels.Rayleigh(Tx_sig, noise_std)
        Rx_sig_condition = channels.Rayleigh(Tx_sig, noise_std_condition)
    elif channel == 'Rician':
        Rx_sig = channels.Rician(Tx_sig, noise_std)
        Rx_sig_condition = channels.Rician(Tx_sig, noise_std_condition)
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")

    memory = model.channel_decoder(Rx_sig)
    memory_condition = model.channel_decoder(Rx_sig_condition)

    # f_p = memory[:, :31, :]  # 前31个通道 发送的时候也是
    # mac_p = memory[:, 31:, :]
    # dec_output = model.decoder(trg_inp, f_p, look_ahead_mask, src_mask, Alice_mapping_final, Bob_kb_final, mac_p)
    # pred = model.dense(dec_output)
    #
    # f_p_condition = memory_condition[:, :31, :]
    # mac_p_condition = memory_condition[:, 31:, :]
    # dec_output_condition = model.decoder(trg_inp, f_p_condition, look_ahead_mask, src_mask, Alice_mapping_final, Bob_kb_final,
    #                                      mac_p_condition)
    # pred_condition = model.dense(dec_output_condition)
    # ntokens = pred.size(-1)

    with torch.no_grad():
        timesteps = torch.randint(0, ddim_scheduler.num_train_timesteps, (bs,), device=device).long()
        noise = torch.randn_like(Tx_sig)
        x_t = ddim_scheduler.add_noise(Tx_sig, noise, timesteps)
        noise_pred = cdmodel(x_t, Rx_sig_condition, timesteps, snr_tensor, context_mask=None)
        loss_eps = F.mse_loss(noise_pred, noise)

    return loss_eps.item()


def greedy_decode(args, deepsc, alice_bob_mac, key_ab, Alice_KB, Bob_KB, Alice_mapping, Bob_mapping, src, noise_std,
                  max_len, pad, start_symbol, channel, cdmodel=None, ddim_scheduler=None, current_snr=0.0):
    trg_inp = src[:, :-1]
    trg_real = src[:, 1:]

    bs = args.batch_size
    src_mask, look_ahead_mask = create_masks(src, trg_inp, pad)

    channels = Channels()
    bs = src.size(0)

    key = generate_key(args, src.shape)

    freeze_net(key_ab, False)
    freeze_net(alice_bob_mac, False)
    freeze_net(deepsc, False)
    freeze_net(Alice_KB, False)
    freeze_net(Bob_KB, False)
    freeze_net(Alice_mapping, False)
    freeze_net(Bob_mapping, False)
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
    Alice_tmp = Alice_KB(Alice_ID)
    Bob_tmp = Bob_KB(Bob_ID)
    Alice_mapping_tmp = Alice_mapping(Alice_tmp)
    Bob_mapping_tmp = Bob_mapping(Bob_tmp)
    Alice_kb_final = Alice_tmp.repeat(bs, 1, 1)
    Bob_kb_final = Bob_tmp.repeat(bs, 1, 1)
    Alice_mapping_final = Alice_mapping_tmp.repeat(bs, 1, 1)
    Bob_mapping_final = Bob_mapping_tmp.repeat(bs, 1, 1)

    key_ebd = key_ab(key)

    enc_output = deepsc.encoder(src, src_mask, Alice_kb_final, Bob_mapping_final)
    enc_output = enc_output[:, :31, :]
    mac = alice_bob_mac.mac_encoder(key_ebd, enc_output, Alice_kb_final, Bob_mapping_final)

    semantic_mac = torch.cat([enc_output, mac], dim=1)

    channel_enc_output = deepsc.channel_encoder(semantic_mac)
    Tx_sig = PowerNormalize(channel_enc_output)

    if channel == 'AWGN':
        Rx_sig = channels.AWGN(Tx_sig, noise_std)
    elif channel == 'Rayleigh':
        Rx_sig = channels.Rayleigh(Tx_sig, noise_std)
    elif channel == 'Rician':
        Rx_sig = channels.Rician(Tx_sig, noise_std)
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")

    memory = deepsc.channel_decoder(Rx_sig)
    # f_p = memory[:, :31, :]  # 前31个通道 发送的时候也是
    # mac_p = memory[:, 31:, :]
    # dec_output = deepsc.decoder(trg_inp, f_p, look_ahead_mask, src_mask, Alice_mapping_final, Bob_kb_final, mac_p)
    # pred = deepsc.dense(dec_output)


    # DDIM 介入修复特征

    if cdmodel is not None and ddim_scheduler is not None:
        snr_tensor = torch.full((bs,), current_snr, device=device, dtype=torch.float32)
        # 极速去噪，此时 DiT 模型内部已经有位置编码(Positional Embedding)的支持了
        # 保持 guidance_scale = 1.0
        Tx_recovered_combined = ddim_scheduler.ddim_sample(
            model=cdmodel,
            f_cond=Rx_sig,
            snr_tensor=snr_tensor,
            num_inference_steps=20,
            guidance_scale=1.0
        )
    else:
        f_recovered_combined = Tx_sig

    # f_huifu = f_recovered_combined[:, :31, :]
    # mac_p_huifu = f_recovered_combined[:, 31:, :]

    memory_huifu = deepsc.channel_decoder(Tx_recovered_combined)
    f_p_huifu = memory_huifu[:, :31, :]
    mac_p_huifu = memory_huifu[:, 31:, :]


    outputs = torch.ones(src.size(0), 1).fill_(start_symbol).type_as(src.data)

    for i in range(max_len - 1):
        trg_mask = (outputs == pad).unsqueeze(-2).type(torch.FloatTensor).to(device)
        look_ahead_mask = subsequent_mask(outputs.size(1)).type(torch.FloatTensor).to(device)

        combined_mask = torch.max(trg_mask, look_ahead_mask)
        combined_mask = combined_mask.to(device)

        dec_output = deepsc.decoder(outputs, f_p_huifu, combined_mask, src_mask, Alice_mapping_final, Bob_kb_final,
                                    mac_p_huifu)
        pred = deepsc.dense(dec_output)

        prob = pred[:, -1:, :]
        _, next_word = torch.max(prob, dim=-1)
        outputs = torch.cat([outputs, next_word], dim=1)

    # outputs = torch.argmax(pred_recovered_combined, dim=-1)  # [batch_size, seq_len]

    return outputs


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

        lr = self.factor * \
             (self.model_size ** (-0.5) *
              min(step ** (-0.5), step * self.warmup ** (-1.5)))

        return lr

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
    src_mask = (src == padding_idx).unsqueeze(-2).type(torch.FloatTensor)  # [batch, 1, seq_len]
    trg_mask = (trg == padding_idx).unsqueeze(-2).type(torch.FloatTensor)  # [batch, 1, seq_len]
    look_ahead_mask = subsequent_mask(trg.size(-1)).type_as(trg_mask.data)
    combined_mask = torch.max(trg_mask, look_ahead_mask)  # 既不能看到填充的部分 也不能看到未来的部分

    return src_mask.to(device), combined_mask.to(device)


criterion = nn.CrossEntropyLoss(reduction='none').to(device)


# 定义损失函数
def loss_function(x, trg, padding_idx):
    loss = criterion(x, trg)  # x与预期的交叉熵
    mask = (trg != padding_idx).type_as(loss.data)  # mask去掉padding的部分
    loss *= mask  # 将padding的部分的loss置为0

    return loss.mean()  # 返回loss的平均值


def PowerNormalize(x):  # 发射前功率归一化
    x_square = torch.mul(x, x)
    power = torch.mean(x_square).sqrt()
    x = torch.div(x, power)

    return x