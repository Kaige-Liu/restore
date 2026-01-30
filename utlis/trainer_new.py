import math
import torch
import torch.nn as nn
import numpy as np
from models.mutual_info import sample_batch, mutual_information
from utlis.tools import l1_norm, gram_for_batch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_step(model, src_hiding, trg_hiding, src_cover, trg_cover, n_var, pad, opt_model, channel, lambda_1, lambda_2, lambda_3, lambda_4, mi_net=None):  # 模型，发送的128个句子，发送的128个句子，噪声标准差(类型数字)，数字0，deepsc优化器，信道类型
    model.train()

    trg_hiding_inp = trg_hiding[:, :-1]  # 把每个句子的最后一个单词(填充的PAD0或END2)去掉
    trg_hiding_real = trg_hiding[:, 1:]  # 把每个句子的第一个单词(开始的START1)去掉

    trg_cover_inp = trg_cover[:, :-1]
    trg_cover_real = trg_cover[:, 1:]

    channels = Channels()
    opt_model.zero_grad()  # 梯度清零，常规操作，不用管

    src_hiding_mask, hiding_look_ahead_mask = create_masks(src_hiding, trg_hiding_inp, pad)
    src_cover_mask, cover_look_ahead_mask = create_masks(src_cover, trg_cover_inp, pad)

    sc_hiding = model.encoder(src_hiding, src_hiding_mask)
    sc_hiding = sc_hiding.unsqueeze(1)  # [128, 1, 31, 128]
    sc_cover = model.encoder(src_cover, src_cover_mask)
    sc_cover = sc_cover.unsqueeze(1)  # [128, 1, 31, 128]

    encoder_hiding_output = model.encoder_hiding(sc_hiding)  # [128, 48, 31, 128]
    embedder_hiding_output = model.embedder_hiding(encoder_hiding_output, sc_cover)
    invarince_output, weights = model.invariance(embedder_hiding_output)
    # 信道编码
    channel_enc_output = model.channel_encoder(invarince_output)
    Tx_sig = PowerNormalize(channel_enc_output)

    if channel == 'AWGN':
        Rx_sig = channels.AWGN(Tx_sig, n_var)
    elif channel == 'Rayleigh':
        Rx_sig = channels.Rayleigh(Tx_sig, n_var)
    elif channel == 'Rician':
        Rx_sig = channels.Rician(Tx_sig, n_var)
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")

    channel_dec_output = model.channel_decoder(Rx_sig)
    extractor_output = model.extractor(channel_dec_output)  # [bs, 1, 31, 128]
    cover_sc_recover = extractor_output.squeeze(1)  # [bs, 31, 128]
    # 下面对cover_sc_recover输入到net.decoder中
    dec_cover_output = model.decoder(trg_cover_inp, cover_sc_recover, cover_look_ahead_mask, src_cover_mask)
    pred_cover = model.dense(dec_cover_output)
    ntokens_cover = pred_cover.size(-1)  # 预测结果的最后一个维度的大小，即词汇表大小  22234

    # 下面对下面对extractor_output输入到hiding.decoder中
    decoder_hiding_output = model.decoder_hiding(extractor_output, extractor_output.size(0))  # [bs, 1, 31, 128]
    hiding_sc_recover = decoder_hiding_output.squeeze(1)  # [bs, 31, 128]
    dec_hiding_output = model.decoder(trg_hiding_inp, hiding_sc_recover, hiding_look_ahead_mask, src_hiding_mask)
    pre_hiding = model.dense(dec_hiding_output)
    ntokens_hiding = pre_hiding.size(-1)

    # y_est = x +  torch.matmul(n, torch.inverse(H))
    # loss1 = torch.mean(torch.pow((x_est - y_est.view(x_est.shape)), 2))

    loss_1 = loss_function(pred_cover.contiguous().view(-1, ntokens_cover), trg_cover_real.contiguous().view(-1), pad)
    loss_2 = loss_function(pre_hiding.contiguous().view(-1, ntokens_hiding), trg_hiding_real.contiguous().view(-1), pad)
    # .contigous()是为了保证张量在内存中是连续的，.view(-1, ntokens)是将张量变成二维的，第一维是batch_size*seq_len，seq_len是词的个数,第二维是词汇表大小(因为计算交叉熵只能计算二维的)


    loss_hiding_1 = lambda_1 * l1_norm(decoder_hiding_output, sc_hiding)  # 隐藏文本的第一范数差,都是4维
    loss_hiding_2 = lambda_2 * l1_norm(embedder_hiding_output, sc_cover)
    b1w, b2w, _ = model.embedder_hiding.convBlock_star(encoder_hiding_output)
    # 使用全连接将embedder_hiding_output映射到48维
    embedder_hiding_output_48 = model.tmp(embedder_hiding_output)
    b1m, b2m, _ = model.embedder_hiding.convBlock_star(embedder_hiding_output_48)
    g1w = gram_for_batch(b1w)
    g1m = gram_for_batch(b1m)
    g2w = gram_for_batch(b2w)
    g2m = gram_for_batch(b2m)
    loss_hiding_3 = lambda_3 * 0.5 * (l1_norm(g1w, g1m) + l1_norm(g2w, g2m))

    # 对weights中的所有元素平方求和
    loss_hiding_4 = lambda_4 * torch.sum(weights ** 2)

    # 前面这个1应该设计成参数才对 同样后面也是
    loss = 100 * (loss_1 + loss_2) + 1 * (loss_hiding_1 + loss_hiding_2 + loss_hiding_3 + loss_hiding_4)

    loss.backward()
    opt_model.step()

    return loss.item(), loss_1.item(), loss_2.item()

def freeze_net(net, is_requires_grad):
    for param in net.parameters():
        param.requires_grad = is_requires_grad
    if is_requires_grad:
        net.train()
    else:
        net.eval()

def train_deepsc(model, src_hiding, trg_hiding, src_cover, trg_cover, n_var, pad, opt_deepsc, channel, lambda_1, lambda_2, lambda_3, lambda_4, mi_net=None):  # 模型，发送的128个句子，发送的128个句子，噪声标准差(类型数字)，数字0，deepsc优化器，信道类型
    freeze_net(model.deepsc, True)  # 训练deepsc
    freeze_net(model.hiding, False)  # 冻结hiding

    trg_hiding_inp = trg_hiding[:, :-1]  # 把每个句子的最后一个单词(填充的PAD0或END2)去掉
    trg_hiding_real = trg_hiding[:, 1:]  # 把每个句子的第一个单词(开始的START1)去掉

    trg_cover_inp = trg_cover[:, :-1]
    trg_cover_real = trg_cover[:, 1:]

    channels = Channels()
    opt_deepsc.zero_grad()  # 梯度清零，常规操作，不用管

    src_hiding_mask, hiding_look_ahead_mask = create_masks(src_hiding, trg_hiding_inp, pad)
    src_cover_mask, cover_look_ahead_mask = create_masks(src_cover, trg_cover_inp, pad)

    sc_hiding = model.deepsc.encoder(src_hiding, src_hiding_mask)
    sc_hiding = sc_hiding.unsqueeze(1)  # [128, 1, 31, 128]
    sc_cover = model.deepsc.encoder(src_cover, src_cover_mask)
    sc_cover = sc_cover.unsqueeze(1)  # [128, 1, 31, 128]

    encoder_hiding_output = model.hiding.encoder_hiding(sc_hiding)  # [128, 48, 31, 128]
    embedder_hiding_output = model.hiding.embedder_hiding(encoder_hiding_output, sc_cover)
    invarince_output, weights = model.hiding.invariance(embedder_hiding_output)
    # 信道编码
    channel_enc_output = model.deepsc.channel_encoder(invarince_output)
    Tx_sig = PowerNormalize(channel_enc_output)

    if channel == 'AWGN':
        Rx_sig = channels.AWGN(Tx_sig, n_var)
    elif channel == 'Rayleigh':
        Rx_sig = channels.Rayleigh(Tx_sig, n_var)
    elif channel == 'Rician':
        Rx_sig = channels.Rician(Tx_sig, n_var)
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")

    channel_dec_output = model.deepsc.channel_decoder(Rx_sig)
    extractor_output = model.hiding.extractor(channel_dec_output)  # [bs, 1, 31, 128]
    cover_sc_recover = extractor_output.squeeze(1)  # [bs, 31, 128]
    # 下面对cover_sc_recover输入到net.decoder中
    dec_cover_output = model.deepsc.decoder(trg_cover_inp, cover_sc_recover, cover_look_ahead_mask, src_cover_mask)
    pred_cover = model.deepsc.dense(dec_cover_output)
    ntokens_cover = pred_cover.size(-1)  # 预测结果的最后一个维度的大小，即词汇表大小  22234

    # 下面对下面对extractor_output输入到hiding.decoder中
    decoder_hiding_output = model.hiding.decoder_hiding(extractor_output, extractor_output.size(0))  # [bs, 1, 31, 128]
    hiding_sc_recover = decoder_hiding_output.squeeze(1)  # [bs, 31, 128]
    dec_hiding_output = model.deepsc.decoder(trg_hiding_inp, hiding_sc_recover, hiding_look_ahead_mask, src_hiding_mask)
    pre_hiding = model.deepsc.dense(dec_hiding_output)
    ntokens_hiding = pre_hiding.size(-1)

    # y_est = x +  torch.matmul(n, torch.inverse(H))
    # loss1 = torch.mean(torch.pow((x_est - y_est.view(x_est.shape)), 2))

    loss_1 = loss_function(pred_cover.contiguous().view(-1, ntokens_cover), trg_cover_real.contiguous().view(-1), pad)
    loss_2 = loss_function(pre_hiding.contiguous().view(-1, ntokens_hiding), trg_hiding_real.contiguous().view(-1), pad)

    # 前面这个1应该设计成参数才对 同样后面也是
    loss = 100 * (loss_1 + loss_2)

    loss.backward()
    opt_deepsc.step()

    return loss.item()

def train_hiding(model, src_hiding, trg_hiding, src_cover, trg_cover, n_var, pad, opt_hiding, channel, lambda_1, lambda_2, lambda_3, lambda_4, mi_net=None):  # 模型，发送的128个句子，发送的128个句子，噪声标准差(类型数字)，数字0，deepsc优化器，信道类型
    model.train()

    trg_hiding_inp = trg_hiding[:, :-1]  # 把每个句子的最后一个单词(填充的PAD0或END2)去掉
    trg_hiding_real = trg_hiding[:, 1:]  # 把每个句子的第一个单词(开始的START1)去掉

    trg_cover_inp = trg_cover[:, :-1]
    trg_cover_real = trg_cover[:, 1:]

    channels = Channels()
    opt_model.zero_grad()  # 梯度清零，常规操作，不用管

    src_hiding_mask, hiding_look_ahead_mask = create_masks(src_hiding, trg_hiding_inp, pad)
    src_cover_mask, cover_look_ahead_mask = create_masks(src_cover, trg_cover_inp, pad)

    sc_hiding = model.encoder(src_hiding, src_hiding_mask)
    sc_hiding = sc_hiding.unsqueeze(1)  # [128, 1, 31, 128]
    sc_cover = model.encoder(src_cover, src_cover_mask)
    sc_cover = sc_cover.unsqueeze(1)  # [128, 1, 31, 128]

    encoder_hiding_output = model.encoder_hiding(sc_hiding)  # [128, 48, 31, 128]
    embedder_hiding_output = model.embedder_hiding(encoder_hiding_output, sc_cover)
    invarince_output, weights = model.invariance(embedder_hiding_output)
    # 信道编码
    channel_enc_output = model.channel_encoder(invarince_output)
    Tx_sig = PowerNormalize(channel_enc_output)

    if channel == 'AWGN':
        Rx_sig = channels.AWGN(Tx_sig, n_var)
    elif channel == 'Rayleigh':
        Rx_sig = channels.Rayleigh(Tx_sig, n_var)
    elif channel == 'Rician':
        Rx_sig = channels.Rician(Tx_sig, n_var)
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")

    channel_dec_output = model.channel_decoder(Rx_sig)
    extractor_output = model.extractor(channel_dec_output)  # [bs, 1, 31, 128]
    cover_sc_recover = extractor_output.squeeze(1)  # [bs, 31, 128]
    # 下面对cover_sc_recover输入到net.decoder中
    dec_cover_output = model.decoder(trg_cover_inp, cover_sc_recover, cover_look_ahead_mask, src_cover_mask)
    pred_cover = model.dense(dec_cover_output)
    ntokens_cover = pred_cover.size(-1)  # 预测结果的最后一个维度的大小，即词汇表大小  22234

    # 下面对下面对extractor_output输入到hiding.decoder中
    decoder_hiding_output = model.decoder_hiding(extractor_output, extractor_output.size(0))  # [bs, 1, 31, 128]
    hiding_sc_recover = decoder_hiding_output.squeeze(1)  # [bs, 31, 128]
    dec_hiding_output = model.decoder(trg_hiding_inp, hiding_sc_recover, hiding_look_ahead_mask, src_hiding_mask)
    pre_hiding = model.dense(dec_hiding_output)
    ntokens_hiding = pre_hiding.size(-1)

    # y_est = x +  torch.matmul(n, torch.inverse(H))
    # loss1 = torch.mean(torch.pow((x_est - y_est.view(x_est.shape)), 2))

    loss_1 = loss_function(pred_cover.contiguous().view(-1, ntokens_cover), trg_cover_real.contiguous().view(-1), pad)
    loss_2 = loss_function(pre_hiding.contiguous().view(-1, ntokens_hiding), trg_hiding_real.contiguous().view(-1), pad)
    # .contigous()是为了保证张量在内存中是连续的，.view(-1, ntokens)是将张量变成二维的，第一维是batch_size*seq_len，seq_len是词的个数,第二维是词汇表大小(因为计算交叉熵只能计算二维的)


    loss_hiding_1 = lambda_1 * l1_norm(decoder_hiding_output, sc_hiding)  # 隐藏文本的第一范数差,都是4维
    loss_hiding_2 = lambda_2 * l1_norm(embedder_hiding_output, sc_cover)
    b1w, b2w, _ = model.embedder_hiding.convBlock_star(encoder_hiding_output)
    # 使用全连接将embedder_hiding_output映射到48维
    embedder_hiding_output_48 = model.tmp(embedder_hiding_output)
    b1m, b2m, _ = model.embedder_hiding.convBlock_star(embedder_hiding_output_48)
    g1w = gram_for_batch(b1w)
    g1m = gram_for_batch(b1m)
    g2w = gram_for_batch(b2w)
    g2m = gram_for_batch(b2m)
    loss_hiding_3 = lambda_3 * 0.5 * (l1_norm(g1w, g1m) + l1_norm(g2w, g2m))

    # 对weights中的所有元素平方求和
    loss_hiding_4 = lambda_4 * torch.sum(weights ** 2)

    # 前面这个1应该设计成参数才对 同样后面也是
    loss = 100 * (loss_1 + loss_2) + 1 * (loss_hiding_1 + loss_hiding_2 + loss_hiding_3 + loss_hiding_4)

    loss.backward()
    opt_model.step()

    return loss.item()



def val_step(model, src_hiding, trg_hiding, src_cover, trg_cover, n_var, pad, channel, lambda_1, lambda_2, lambda_3, lambda_4):  # 参数模型，发送的128个句子，发送的128个句子，噪声标准差(数字0.1)，数字0，信道类型
    trg_hiding_inp = trg_hiding[:, :-1]  # 把每个句子的最后一个单词(填充的PAD0或END2)去掉
    trg_hiding_real = trg_hiding[:, 1:]  # 把每个句子的第一个单词(开始的START1)去掉
    trg_cover_inp = trg_cover[:, :-1]
    trg_cover_real = trg_cover[:, 1:]

    channels = Channels()

    src_hiding_mask, hiding_look_ahead_mask = create_masks(src_hiding, trg_hiding_inp, pad)
    src_cover_mask, cover_look_ahead_mask = create_masks(src_cover, trg_cover_inp, pad)

    sc_hiding = model.encoder(src_hiding, src_hiding_mask)
    sc_hiding = sc_hiding.unsqueeze(1)  # [128, 1, 31, 128]
    sc_cover = model.encoder(src_cover, src_cover_mask)
    sc_cover = sc_cover.unsqueeze(1)  # [128, 1, 31, 128]

    encoder_hiding_output = model.encoder_hiding(sc_hiding)  # [128, 48, 31, 128]
    embedder_hiding_output = model.embedder_hiding(encoder_hiding_output, sc_cover)
    invarince_output, weights = model.invariance(embedder_hiding_output)
    # 信道编码
    channel_enc_output = model.channel_encoder(invarince_output)
    Tx_sig = PowerNormalize(channel_enc_output)

    if channel == 'AWGN':
        Rx_sig = channels.AWGN(Tx_sig, n_var)
    elif channel == 'Rayleigh':
        Rx_sig = channels.Rayleigh(Tx_sig, n_var)
    elif channel == 'Rician':
        Rx_sig = channels.Rician(Tx_sig, n_var)
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")

    channel_dec_output = model.channel_decoder(Rx_sig)
    extractor_output = model.extractor(channel_dec_output)
    cover_sc_recover = extractor_output.squeeze(1)  # [bs, 31, 128]

    # 下面对cover_sc_recover输入到net.decoder中
    dec_cover_output = model.decoder(trg_cover_inp, cover_sc_recover, cover_look_ahead_mask, src_cover_mask)
    pred_cover = model.dense(dec_cover_output)
    ntokens_cover = pred_cover.size(-1)  # 预测结果的最后一个维度的大小，即词汇表大小  22234

    # 下面对下面对extractor_output输入到hiding.decoder中
    decoder_hiding_output = model.decoder_hiding(extractor_output, extractor_output.size(0))
    hiding_sc_recover = decoder_hiding_output.squeeze(1)  # [bs, 31, 128]
    dec_hiding_output = model.decoder(trg_hiding_inp, hiding_sc_recover, hiding_look_ahead_mask, src_hiding_mask)
    pre_hiding = model.dense(dec_hiding_output)
    ntokens_hiding = pre_hiding.size(-1)


    loss_1 = loss_function(pred_cover.contiguous().view(-1, ntokens_cover), trg_cover_real.contiguous().view(-1), pad)
    loss_2 = loss_function(pre_hiding.contiguous().view(-1, ntokens_hiding), trg_hiding_real.contiguous().view(-1), pad)

    loss_hiding_1 = lambda_1 * l1_norm(decoder_hiding_output, sc_hiding)  # 隐藏文本的第一范数差
    loss_hiding_2 = lambda_2 * l1_norm(embedder_hiding_output, sc_cover)
    b1w, b2w, _ = model.embedder_hiding.convBlock_star(encoder_hiding_output)
    # 使用全连接将embedder_hiding_output映射到48维
    embedder_hiding_output_48 = model.tmp(embedder_hiding_output)
    b1m, b2m, _ = model.embedder_hiding.convBlock_star(embedder_hiding_output_48)
    g1w = gram_for_batch(b1w)
    g1m = gram_for_batch(b1m)
    g2w = gram_for_batch(b2w)
    g2m = gram_for_batch(b2m)
    loss_hiding_3 = lambda_3 * 0.5 * (l1_norm(g1w, g1m) + l1_norm(g2w, g2m))

    # 对weights中的所有元素平方求和
    loss_hiding_4 = lambda_4 * torch.sum(weights ** 2)

    # 前面这个1应该设计成参数才对 同样后面也是
    loss = 100 * (loss_1 + loss_2) + 1 * (loss_hiding_1 + loss_hiding_2 + loss_hiding_3 + loss_hiding_4)

    return loss.item(), loss_1.item(), loss_2.item()

def greedy_decode(model, hiding_data, cover_data, n_var, max_len, padding_idx, start_symbol, channel):
    """
    这里采用贪婪解码器，如果需要更好的性能情况下，可以使用beam search decode
    """
    # create src_mask
    channels = Channels()
    src_hiding_mask = (hiding_data == padding_idx).unsqueeze(-2).type(torch.FloatTensor).to(device)  # [batch, 1, seq_len]
    src_cover_mask = (cover_data == padding_idx).unsqueeze(-2).type(torch.FloatTensor).to(device)

    sc_hiding = model.encoder(hiding_data, src_hiding_mask)
    sc_hiding = sc_hiding.unsqueeze(1)  # [128, 1, 31, 128]
    sc_cover = model.encoder(cover_data, src_cover_mask)
    sc_cover = sc_cover.unsqueeze(1)  # [128, 1, 31, 128]

    encoder_hiding_output = model.encoder_hiding(sc_hiding)  # [128, 48, 31, 128]
    embedder_hiding_output = model.embedder_hiding(encoder_hiding_output, sc_cover)
    invarince_output, weights = model.invariance(embedder_hiding_output)
    # 信道编码
    channel_enc_output = model.channel_encoder(invarince_output)
    Tx_sig = PowerNormalize(channel_enc_output)

    if channel == 'AWGN':
        Rx_sig = channels.AWGN(Tx_sig, n_var)
    elif channel == 'Rayleigh':
        Rx_sig = channels.Rayleigh(Tx_sig, n_var)
    elif channel == 'Rician':
        Rx_sig = channels.Rician(Tx_sig, n_var)
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")







    # 我现在严重怀疑就是下面这些写错了
    channel_dec_output = model.channel_decoder(Rx_sig)  # 那个memory，但原来那个是语义，这个应该就是得到的全连接层的信息，下面需要提取载体和隐藏的语义
    extractor_output = model.extractor(channel_dec_output)  # 载体的语义，应该对照着源代码的memory的内容，即语义
    cover_sc_recover = extractor_output.squeeze(1)  # [bs, 31, 128]







    # # 将extractor_output输入到net.decoder中
    # dec_cover_output = model.decoder

    # channel_enc_output = model.blind_csi(channel_enc_output)

    # memory = model.channel_decoder(Rx_sig)  # 这是原本的语义

    hiding_outputs = torch.ones(hiding_data.size(0), 1).fill_(start_symbol).type_as(hiding_data.data)
    cover_outputs = torch.ones(cover_data.size(0), 1).fill_(start_symbol).type_as(cover_data.data)


    # 首先解码cover文本
    for i in range(max_len - 1):
        # create the decode mask
        trg_mask = (cover_outputs == padding_idx).unsqueeze(-2).type(torch.FloatTensor)  # [batch, 1, seq_len]
        look_ahead_mask = subsequent_mask(cover_outputs.size(1)).type(torch.FloatTensor)
        #        print(look_ahead_mask)
        combined_mask = torch.max(trg_mask, look_ahead_mask)
        combined_mask = combined_mask.to(device)

        # decode the received signal
        cover_dec_output = model.decoder(cover_outputs, cover_sc_recover, combined_mask, None)
        cover_pred = model.dense(cover_dec_output)

        # predict the word
        cover_prob = cover_pred[:, -1:, :]  # (batch_size, 维度， 1, vocab_size)
        # prob = prob.squeeze()

        # return the max-prob index
        _, next_word = torch.max(cover_prob, dim=-1)
        # next_word = next_word.unsqueeze(1)

        # next_word = next_word.data[0]
        cover_outputs = torch.cat([cover_outputs, next_word], dim=1)

    # 解码hiding_data之前先经过hiding的decoder
    decoder_hiding_output = model.decoder_hiding(extractor_output, extractor_output.size(0))
    hiding_sc_recover = decoder_hiding_output.squeeze(1)  # [bs, 31, 128]
    for i in range(max_len - 1):
        # create the decode mask
        trg_mask = (hiding_outputs == padding_idx).unsqueeze(-2).type(torch.FloatTensor)  # [batch, 1, seq_len]
        look_ahead_mask = subsequent_mask(hiding_outputs.size(1)).type(torch.FloatTensor)
        #        print(look_ahead_mask)
        combined_mask = torch.max(trg_mask, look_ahead_mask)
        combined_mask = combined_mask.to(device)

        # decode the received signal
        hiding_dec_output = model.decoder(hiding_outputs, hiding_sc_recover, combined_mask, None)
        hiding_pred = model.dense(hiding_dec_output)

        # predict the word
        hiding_prob = hiding_pred[:, -1:, :]  # (batch_size, 维度， 1, vocab_size)
        # prob = prob.squeeze()

        # return the max-prob index
        _, next_word = torch.max(hiding_prob, dim=-1)
        # next_word = next_word.unsqueeze(1)

        # next_word = next_word.data[0]
        hiding_outputs = torch.cat([hiding_outputs, next_word], dim=1)

    return hiding_outputs, cover_outputs



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
        Rx_sig = Tx_sig + torch.normal(0, n_var, size=Tx_sig.shape).to(device)
        return Rx_sig  # 返回接收信号

    def Rayleigh(self, Tx_sig, n_var):
        shape = Tx_sig.shape
        H_real = torch.normal(0, math.sqrt(1 / 2), size=[1]).to(device)
        H_imag = torch.normal(0, math.sqrt(1 / 2), size=[1]).to(device)
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
    combined_mask = torch.max(trg_mask, look_ahead_mask)
    # print("combined_mask: ", combined_mask)  # 1x30x30，由30个30x30的矩阵构成

    return src_mask.to(device), combined_mask.to(device)



criterion = nn.CrossEntropyLoss(reduction='none')

# 定义损失函数
def loss_function(x, trg, padding_idx):
    loss = criterion(x, trg)  # x与预期的交叉熵
    mask = (trg != padding_idx).type_as(loss.data)  # mask去掉padding的部分
    loss *= mask  # 将padding的部分的loss置为0，因为我们通常会使用填充标记来对齐不同长度的序列，但是这些填充部分不应该对损失产生影响

    return loss.mean()  # 返回loss的平均值


def PowerNormalize(x):
    x_square = torch.mul(x, x)
    power = torch.mean(x_square).sqrt()
    if power > 1:
        x = torch.div(x, power)

    return x