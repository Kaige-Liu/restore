import torch

from models.deepsc_MAC import Transmitter, Receiver, Key_net, generate_key
from models.modules import create_masks
from utlis.trainer import subsequent_mask

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def create_mask(src, trg, padding_idx):  # 输入的128个句子，输入的128个去掉最后一个单词的句子，数字0
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

    return src_mask, combined_mask


# def train_step(inp, tar, alice: Transmitter, bob: Receiver, eve: Receiver, key_ab: Key_net, key_e: Key_net,
#                n_std, optim_joint,
#                optim_e, args, batch):
#     tar_inp = tar[:, :-1]
#     tar_real = tar[:, 1:]
#     enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
#     src_mask, look_ahead_mask = create_mask(inp, tar_inp, 0)
#     key = generate_key(args, inp.shape)
#     key_wrong = generate_key(args, inp.shape)


def train_step_mac(inp, tar, alice, bob, key_ab, n_std, optim_joint, args, batch):
    tar_inp = inp[:, :-1]
    tar_real = inp[:, 1:]


    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
    src_mask, look_ahead_mask = create_mask(inp, tar_inp, 0)


    key = generate_key(args, inp.shape)
    key_em = key_ab(key)

    received_channel_enc_output, channel_enc_output, semantic_mac = alice(inp,
                                                                                        key_em,
                                                                                        n_std,
                                                                                        True,
                                                                                        enc_padding_mask)
    pred_b, result = bob(received_channel_enc_output, tar_inp, key_em, look_ahead_mask, src_mask, True, combined_mask)



    return received_channel_enc_output, channel_enc_output, semantic_mac, pred_b, result