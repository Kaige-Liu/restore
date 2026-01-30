import json  # 简单测试能不能提交
# 再测试mac能不能提交
# 测试完了win能拉取mac的提交

# 测试mac提交
# 再测试win提交

import torch
from torch import nn
import torch.nn.functional as F

from main import parser
from models.deepsc_MAC import Transmitter, generate_key, Key_net, Receiver
from models.modules import create_masks
from utlis.tools import SeqtoText
from utlis.trainer import subsequent_mask
from utlis.trainer_step import create_mask

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size = 64
length = 31

# 创建一个空的张量
tensor = torch.zeros((batch_size, length), dtype=torch.int32)

# 填充第一个元素为1
tensor[:, 0] = 1

# 填充中间的元素为范围 [2, 127] 的随机整数
tensor[:, 1:-1] = torch.randint(2, 128, (batch_size, length - 2), dtype=torch.int32)

# 最后一个元素已经是0，不需要显式填充

inp = tensor


def train_step(inp, tar, alice, bob, key_ab, n_std, optim_joint, args, batch):
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


args = parser.parse_args()
with open(args.vocab_path, 'r') as f:  # 注意这里使用'r'而不是'rb'，因为json.load默认读取文本
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

alice = Transmitter(args)
bob = Receiver(args)

key_ab = Key_net(args)

received_channel_enc_output, channel_enc_output, semantic_mac, pre_b, result = train_step(inp, inp, alice, bob, key_ab, 0.1, None, args, 0)

print()
# 收到的received_channel_enc_output是[bs, 32, 16]








# class Channel_Decoder(nn.Module):
#     def __init__(self, size1=128, size2=512):  # size1=128, size2=512
#         super(Channel_Decoder, self).__init__()
#         self.dense1 = nn.Linear(16, size1)  # 信道完了就是[bs, 30, 16]
#         self.dense2 = nn.Linear(size1, size2)
#         self.dense3 = nn.Linear(size2, size1)
#         self.layernorm1 = nn.LayerNorm(size1, eps=1e-6)  # 表示
#
#     def forward(self, receives):
#         x1 = F.relu(self.dense1(receives))
#         x2 = F.relu(self.dense2(x1))
#         x3 = self.dense3(x2)
#         output = self.layernorm1(x1 + x3)
#         return output
#
# # 生成随机的[bs, 30, 128]的张量
# inputs = torch.randn(2, 30, 16)
#
# net = Channel_Decoder()
# outputs = net(inputs)
# print(outputs.shape)  # torch.Size([2, 30, 16])
