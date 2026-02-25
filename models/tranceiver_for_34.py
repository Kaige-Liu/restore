# -*- coding: utf-8 -*-
"""
Transformer includes:
    Encoder
        1. Positional coding
        2. Multihead-attention
        3. PositionwiseFeedForward
    Decoder
        1. Positional coding
        2. Multihead-attention
        3. Multihead-attention
        4. PositionwiseFeedForward
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import math

# from models.text_hiding import Encoder_hiding, Embedder_hiding, CustomLinearLayer, Extractor, Decoder_hiding

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 定义Key_net类
class Key_net(nn.Module):
    def __init__(self, args):
        super(Key_net, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=args.vocab_size, embedding_dim=args.encoder_d_model)

    def forward(self, key):  # 输入[bs, 10] 输出[bs, 10, 128]
        key_ebd = self.embedding(key)
        return key_ebd.to(device)


# class MAC_generate(nn.Module):
#     def __init__(self):  # 输入三个张量，分别是[bs, 10, 128] [ns, 31, 128] [bs, 8, 128] [bs, 8, 128]，输出是[bs, 1, 128]
#         super(MAC_generate, self).__init__()
#         # 定义第一个输入的卷积层（密钥）
#         self.conv1 = nn.Conv1d(in_channels=10, out_channels=16, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm1d(16)
#
#         # 定义第二个输入的卷积层（特征）
#         self.conv2 = nn.Conv1d(in_channels=31, out_channels=16, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm1d(16)
#
#         # 定义第三个输入的卷积层（alice知识库）
#         self.conv3 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
#         self.bn3 = nn.BatchNorm1d(16)
#
#         # 定义第四个输入的卷积层（bob知识库映射）
#         self.conv4 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
#         self.bn4 = nn.BatchNorm1d(16)
#
#         # 定义融合后的卷积层
#         self.conv5 = nn.Conv1d(in_channels=64, out_channels=16, kernel_size=3, padding=1)
#         self.bn5 = nn.BatchNorm1d(16)
#
#         # 定义输出层
#         self.conv6 = nn.Conv1d(in_channels=16, out_channels=1, kernel_size=3, padding=1)
#         self.bn6 = nn.BatchNorm1d(1)
#
#     def forward(self, x1, x2, x3, x4):  # x1是秘钥，x2是特征， x3是alice知识库，x4是bob知识库映射
#         # 密钥的前向传播
#         x1 = self.conv1(x1)
#         x1 = self.bn1(x1)
#         x1 = F.relu(x1)
#
#         # 特征的前向传播
#         x2 = self.conv2(x2)
#         x2 = self.bn2(x2)
#         x2 = F.relu(x2)
#
#         # alice知识库的前向传播
#         x3 = self.conv3(x3)
#         x3 = self.bn3(x3)
#         x3 = F.relu(x3)
#
#         x4 = self.conv4(x4)
#         x4 = self.bn4(x4)
#         x4 = F.relu(x4)
#
#         # 将两个输入在通道维度上拼接
#         x = torch.cat((x1, x2, x3, x4), dim=1)
#
#         # 融合后的前向传播
#         x = self.conv5(x)
#         x = self.bn5(x)
#         x = F.relu(x)
#
#         # 输出层前向传播
#         x = self.conv6(x)
#         x = self.bn6(x)
#
#         return x

# class MAC_verify(nn.Module):  # 输入是[bs, 1, 128] [bs, 31, 128] [bs, 10, 128] [bs, 8, 128] [bs, 8, 128] 输出是[bs, 2]
#     def __init__(self):
#         super(MAC_verify, self).__init__()
#         self.fc1 = nn.Linear(1 * 128, 128)  # 处理mac
#         self.fc2 = nn.Linear(31 * 128, 128)  # 处理f
#         self.fc3 = nn.Linear(10 * 128, 128)  # 处理key_bed
#         self.fc_alice = nn.Linear(8 * 128, 128)  # 处理alice知识库
#         self.fc_bob = nn.Linear(8 * 128, 128)  # 处理bob知识库映射
#         self.fc4 = nn.Linear(128 * 5, 128)  # 合并后的全连接层
#         self.fc5 = nn.Linear(128, 1)  # 输出层，sigmoid二分类 输出维度是1
#
#     def forward(self, x1, x2, x3, x4, x5):  # mac,f,key_ebd 输出为[bs, 2] 即概率
#         x1 = x1.view(-1, 1 * 128)  # 展平mac
#         x2 = x2.view(-1, 31 * 128)  # 展平f
#         x3 = x3.view(-1, 10 * 128)  # 展平key_ebd
#         x4 = x4.view(-1, 8 * 128)  # 展平alice知识库
#         x5 = x5.view(-1, 8 * 128)   # 展平bob知识库映射
#
#         x1 = torch.relu(self.fc1(x1))  # 第一层激活函数
#         x2 = torch.relu(self.fc2(x2))  # 第二层激活函数
#         x3 = torch.relu(self.fc3(x3))  # 第三层激活函数
#         x4 = torch.relu(self.fc_alice(x4))  # 第四层激活函数
#         x5 = torch.relu(self.fc_bob(x5))  # 第五层激活函数
#
#         x = torch.cat((x1, x2, x3, x4, x5), 1)  # 将5个输入合并
#
#         x = torch.relu(self.fc4(x))  # 第三层激活函数
#         x = torch.sigmoid(self.fc5(x))  # 输出层激活函数，输出概率
#         # x = self.fc5(x)
#         # prob = F.softmax(x, dim=1)  # 输出归一化概率 形状为[bs, 2] 之后计算loss用
#         return x

class BLEU_predictor(nn.Module):  # 输入是[bs, 1, 128] [bs, 31, 128] [bs, 10, 128] [bs, 8, 128] [bs, 8, 128] 输出是[bs, 1]
    def __init__(self):
        super(BLEU_predictor, self).__init__()
        self.fc1 = nn.Linear(1 * 128, 128)  # 处理mac
        self.fc2 = nn.Linear(31 * 128, 128)  # 处理f
        self.fc_alice = nn.Linear(8 * 128, 128)  # 处理alice知识库映射
        self.fc_bob = nn.Linear(8 * 128, 128)  # 处理bob知识库
        self.fc3 = nn.Linear(128 * 4, 128)  # 合并后的全连接层
        self.fc4 = nn.Linear(128, 64)  # 过渡层，提升表达能力
        self.fc5 = nn.Linear(64, 1)  # 输出层，sigmoid二分类 输出维度是1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2, x3, x4):  # mac,f，两个知识库和映射 输出为[bs, 2] 即概率
        x1 = x1.view(-1, 1 * 128)  # 展平mac
        x2 = x2.view(-1, 31 * 128)  # 展平f
        x3 = x3.view(-1, 8 * 128)  # 展平alice知识库映射
        x4 = x4.view(-1, 8 * 128)  # 展平bob知识库

        x1 = torch.relu(self.fc1(x1))  # 第一层激活函数
        x2 = torch.relu(self.fc2(x2))  # 第二层激活函数
        x3 = torch.relu(self.fc_alice(x3))  # 第3层激活函数
        x4 = torch.relu(self.fc_bob(x4))  # 第4层激活函数

        x = torch.cat((x1, x2, x3, x4), 1)  # 将4个输入合并

        x = torch.relu(self.fc3(x))  # 第三层激活函数
        x = torch.relu(self.fc4(x))  # 第四层激活函数
        x = self.sigmoid(self.fc5(x))  # 输出层激活函数，输出BLEU值
        return x


class KnowledgeBase_old(nn.Module):
    def __init__(self):
        super(KnowledgeBase_old, self).__init__()  # 输入是[bs, 1, 128] 输出是[bs, 8, 128]
        # 第一个全连接层，输入特征为128，输出特征为256
        self.fc1 = nn.Linear(128, 256)
        # 第二个全连接层，输入特征为256，输出特征为8*128
        self.fc2 = nn.Linear(256, 8 * 128)

    def forward(self, x):
        # 应用第一个全连接层并使用ReLU激活函数
        x = F.relu(self.fc1(x))
        # 应用第二个全连接层
        x = self.fc2(x)
        # 重塑成[bs, 8, 128]
        x = x.view(-1, 8, 128)
        return x


# class KnowledgeBase(nn.Module):
#     def __init__(self):
#         super(KnowledgeBase, self).__init__()  # 输入是[1, 128] 输出是[8, 128]
#         # 第一个全连接层，输入特征为128，输出特征为256
#         self.fc1 = nn.Linear(128, 256)
#         # 第二个全连接层，输入特征为256，输出特征为8*128
#         self.fc2 = nn.Linear(256, 8 * 128)
#
#     def forward(self, x):
#         # 应用第一个全连接层并使用ReLU激活函数
#         x = F.relu(self.fc1(x))
#         # 应用第二个全连接层
#         x = self.fc2(x)
#         # 重塑成[8, 128]
#         x = x.view(8, 128)
#         return x
#
# class KB_Mapping(nn.Module):  # 输入为[8, 128] 输出为[8, 128] 形状不变
#     def __init__(self):
#         super(KB_Mapping, self).__init__()
#         # 全连接层，输入特征为8*128，输出特征为1024（中间层）
#         self.fc1 = nn.Linear(8 * 128, 1024)
#         # 全连接层，输入特征为1024，输出特征为8*128
#         self.fc2 = nn.Linear(1024, 8 * 128)
#
#     def forward(self, x):
#         # 将输入的[8, 128]张量展平为[1, 8*128]
#         x = x.view(1, -1)
#         # 通过第一个全连接层并使用ReLU激活函数
#         x = F.relu(self.fc1(x))
#         # 通过第二个全连接层
#         x = self.fc2(x)
#         # 重新将[1, 8*128]张量重塑为[8, 128]
#         x = x.view(8, 128)
#         return x
class KnowledgeBase(nn.Module):  # 没有bs 输入是一个[1, 128]的张量 输出是一个[8, 128]的张量
    def __init__(self):
        super(KnowledgeBase, self).__init__()
        # 核心：深度可分离卷积（3x3 DW + 1x1 PW）实现128→1024通道扩展
        # 无Batch维度，仅处理[1,128]输入
        self.depthwise = nn.Conv2d(  # 下面的可以重复使用
            in_channels=128,
            out_channels=128,
            kernel_size=3,  # 要求的3x3 DW卷积核
            padding=1,  # 保证卷积后空间维度不变
            groups=128,  # 深度卷积核心（逐通道）
            bias=False
        )
        # 1x1 PW卷积：将128通道扩展到8*128=1024通道（实现1→8组的核心）
        self.pointwise = nn.Conv2d(
            in_channels=128,
            out_channels=8 * 128,  # 如果改的话 就是改这里
            kernel_size=1,
            bias=False
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 输入：[1, 128] → 适配卷积的4维输入 [128, 1, 1]（通道, 高, 宽）
        # 注：这里省略batch维度，直接按单样本处理
        x = x.unsqueeze(-1).unsqueeze(-1)  # [1,128] → [1,128,1,1] → 挤压为[128,1,1]
        x = x.squeeze(0)  # 去掉多余的第0维 → [128,1,1]

        # 深度可分离卷积：特征变换+通道扩展
        x = self.relu(self.depthwise(x))  # [128,1,1] → 维度不变
        x = self.relu(self.pointwise(x))  # [128,1,1] → [1024,1,1]

        # 维度重塑：[1024,1,1] → [8,128]
        x = x.flatten()  # 展平为[1024]
        x = x.view(8, 128)  # 拆分为8组×128维

        return x


class KB_Mapping(nn.Module):
    def __init__(self, mask_seed=42):
        super(KB_Mapping, self).__init__()
        self.channels = 128
        self.mask_seed = mask_seed  # 新增：固定掩码种子（保证推理稳定）

        # ===================== 分支1：轻量特征变换（DW+PW） =====================
        self.branch1_dw = nn.Conv2d(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=3,
            padding=1,
            groups=self.channels,  # 深度卷积
            bias=False
        )
        self.branch1_pw = nn.Conv2d(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=1,
            bias=False
        )

        # ===================== 分支2：主分支（1x1卷积+残差+掩码+DW+PW） =====================
        self.branch2_conv1x1 = nn.Conv2d(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=1,  # 1x1卷积（少参高速）
            bias=False
        )
        self.branch2_dw = nn.Conv2d(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=3,
            padding=1,
            groups=self.channels,
            bias=False
        )
        self.branch2_pw = nn.Conv2d(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=1,
            bias=False
        )

        # ===================== 融合层：Concat后1x1卷积压缩维度 =====================
        self.fusion_conv1x1 = nn.Conv2d(
            in_channels=self.channels * 2,  # Concat后通道数256
            out_channels=self.channels,  # 压缩回128
            kernel_size=1,
            bias=False
        )

        # 激活函数
        self.relu = nn.ReLU(inplace=True)

    def random_masking(self, x, p=0.5):
        """
        适配隐私混淆的掩码：训练随机种子（增强鲁棒性），推理固定种子（保证稳定）
        :param x: 输入张量 [128,8,1]
        :param p: 置零概率
        :return: 掩码后的张量
        """
        # 推理阶段：固定种子 → 同一输入始终生成相同掩码（混淆但稳定）
        if not self.training:
            torch.manual_seed(self.mask_seed)
            if x.is_cuda:  # 兼容GPU
                torch.cuda.manual_seed(self.mask_seed)

        # 生成掩码（核心逻辑和你的一致）
        mask = torch.bernoulli(torch.ones_like(x) * (1 - p))
        out = x * mask

        # 训练阶段：重置种子（避免影响其他随机操作） 训的时候 随机种子不固定 什么乱七八糟的都学 避免过拟合
        if self.training:
            torch.seed()
            if x.is_cuda:
                torch.cuda.seed()

        return out

    def forward(self, x):
        # 输入形状：[8, 128] → 适配卷积的4维输入 [128, 8, 1]
        x = x.permute(1, 0).unsqueeze(-1)

        # ===================== 分支1：轻量特征变换 =====================
        branch1_out = self.relu(self.branch1_dw(x))
        branch1_out = self.relu(self.branch1_pw(branch1_out))

        # ===================== 分支2：主分支 =====================
        branch2_out = self.relu(self.branch2_conv1x1(x))
        branch2_out = branch2_out + branch1_out
        branch2_out = self.random_masking(branch2_out)  # 训练/推理都掩码
        branch2_out = self.relu(self.branch2_dw(branch2_out))
        branch2_out = self.relu(self.branch2_pw(branch2_out))

        # ===================== 特征融合 =====================
        concat_out = torch.cat([x, branch2_out], dim=0)
        fusion_out = self.relu(self.fusion_conv1x1(concat_out.unsqueeze(0)))
        fusion_out = fusion_out.squeeze(0)

        # ===================== 还原输出形状 =====================
        output = fusion_out.squeeze(-1).permute(1, 0)

        return output


class DeepSC(nn.Module):
    def __init__(self, num_layers, src_vocab_size, trg_vocab_size, src_max_len,
                 trg_max_len, d_model, num_heads, dff, dropout=0.1):
        super(DeepSC, self).__init__()

        self.encoder = Encoder(num_layers, src_vocab_size, src_max_len,
                               d_model, num_heads, dff, dropout)  # 这个就是语义编码器

        # self.mac_encoder = MAC_generate()

        self.channel_encoder = nn.Sequential(nn.Linear(d_model, 256),
                                             # nn.ELU(inplace=True),
                                             nn.ReLU(),
                                             nn.Linear(256, 16))

        self.channel_decoder = ChannelDecoder(16, d_model, 512)

        # self.mac_decoder = MAC_verify()

        self.decoder = Decoder(num_layers, trg_vocab_size, trg_max_len,
                               d_model, num_heads, dff, dropout)

        self.dense = nn.Linear(d_model, trg_vocab_size)  # 输出层


class MAC(nn.Module):
    def __init__(self):
        super(MAC, self).__init__()
        self.mac_encoder = MAC_generate()
        self.mac_decoder = MAC_verify()


# class MAC_generate_attacker(nn.Module):  # 输入是[bs, 31, 128] 输出是[bs, 1, 128]-----------------------待定 换成上面的MAC_generate了 同样也是普通用户的MAC生成器 因为这样设置没道理
#     def __init__(self, args):
#         super(MAC_generate_attacker, self).__init__()
#         self.linear = nn.Linear(31 * 128, 1 * 128)
#
#     def forward(self, x):
#         x = x.view(x.size(0), -1)  # 将x的形状从[bs, 31, 128]变为[bs, 31*128]
#         x = self.linear(x)
#         # 将输出x的形状调整为[bs, 1, 128]
#         x = x.view(x.size(0), 1, 128)
#         return x

class MAC_generate_attacker(nn.Module):
    def __init__(self):  # 输入三个张量，分别是[bs, 31, 128] [bs, 8, 128] [bs, 8, 128]，输出是[bs, 1, 128]
        super(MAC_generate_attacker, self).__init__()
        # 定义第1个输入的卷积层（特征）
        self.conv1 = nn.Conv1d(in_channels=31, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        # 定义第2个输入的卷积层（Eve知识库）
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(16)
        # 定义第3个输入的卷积层（bob知识库映射）
        self.conv3 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(16)
        # 定义融合后的卷积层
        self.conv4 = nn.Conv1d(in_channels=48, out_channels=16, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(16)
        # 定义输出层
        self.conv5 = nn.Conv1d(in_channels=16, out_channels=1, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm1d(1)

    def forward(self, x1, x2, x3):  # x1是特征，x2是Eve知识库，x3是bob知识库映射
        # f的前向传播
        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        # Eve知识库的前向传播
        x2 = self.conv2(x2)
        x2 = self.bn2(x2)
        x2 = F.relu(x2)
        # Bob知识库映射的前向传播
        x3 = self.conv3(x3)
        x3 = self.bn3(x3)
        x3 = F.relu(x3)
        # 将3个输入在通道维度上拼接
        x = torch.cat((x1, x2, x3), dim=1)
        # 融合后的前向传播
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        # 输出层前向传播
        x = self.conv5(x)
        x = self.bn5(x)
        return x


class NoiseNet(nn.Module):  # 输入输出都是[bs, 32, 16] 模拟噪声网络  # 暂时废弃不用了
    def __init__(self):
        super(NoiseNet, self).__init__()
        self.conv1 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(32, 32, kernel_size=3, padding=1)

    def forward(self, x):
        original_length = x.size(1)
        # 如果x的第一维度不是32 就填充成32
        if x.size(1) != 32:
            x = F.pad(x, (0, 0, 0, 32 - x.size(1)))

        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        # 由于卷积层的padding=1，所以输出长度会比输入多2（每边多1）
        # 我们需要裁剪掉两边的1个元素来保持长度一致
        x = x[:, :original_length, :]
        return x


class Burst(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=32, seq_len=32):  # 添加seq_len参数
        super(Burst, self).__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len  # 保存序列长度
        self.fc1 = nn.Linear(input_dim * seq_len, hidden_dim)  # 使用seq_len
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, input_dim * seq_len)  # 使用seq_len

    def forward(self, x):
        input_power = self.calculate_power(x)
        batch_size = x.size(0)

        # 展平输入（使用self.seq_len）
        flat_x = x.view(batch_size, self.input_dim * self.seq_len)

        # 通过网络
        hidden = self.relu(self.fc1(flat_x))
        output_flat = self.fc2(hidden)

        # 恢复形状（使用self.seq_len）
        output = output_flat.view(batch_size, self.seq_len, self.input_dim)

        # 功率匹配
        output_power = self.calculate_power(output)
        scale = torch.sqrt(input_power / (output_power + 1e-8)).unsqueeze(1).unsqueeze(2)
        return output * scale

    @staticmethod
    def calculate_power(signals: torch.Tensor) -> torch.Tensor:
        """计算每个信号的全局功率"""
        squared = signals.pow(2)
        return squared.mean(dim=[1, 2])
    # def __init__(self):
    #     super(Burst, self).__init__()
    #
    # def forward(self, x):
    #     # 计算输入的功率 (每个样本单独计算)
    #     input_power = torch.mean(x.pow(2), dim=(1, 2), keepdim=True)
    #
    #     # 生成随机噪声，形状与输入相同
    #     noise = torch.randn_like(x)
    #
    #     # 计算随机噪声的功率
    #     noise_power = torch.mean(noise.pow(2), dim=(1, 2), keepdim=True)
    #
    #     # 调整噪声功率以匹配输入功率
    #     scaling_factor = torch.sqrt(input_power / (noise_power + 1e-10))
    #     output = noise * scaling_factor
    #
    #     return output


class Attacker(nn.Module):
    def __init__(self):
        super(Attacker, self).__init__()
        self.mac_encoder = MAC_generate_attacker()  # 这里得改一下 他的MAC生成器应该是和普通的一样才对 同时接收f Eve_KB Bob_KB_mapping（就是不要密钥）
        # self.burst = Burst()


class tmp_full(nn.Module):  # 将1维度的展成48维度的 便于计算B2
    def __init__(self):
        super(tmp_full, self).__init__()
        self.fc = nn.Linear(31 * 128 * 1, 31 * 128 * 48)

    def forward(self, x):
        # 输入(batch_size, 1, 31, 1281)
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # 展平操作
        x = self.fc(x)  # 全连接层
        x = x.view(batch_size, 48, 31, 128)  # 恢复维度
        return x


# class Hiding(nn.Module):
#     def __init__(self, N):
#         super(Hiding, self).__init__()
#         self.encoder_hiding = Encoder_hiding()
#         self.embedder_hiding = Embedder_hiding()
#         self.invariance = CustomLinearLayer(1, N)
#         self.extractor = Extractor(N)
#         self.decoder_hiding = Decoder_hiding()
#         self.tmp = tmp_full()


class H_DeepSC(nn.Module):  # 没用了
    def __init__(self, N, num_layers, src_vocab_size, trg_vocab_size, src_max_len,
                 trg_max_len, d_model, num_heads, dff, dropout=0.1):
        super(H_DeepSC, self).__init__()

        self.deepsc = DeepSC(num_layers, src_vocab_size, trg_vocab_size, src_max_len,
                             trg_max_len, d_model, num_heads, dff, dropout)
        self.hiding = Hiding(N)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))  # math.log(math.exp(1)) = 1
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        x = self.dropout(x)
        return x


class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % num_heads == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.dense = nn.Linear(d_model, d_model)

        # self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query = self.wq(query).view(nbatches, -1, self.num_heads, self.d_k)
        query = query.transpose(1, 2)

        key = self.wk(key).view(nbatches, -1, self.num_heads, self.d_k)
        key = key.transpose(1, 2)

        value = self.wv(value).view(nbatches, -1, self.num_heads, self.d_k)
        value = value.transpose(1, 2)

        #        query, key, value = \
        #            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        #             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = self.attention(query, key, value, mask=mask)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.num_heads * self.d_k)

        x = self.dense(x)
        x = self.dropout(x)

        return x

    def attention(self, query, key, value, mask=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        # print(mask.shape)
        if mask is not None:
            # 根据mask，指定位置填充 -1e9
            scores += (mask * -1e9)
            # attention weights
        p_attn = F.softmax(scores, dim=-1)
        return torch.matmul(p_attn, value), p_attn


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_1(x)
        x = F.relu(x)
        x = self.w_2(x)
        x = self.dropout(x)
        return x


# class LayerNorm(nn.Module):
#    "Construct a layernorm module (See citation for details)."
#    # features = d_model
#    def __init__(self, features, eps=1e-6):
#        super(LayerNorm, self).__init__()
#        self.a_2 = nn.Parameter(torch.ones(features))
#        self.b_2 = nn.Parameter(torch.zeros(features))
#        self.eps = eps
#
#    def forward(self, x):
#        mean = x.mean(-1, keepdim=True)
#        std = x.std(-1, keepdim=True)
#        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, d_model, num_heads, dff, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadedAttention(num_heads, d_model, dropout=0.1)
        self.ffn = PositionwiseFeedForward(d_model, dff, dropout=0.1)

        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        attn_output = self.mha(x, x, x, mask)
        x = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(x)
        x = self.layernorm2(x + ffn_output)

        return x


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, d_model, num_heads, dff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_mha = MultiHeadedAttention(num_heads, d_model, dropout=0.1)
        self.src_mha = MultiHeadedAttention(num_heads, d_model, dropout=0.1)
        self.ffn = PositionwiseFeedForward(d_model, dff, dropout=0.1)

        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm3 = nn.LayerNorm(d_model, eps=1e-6)

        # self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, look_ahead_mask, trg_padding_mask):
        "Follow Figure 1 (right) for connections."
        # m = memory

        attn_output = self.self_mha(x, x, x, look_ahead_mask)
        x = self.layernorm1(x + attn_output)

        src_output = self.src_mha(x, memory, memory, trg_padding_mask)  # q, k, v
        x = self.layernorm2(x + src_output)

        fnn_output = self.ffn(x)
        x = self.layernorm3(x + fnn_output)
        return x


class Encoder(nn.Module):  # This is the encoder of transformer
    "Core encoder is a stack of N layers"

    def __init__(self, num_layers, src_vocab_size, max_len,
                 d_model, num_heads, dff, dropout=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.embedding = nn.Embedding(src_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len)
        self.enc_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, dff, dropout)
                                         for _ in range(num_layers)])

    def forward(self, x, src_mask, Alice_ID, Bob_ID):  # ID is float
        "Pass the input (and mask) through each layer in turn."
        # the input size of x is [batch_size, seq_len]
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        # then cat the ID
        x = torch.cat((x, Alice_ID, Bob_ID), 1)
        # cat the mask
        Alice_KB_mask = torch.zeros(x.size(0), 1, 8).to(device)
        Bob_KB_mask = torch.zeros(x.size(0), 1, 8).to(device)
        mask = torch.cat((src_mask, Alice_KB_mask, Bob_KB_mask), 2)

        for enc_layer in self.enc_layers:
            x = enc_layer(x, mask)

        return x


class Decoder(nn.Module):
    def __init__(self, num_layers, trg_vocab_size, max_len,
                 d_model, num_heads, dff, dropout=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.embedding = nn.Embedding(trg_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len)
        self.dec_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, dff, dropout)
                                         for _ in range(num_layers)])

    def forward(self, x, memory, look_ahead_mask, trg_padding_mask, Alice_kb_final, Bob_kb_final, mac):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        memory = torch.cat((memory, Alice_kb_final, Bob_kb_final, mac), 1)

        # cat the mask
        Alice_KB_mask = torch.zeros(x.size(0), 1, 8).to(device)
        Bob_KB_mask = torch.zeros(x.size(0), 1, 8).to(device)
        mac_mask = torch.zeros(x.size(0), 1, 1).to(device)
        mask = torch.cat((trg_padding_mask, Alice_KB_mask, Bob_KB_mask, mac_mask), 2)

        for dec_layer in self.dec_layers:
            x = dec_layer(x, memory, look_ahead_mask, mask)

        return x


class ChannelDecoder(nn.Module):
    def __init__(self, in_features, size1, size2):
        super(ChannelDecoder, self).__init__()

        self.linear1 = nn.Linear(in_features, size1)
        self.linear2 = nn.Linear(size1, size2)
        self.linear3 = nn.Linear(size2, size1)
        # self.linear4 = nn.Linear(size1, d_model)

        self.layernorm = nn.LayerNorm(size1, eps=1e-6)

    def forward(self, x):
        x1 = self.linear1(x)
        x2 = F.relu(x1)
        x3 = self.linear2(x2)
        x4 = F.relu(x3)
        x5 = self.linear3(x4)

        output = self.layernorm(x1 + x5)

        return output


class LinearAttention(nn.Module):
    """线性多头注意力（轻量级，2头）+ 残差+LayerNorm"""

    def __init__(self, head=2, d_model=128):
        super().__init__()
        self.head = head
        self.d_model = d_model
        self.d_k = d_model // head

        # 保留你的Q/K/V三个独立线性层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # 新增：LayerNorm（用于残差后）
        self.norm = nn.LayerNorm(d_model)
        # 新增：残差缩放参数（避免梯度爆炸）
        self.residual_scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, q, k, v):
        # 保留你的核心代码
        bs = q.size(0)
        Q = self.W_q(q)
        K = self.W_k(k)
        V = self.W_v(v)

        # 重塑张量实现多头注意力
        Q = Q.view(bs, -1, self.head, self.d_k).transpose(1, 2)
        K = K.view(bs, -1, self.head, self.d_k).transpose(1, 2)
        V = V.view(bs, -1, self.head, self.d_k).transpose(1, 2)

        # 应用核函数实现线性注意力
        Q = F.elu(Q) + 1
        K = F.elu(K) + 1

        # 线性注意力计算
        KV = torch.matmul(K.transpose(-1, -2), V)
        Z = 1 / (torch.matmul(Q, K.sum(dim=-2, keepdim=True).transpose(-1, -2)))
        attention = torch.matmul(Q, KV) * Z

        # 重塑回原始维度
        res = attention.transpose(dim0=1, dim1=2).contiguous().view(bs, -1, self.d_model)
        res = self.W_o(res)

        # 新增：残差连接（原始输入q + 注意力输出） + LayerNorm
        # 注：这里q是原始输入，因为是自注意力（q=k=v=f）
        res = self.norm(res + self.residual_scale * q)
        return res


class FusionLayer(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.weight_conv = nn.Conv1d(dim, 1, kernel_size=1)
        self.out_conv = nn.Conv1d(dim, dim, kernel_size=1)

    def forward(self, modalities):
        # 1. 计算每个模态的权重（确保形状完全对齐）
        weights = []
        for feat in modalities:
            # feat: [bs,1,128] → 转置为[bs,128,1]（Conv1d输入格式）
            feat_conv = feat.transpose(1, 2)  # [bs,128,1]
            w = self.weight_conv(feat_conv)  # [bs,1,1]
            weights.append(w)  # 每个权重：[bs,1,1]

        # 2. Softmax归一化权重
        weights = torch.cat(weights, dim=1)  # [bs,3,1]
        weights = F.softmax(weights, dim=1)  # [bs,3,1]

        # 3. 加权求和（彻底避免广播问题）
        weighted_sum = torch.zeros_like(modalities[0].transpose(1, 2))  # [bs,128,1]
        for i in range(len(modalities)):
            # 模态特征：[bs,1,128] → [bs,128,1]
            feat = modalities[i].transpose(1, 2)
            # 对应权重：[bs,3,1] → 取第i个权重，形状[bs,1,1]
            w = weights[:, i:i + 1, :]  # 关键：保持维度为[bs,1,1]
            # 相乘：[bs,128,1] * [bs,1,1] → 广播匹配
            weighted_sum += feat * w

        # 4. 最终输出
        out = self.out_conv(weighted_sum)  # [bs,128,1]
        out = torch.tanh(out).transpose(1, 2)  # [bs,1,128]
        return out


class MAC_generate(nn.Module):
    """整体网络：k/f/kb+kb_M处理 + 融合输出"""

    def __init__(self):
        super(MAC_generate, self).__init__()
        # 1. k的处理：深度可分离卷积
        self.k_dwconv = nn.Conv1d(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            padding=1,
            groups=128  # 深度卷积
        )
        self.k_pwconv = nn.Conv1d(
            in_channels=128,
            out_channels=128,
            kernel_size=1  # 逐点卷积
        )

        # 2. f的处理：线性多头注意力（2头）
        self.f_attention = LinearAttention(head=2, d_model=128)

        # 3. kb和kb_M的处理：1x1 Conv + 拼接 + 1x1 Conv
        self.kb_conv = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1)
        self.kbM_conv = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1)
        self.kb_concat_conv = nn.Conv1d(
            in_channels=128,  # 拼接后通道仍保持128（仅序列长度增加）
            out_channels=128,
            kernel_size=1
        )

        # 4. 全局平均池化（统一序列长度到1）
        self.global_pool = lambda x: torch.mean(x, dim=1, keepdim=True)

        # 5. 融合层
        self.fusion = FusionLayer(dim=128)

        # 激活函数
        self.relu = nn.ReLU(inplace=True)

    def forward(self, k, f, kb, kb_M):
        """
        输入：
            k: [bs,10,128], f: [bs,31,128], kb: [bs,8,128], kb_M: [bs,8,128]
        输出：
            y: [bs,1,128]
        """
        # ===================== 处理k =====================
        # [bs,10,128] → 转置为[bs,128,10]（适配Conv1d）
        k_ = k.transpose(1, 2)  # [bs,128,10]
        k_dw = self.relu(self.k_dwconv(k_))  # [bs,128,10]
        k_pw = self.relu(self.k_pwconv(k_dw))  # [bs,128,10]
        # 转回[bs,10,128] → 全局池化→[bs,1,128]
        k_feat = self.global_pool(k_pw.transpose(1, 2))  # [bs,1,128]

        # ===================== 处理f =====================
        f_attn = self.f_attention(q=f, k=f, v=f)  # [bs,31,128]（线性注意力+残差）
        f_feat = self.global_pool(f_attn)  # [bs,1,128]

        # ===================== 处理kb和kb_M =====================
        # 各自1x1 Conv
        kb_ = kb.transpose(1, 2)  # [bs,128,8]
        kb_conv = self.relu(self.kb_conv(kb_))  # [bs,128,8]
        kbM_ = kb_M.transpose(1, 2)  # [bs,128,8]
        kbM_conv = self.relu(self.kbM_conv(kbM_))  # [bs,128,8]
        # 拼接（沿序列长度维度，8+8=16）
        kb_concat = torch.cat([kb_conv, kbM_conv], dim=2)  # [bs,128,16]
        # 拼接后1x1 Conv
        kb_out = self.relu(self.kb_concat_conv(kb_concat))  # [bs,128,16]
        # 转回[bs,16,128] → 全局池化→[bs,1,128]
        kb_feat = self.global_pool(kb_out.transpose(1, 2))  # [bs,1,128]

        # ===================== 融合输出 =====================
        # 3个模态特征融合
        y = self.fusion([k_feat, f_feat, kb_feat])  # [bs,1,128]
        return y


class MAC_verify(nn.Module):
    """扩展版：新增IBSID输入 + 输出[bs,2]（Sigmoid激活）"""

    def __init__(self):
        super(MAC_verify, self).__init__()
        # 1. 复用原网络的k处理模块
        self.k_dwconv = nn.Conv1d(128, 128, 3, padding=1, groups=128)
        self.k_pwconv = nn.Conv1d(128, 128, 1)

        # 2. 新增：IBSID处理模块（和k的处理逻辑完全一致）
        self.ibsid_dwconv = nn.Conv1d(128, 128, 3, padding=1, groups=128)
        self.ibsid_pwconv = nn.Conv1d(128, 128, 1)

        # 3. 复用原网络的f注意力模块
        self.f_attention = LinearAttention(head=2, d_model=128)

        # 4. 复用原网络的kb+kb_M处理模块
        self.kb_conv = nn.Conv1d(128, 128, 1)
        self.kbM_conv = nn.Conv1d(128, 128, 1)
        self.kb_concat_conv = nn.Conv1d(128, 128, 1)

        # 5. 复用原网络的全局池化（统一序列长度到1）
        self.global_pool = lambda x: torch.mean(x, dim=1, keepdim=True)

        # 6. 复用融合层（自动适配4个模态）
        self.fusion = FusionLayer(dim=128)

        # 7. 新增：输出头（128维→2维，Sigmoid激活）
        self.output_head = nn.Sequential(
            nn.Conv1d(128, 1, kernel_size=1),  # [bs,128,1] → [bs,2,1]
            nn.Sigmoid()  # 输出2分类概率
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, IBSID, f, k, kb_M, kb):
        """
        输入：
            k: [bs,10,128], f: [bs,31,128], kb: [bs,8,128], kb_M: [bs,8,128], IBSID: [bs,1,128]
        输出：
            y: [bs,2]
        """
        # ===================== 处理k（复用原逻辑） =====================
        k_ = k.transpose(1, 2)
        k_dw = self.relu(self.k_dwconv(k_))
        k_pw = self.relu(self.k_pwconv(k_dw))
        k_feat = self.global_pool(k_pw.transpose(1, 2))  # [bs,1,128]

        # ===================== 处理IBSID（和k逻辑完全一致） =====================
        ibsid_ = IBSID.transpose(1, 2)  # [bs,128,1]
        ibsid_dw = self.relu(self.ibsid_dwconv(ibsid_))
        ibsid_pw = self.relu(self.ibsid_pwconv(ibsid_dw))
        ibsid_feat = self.global_pool(ibsid_pw.transpose(1, 2))  # [bs,1,128]

        # ===================== 处理f（复用原逻辑） =====================
        f_attn = self.f_attention(q=f, k=f, v=f)
        f_feat = self.global_pool(f_attn)  # [bs,1,128]

        # ===================== 处理kb+kb_M（复用原逻辑） =====================
        kb_ = kb.transpose(1, 2)
        kb_conv = self.relu(self.kb_conv(kb_))
        kbM_ = kb_M.transpose(1, 2)
        kbM_conv = self.relu(self.kbM_conv(kbM_))
        kb_concat = torch.cat([kb_conv, kbM_conv], dim=2)
        kb_out = self.relu(self.kb_concat_conv(kb_concat))
        kb_feat = self.global_pool(kb_out.transpose(1, 2))  # [bs,1,128]

        # ===================== 4模态融合（复用融合层） =====================
        fusion_out = self.fusion([k_feat, f_feat, kb_feat, ibsid_feat])  # [bs,1,128]

        # ===================== 输出头（128维→1维） =====================
        out = fusion_out.transpose(1, 2)  # [bs,128,1]
        out = self.output_head(out)  # [bs,1,1]
        out = out.squeeze(-1)  # [bs,1]

        return out


# 12部分
# 对SNR进行形状检查和调整的辅助函数
import torch
import torch.nn as nn


def _snr_to_B1(snr: torch.Tensor) -> torch.Tensor:
    """snr: [B] or [B,1] -> [B,1]"""
    if snr.dim() == 1:
        return snr.unsqueeze(-1)
    if snr.dim() == 2 and snr.size(-1) == 1:
        return snr
    raise ValueError(f"snr must be [B] or [B,1], got {tuple(snr.shape)}")


# -------------------------
# ResNet Block (1D version)
# -------------------------
class ResNetBlock1D(nn.Module):
    """
    ResNet block on 1D length dimension.
    Input/Output: x [B, C, L]
    """

    def __init__(self, C: int, kernel_size: int = 3):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(C, C, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(C, C, kernel_size=kernel_size, padding=padding)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.conv1(x))
        y = self.conv2(y)
        return x + y


# -------------------------
# CBAM: CA + SA (1D version)
# -------------------------
class ChannelAttentionSNR(nn.Module):
    """
    Channel Attention (CA) with SNR condition.
    x:   [B, C, L]
    snr: [B] or [B,1]
    out: [B, C, L]
    """

    def __init__(self, C: int, reduction: int = 8):
        super().__init__()
        hidden = max(C // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Linear(C + 1, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, C)
        )

    def forward(self, x: torch.Tensor, snr: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        snr = _snr_to_B1(snr)  # [B,1]

        # CBAM-CA uses avg & max pooling over spatial/length dim
        x_avg = x.mean(dim=-1)  # [B,C]
        x_max = x.amax(dim=-1)  # [B,C]

        v_avg = torch.cat([x_avg, snr], dim=-1)  # [B,C+1]
        v_max = torch.cat([x_max, snr], dim=-1)

        w = torch.sigmoid(self.mlp(v_avg) + self.mlp(v_max))  # [B,C]
        w = w.unsqueeze(-1)  # [B,C,1], broadcast on L
        return x * w


class SpatialAttention1D(nn.Module):
    """
    Spatial Attention (SA) adapted to 1D length L.
    x:   [B, C, L]
    out: [B, C, L]
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CBAM-SA: avg & max over channels -> [B,1,L], then conv -> [B,1,L]
        x_avg = x.mean(dim=1, keepdim=True)  # [B,1,L]
        x_max = x.amax(dim=1, keepdim=True)  # [B,1,L]
        m = torch.cat([x_avg, x_max], dim=1)  # [B,2,L]
        w = torch.sigmoid(self.conv(m))  # [B,1,L]
        return x * w


class CBAM_SNR_1D(nn.Module):
    """
    CBAM = CA(SNR) -> SA
    """

    def __init__(self, C: int, ca_reduction: int = 8, sa_kernel: int = 7):
        super().__init__()
        self.ca = ChannelAttentionSNR(C=C, reduction=ca_reduction)
        self.sa = SpatialAttention1D(kernel_size=sa_kernel)

    def forward(self, x: torch.Tensor, snr: torch.Tensor) -> torch.Tensor:
        x = self.ca(x, snr)
        x = self.sa(x)
        return x


# -------------------------
# CCAM: FiLM (gamma/beta) with SNR
# -------------------------
class CCAM_SNR(nn.Module):
    """
    CCAM (FiLM style):
      pooled(x) + snr -> gamma, beta (per channel)
      out = gamma * x + beta

    x:   [B, C, L]
    snr: [B] or [B,1]
    out: [B, C, L]
    """

    def __init__(self, C: int, hidden: int = 64):
        super().__init__()
        in_dim = C + 1
        self.gamma = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, C)
        )
        self.beta = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, C)
        )

    def forward(self, x: torch.Tensor, snr: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        snr = _snr_to_B1(snr)

        x_pool = x.mean(dim=-1)  # [B,C]
        cond = torch.cat([x_pool, snr], dim=-1)  # [B,C+1]

        g = self.gamma(cond).unsqueeze(-1)  # [B,C,1]
        b = self.beta(cond).unsqueeze(-1)  # [B,C,1]
        return g * x + b


# -------------------------
# One Block in Fig.2
# -------------------------
class CAEMBlock(nn.Module):
    """
    Block in Fig.2:
      Input -> (optional) ResNet Block -> CBAM -> CCAM -> Output

    x:   [B,C,L]
    snr: [B] or [B,1]
    """

    def __init__(
            self,
            C: int,
            use_resnet: bool = False,
            ca_reduction: int = 8,
            sa_kernel: int = 7,
            ccam_hidden: int = 64,
    ):
        super().__init__()
        self.use_resnet = use_resnet
        self.res = ResNetBlock1D(C=C) if use_resnet else nn.Identity()
        self.cbam = CBAM_SNR_1D(C=C, ca_reduction=ca_reduction, sa_kernel=sa_kernel)
        self.ccam = CCAM_SNR(C=C, hidden=ccam_hidden)

    def forward(self, x: torch.Tensor, snr: torch.Tensor) -> torch.Tensor:
        x = self.res(x)
        x = self.cbam(x, snr)
        x = self.ccam(x, snr)
        return x


class CAEM_Fig2_SNR_1D(nn.Module):
    """
    CAEM in Fig.2 (SNR-only, 1D adaptation):
      x -> Block1 -> Block2 -> Conv(k=3) -> Z0

    Input:
      x   [B, C_in, L]   (your case: C_in=31, L=128)
      snr [B] or [B,1]
    Output:
      z0  [B, C_out, L]
    """

    def __init__(
            self,
            C_in: int = 31,
            C_out: int = 16,  # 对应论文 256->16 的“压通道”思想，你可改成 31 保持不变
            use_resnet: bool = False,  # 你说 resnet 可能不需要，就设 False
            ca_reduction: int = 8,
            sa_kernel: int = 7,
            ccam_hidden: int = 64,
    ):
        super().__init__()
        self.block1 = CAEMBlock(C=C_in, use_resnet=use_resnet,
                                ca_reduction=ca_reduction, sa_kernel=sa_kernel, ccam_hidden=ccam_hidden)
        self.block2 = CAEMBlock(C=C_in, use_resnet=use_resnet,
                                ca_reduction=ca_reduction, sa_kernel=sa_kernel, ccam_hidden=ccam_hidden)

        # Fig.2: Conv Layer 256x16x3 => here: Conv1d(C_in -> C_out, k=3)
        self.final_conv = nn.Conv1d(C_in, C_out, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, snr: torch.Tensor) -> torch.Tensor:
        x = self.block1(x, snr)
        x = self.block2(x, snr)
        z0 = self.final_conv(x)
        return z0


# 下面是特征选择模块
# 计算特征图熵的近似值
def feature_entropy_approx(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute a per-channel "entropy-like" scalar for real-valued feature maps.

    Paper uses "Entropy of Feature Maps" as C×1. For continuous features,
    an exact Shannon entropy isn't directly defined. A common surrogate is:

      - interpret values along length L as a discrete distribution via softmax
      - entropy = -sum(p log p)

    Input:
        x: [B, C, L]
    Output:
        ent: [B, C, 1]
    """
    p = F.softmax(x, dim=-1)  # [B,C,L], sum over L = 1
    ent = -(p * (p + eps).log()).sum(dim=-1, keepdim=True)  # [B,C,1]
    return ent


# 那个policy网络
class PolicyNetwork_SNR_AllC(nn.Module):
    """
    Policy network that outputs a prefix-selection decision over ALL C channels.

    Action space: {0, 1, ..., C}
      action K means: select first K channels (thermometer coding)

    Inputs:
      - z0 : [B, C, L]  (All Selective features = CAEM output)
      - ent: [B, C, 1]  (entropy per channel)
      - snr: [B] or [B,1]

    Outputs:
      - Mk     : [B, C, 1]      (thermometer / prefix mask)    这里可以考虑改一下 这样会导致可能有的样本一个channel都不选 可以改成前一半必选 后一半可选 就是注释的那个代码
      - probs  : [B, C+1]       (soft action probabilities)
      - onehot : [B, C+1]       (sampled action, one-hot)
    """

    def __init__(self, C: int, hidden: int = 128):
        super().__init__()
        self.C = C

        # Following Fig.9 spirit:
        #   concat entropy to feature map -> [B,C,L+1]
        #   pooling over length -> [B,C]
        #   concat SNR -> [B,C+1]
        #   MLP -> logits over (C+1) actions (0..C)
        self.mlp = nn.Sequential(
            nn.Linear(C + 1, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, C + 1)
        )

    @staticmethod
    def onehot_to_thermometer(onehot: torch.Tensor) -> torch.Tensor:
        """
        Convert action one-hot (0..C) to thermometer coding Mk of length C.

        Example (C=5):
          action=0 -> Mk=[0,0,0,0,0]
          action=2 -> Mk=[1,1,0,0,0]
          action=5 -> Mk=[1,1,1,1,1]
        """
        action = onehot.argmax(dim=-1)  # [B], 0..C
        B = action.size(0)
        C = onehot.size(1) - 1

        idx = torch.arange(C, device=onehot.device).view(1, C)  # [1,C]
        Mk = (idx < action.view(B, 1)).float()  # [B,C]
        return Mk.unsqueeze(-1)  # [B,C,1]

    def forward(self, z0: torch.Tensor, ent: torch.Tensor, snr: torch.Tensor,
                tau: float = 1.0, hard: bool = True):
        B, C, L = z0.shape
        assert C == self.C, f"Expected C={self.C}, got C={C}"
        snr = _snr_to_B1(snr)  # [B,1]

        # concat entropy as an extra "position" along L: [B,C,L+1]
        z_cat = torch.cat([z0, ent], dim=-1)

        # pooling over length -> [B,C]
        pooled = z_cat.mean(dim=-1)

        # concat SNR -> [B,C+1]
        feat = torch.cat([pooled, snr], dim=-1)

        logits = self.mlp(feat)  # [B,C+1]
        probs = F.softmax(logits, dim=-1)  # [B,C+1]

        # Gumbel-Softmax sampling (differentiable)
        onehot = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)  # [B,C+1]

        # convert to thermometer mask over C channels
        Mk = self.onehot_to_thermometer(onehot)  # [B,C,1]
        return Mk, probs, onehot


class FeatureMapSelectionModule_SNR_AllC(nn.Module):
    """
    Your modified Feature Map Selection Module:

    - All Selective feature = CAEM output z0: [B,C,L]
    - Entropy computed from z0
    - Policy network uses (z0, entropy, SNR) -> Mk of size [B,C,1]
    - Select: z1 = z0 * Mk (broadcast on L)

    Outputs:
      z1    : [B,C,L]      (fixed shape; "unselected" channels are zeroed)
      Mk    : [B,C,1]      (mask over ALL channels)
      ent   : [B,C,1]
      probs : [B,C+1]
    """

    def __init__(self, C: int, hidden: int = 128):
        super().__init__()
        self.C = C
        self.entropy_fn = feature_entropy_approx
        self.policy = PolicyNetwork_SNR_AllC(C=C, hidden=hidden)

    def forward(self, z0: torch.Tensor, snr: torch.Tensor,
                tau: float = 1.0, hard: bool = True):  # tau表示温度系数 作用是控制采样的平滑度
        """
        Inputs:
          z0:  [B,C,L]  (CAEM output)
          snr: [B] or [B,1]

        Returns:
          z1:   [B,C,L]
          Mk:   [B,C,1]
          ent:  [B,C,1]
          probs:[B,C+1]
          onehot:[B,C+1]
        """
        B, C, L = z0.shape
        assert C == self.C, f"Expected C={self.C}, got C={C}"

        ent = self.entropy_fn(z0)  # [B,C,1]
        Mk, probs, onehot = self.policy(z0, ent, snr, tau=tau, hard=hard)  # Mk: [B,C,1]

        # Apply mask to ALL channels
        z1 = z0 * Mk  # broadcast over L -> [B,C,L]
        return z1, Mk, ent, probs, onehot


# # 如果是前一半必选 后一半可选的策略网络
# class PolicyHalfPrefix(nn.Module):
#     """
#     Policy outputs an action in {0..C2} that means selecting first K channels
#     within the ADAPTIVE half.
#     """
#     def __init__(self, C: int, hidden: int = 128):
#         super().__init__()
#         assert C % 2 == 0
#         self.C = C
#         self.C2 = C // 2
#
#         # Input: pooled features + SNR -> logits over (C2+1) actions
#         self.mlp = nn.Sequential(
#             nn.Linear(C + 1, hidden),
#             nn.ReLU(inplace=True),
#             nn.Linear(hidden, self.C2 + 1)
#         )
#
#     @staticmethod
#     def onehot_to_thermometer(onehot: torch.Tensor) -> torch.Tensor:
#         # onehot: [B, C2+1] -> Mk_adapt: [B, C2, 1]
#         action = onehot.argmax(dim=-1)  # 0..C2
#         B = action.size(0)
#         C2 = onehot.size(1) - 1
#         idx = torch.arange(C2, device=onehot.device).view(1, C2)
#         Mk = (idx < action.view(B, 1)).float()
#         return Mk.unsqueeze(-1)
#
#     def forward(self, z0: torch.Tensor, ent: torch.Tensor, snr: torch.Tensor,
#                 tau: float = 1.0, hard: bool = True):
#         # z0: [B,C,L], ent: [B,C,1]
#         B, C, L = z0.shape
#         snr = _snr_to_B1(snr)
#
#         # Fig.9 style: concat entropy then pool -> [B,C]
#         z_cat = torch.cat([z0, ent], dim=-1)  # [B,C,L+1]
#         pooled = z_cat.mean(dim=-1)           # [B,C]
#         feat = torch.cat([pooled, snr], dim=-1)  # [B,C+1]
#
#         logits = self.mlp(feat)               # [B,C2+1]
#         probs = F.softmax(logits, dim=-1)
#         onehot = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)  # [B,C2+1]
#         Mk_adapt = self.onehot_to_thermometer(onehot)  # [B,C2,1]
#         return Mk_adapt, probs, onehot
#
#
# class FeatureMapSelection_BasePlusAdaptive(nn.Module):
#     """
#     Base+Adaptive selection:
#       - Base half: always selected (mask=1)
#       - Adaptive half: policy decides (mask from policy)
#
#     Output mask Mk has shape [B,C,1] and is applied to all channels.
#     """
#     def __init__(self, C: int, hidden: int = 128, base_first_half: bool = True):
#         super().__init__()
#         assert C % 2 == 0
#         self.C = C
#         self.C2 = C // 2
#         self.base_first_half = base_first_half
#         self.policy = PolicyHalfPrefix(C=C, hidden=hidden)
#
#     def forward(self, z0: torch.Tensor, snr: torch.Tensor,
#                 tau: float = 1.0, hard: bool = True):
#         """
#         z0: [B,C,L]  (CAEM output)
#         snr: [B] or [B,1]
#         """
#         B, C, L = z0.shape
#         assert C == self.C
#
#         ent = feature_entropy_approx(z0)  # [B,C,1]
#
#         Mk_adapt, probs, onehot = self.policy(z0, ent, snr, tau=tau, hard=hard)  # [B,C/2,1]
#
#         # Base mask = ones
#         base = torch.ones(B, self.C2, 1, device=z0.device, dtype=z0.dtype)
#
#         # Concatenate to full Mk
#         if self.base_first_half:
#             # [base | adaptive]
#             Mk = torch.cat([base, Mk_adapt], dim=1)  # [B,C,1]
#         else:
#             # [adaptive | base]
#             Mk = torch.cat([Mk_adapt, base], dim=1)
#
#         z1 = z0 * Mk  # [B,C,L]
#         return z1, Mk, ent, probs, onehot


class Conv1dAggregator(nn.Module):
    """
    Lightweight 1D conv aggregator over token/length dimension L.
    Input:  m: [B, 2C, L]
    Output: v: [B, H]
    """

    def __init__(self, in_channels: int, hidden_channels: int = 64, num_layers: int = 3, kernel_size: int = 3):
        super().__init__()
        assert num_layers >= 1
        padding = (kernel_size - 1) // 2

        layers = []
        c_in = in_channels
        c_h = hidden_channels
        for _ in range(num_layers):
            layers.append(nn.Conv1d(c_in, c_h, kernel_size=kernel_size, padding=padding))
            layers.append(nn.ReLU(inplace=True))
            c_in = c_h

        self.net = nn.Sequential(*layers)

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        h = self.net(m)  # [B, H, L]
        v = h.mean(dim=-1)  # [B, H]
        return v


# alice的分类网络，第一步是算差异与乘积（捕捉哪里一样和哪里不一样），然后conv1d聚合，最后mlp分类
class VerificationDiscriminatorLN(nn.Module):
    """
    Verification discriminator with domain-alignment normalization (Principle 2 - method a).

    Inputs:
        g     : [B, C, L]  (Alice full feature)
        g_hat : [B, C, L]  (Bob pruned/selected feature; many channels may be zeroed)
    Output:
        logits or probability of "legitimate"
    """

    def __init__(
            self,
            C: int,
            L: int = 128,
            agg_hidden: int = 64,
            agg_layers: int = 3,
            agg_kernel: int = 3,
            mlp_hidden: int = 64,
            output_logits: bool = True,
            eps: float = 1e-5,
    ):
        super().__init__()
        self.C = C
        self.L = L
        self.output_logits = output_logits

        # Token-wise LayerNorm over channel dimension C:
        # apply LN to [B, L, C]
        self.ln = nn.LayerNorm(C, eps=eps)

        self.aggregator = Conv1dAggregator(
            in_channels=2 * C,
            hidden_channels=agg_hidden,
            num_layers=agg_layers,
            kernel_size=agg_kernel,
        )

        self.mlp = nn.Sequential(
            nn.Linear(agg_hidden, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden, 1),
        )

    def _tokenwise_channel_ln(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, L] -> apply LayerNorm over C at each token position -> [B, C, L]
        """
        # [B, C, L] -> [B, L, C]
        x_t = x.transpose(1, 2)
        # LN over last dim C
        x_t = self.ln(x_t)
        # back to [B, C, L]
        return x_t.transpose(1, 2)

    def forward(self, g: torch.Tensor, g_hat: torch.Tensor) -> torch.Tensor:
        assert g.dim() == 3 and g_hat.dim() == 3, "g and g_hat must be [B, C, L]"
        B, C, L = g.shape
        assert g_hat.shape == (B, C, L), f"g_hat must match g. got {g_hat.shape}, expected {(B, C, L)}"
        assert C == self.C, f"Expected C={self.C}, got C={C}"
        assert L == self.L, f"Expected L={self.L}, got L={L}"

        # === Principle 2 (a): domain alignment via identical normalization ===
        g_n = self._tokenwise_channel_ln(g)
        g_hat_n = self._tokenwise_channel_ln(g_hat)

        # 1) token-wise matching features
        diff = torch.abs(g_n - g_hat_n)  # [B,C,L]
        prod = g_n * g_hat_n  # [B,C,L]
        m = torch.cat([diff, prod], dim=1)  # [B,2C,L]

        # 2) Conv1d aggregation over L
        v = self.aggregator(m)  # [B, agg_hidden]

        # 3) MLP head
        logits = self.mlp(v)  # [B,1]
        if self.output_logits:
            return logits
        return torch.sigmoid(logits)


# 下面是34的部分

import math
import torch
import torch.nn as nn

# =========================
# 1) SCHEDULE: 修正索引约定 + 可选 cosine
# =========================
class DiffusionSchedule:
    """
    约定：
      - alpha_bars: [T+1]
      - alpha_bars[0] = 1 (t=0 完全无噪声)
      - 扩散/训练时间步 t ∈ {1,...,T}
    """
    def __init__(self, T: int, schedule="cosine", beta_start=1e-4, beta_end=2e-2, device="cpu"):
        self.T = T
        self.device = device

        if schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, T, device=device)  # [T], 对应 t=1..T
            alphas = 1.0 - betas
            alpha_bars = torch.cumprod(alphas, dim=0)                       # [T]
            alpha_bars = torch.cat([torch.ones(1, device=device), alpha_bars], dim=0)  # [T+1]

        elif schedule == "cosine":
            # cosine schedule: 直接构造 alpha_bar，再推 beta
            # 常用 s=0.008
            s = 0.008
            steps = torch.arange(T + 1, device=device).float()  # 0..T
            f = torch.cos(((steps / T) + s) / (1 + s) * math.pi / 2) ** 2
            alpha_bars = f / f[0]  # 归一化，让 alpha_bars[0]=1

            # 推出 betas(t)=1 - alpha_bar(t)/alpha_bar(t-1)  for t=1..T
            betas = 1.0 - (alpha_bars[1:] / alpha_bars[:-1])
            betas = betas.clamp(1e-8, 0.999)
            alphas = 1.0 - betas

        else:
            raise ValueError("schedule must be 'linear' or 'cosine'")

        self.betas = betas              # [T]  (t=1..T)
        self.alphas = alphas            # [T]
        self.alpha_bars = alpha_bars    # [T+1]

    def sample_timesteps(self, bs: int):
        # 训练用 t ∈ [1, T]
        return torch.randint(1, self.T + 1, (bs,), device=self.device, dtype=torch.long)

    def q_sample(self, f0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor = None):
        """
        f_t = sqrt(alpha_bar[t]) * f0 + sqrt(1-alpha_bar[t]) * eps
        t: [bs] in {1..T}
        """
        if eps is None:
            eps = torch.randn_like(f0)

        # alpha_bar[t] 取出来并 reshape 以便广播
        a_bar = self.alpha_bars.to(f0.device)[t].view(-1, 1, 1)  # [bs,1,1]

        return torch.sqrt(a_bar) * f0 + torch.sqrt(1.0 - a_bar) * eps


# =========================
# 2) DDIM 采样：修正 t 范围，和 schedule 对齐
# =========================
@torch.no_grad()
def diffusion_sample_ddim(model, schedule: DiffusionSchedule, hat_f: torch.Tensor):
    """
    确定性 DDIM (eta=0)：
      - 从 x_T ~ N(0,1) 开始
      - t: T -> 1
      - 输出 x0
    """
    model.eval()
    device = hat_f.device
    bs, L, D = hat_f.shape

    T = schedule.T
    alpha_bars = schedule.alpha_bars.to(device)  # [T+1]

    # x_T from pure noise
    x = torch.randn_like(hat_f)

    for t in range(T, 0, -1):
        t_batch = torch.full((bs,), t, device=device, dtype=torch.long)

        a_t = alpha_bars[t].view(1, 1, 1)        # scalar -> broadcast
        a_prev = alpha_bars[t-1].view(1, 1, 1)

        eps_pred = model(f_t=x, t=t_batch, hat_f=hat_f)  # [bs,L,D]

        # predict x0
        x0_pred = (x - torch.sqrt(1.0 - a_t) * eps_pred) / torch.sqrt(a_t)

        # deterministic DDIM update
        x = torch.sqrt(a_prev) * x0_pred + torch.sqrt(1.0 - a_prev) * eps_pred

    return x  # x0


class ConditionalDenoiser(nn.Module):
    """
    更稳的条件融合版本：
      x = proj_noisy(f_t) + proj_cond(hat_f) + time_emb
    """
    def __init__(
        self,
        feature_dim=128,
        model_dim=256,
        num_layers=4,
        num_heads=8,
        time_dim=256,
        dropout=0.1
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.model_dim = model_dim

        # time embedding
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )

        # separate projections
        self.noisy_proj = nn.Linear(feature_dim, model_dim)
        self.cond_proj  = nn.Linear(feature_dim, model_dim)

        # stabilize
        self.in_norm = nn.LayerNorm(model_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.out_norm = nn.LayerNorm(model_dim)
        self.out_proj = nn.Linear(model_dim, feature_dim)

    def forward(self, f_t: torch.Tensor, t: torch.Tensor, hat_f: torch.Tensor):
        bs, L, D = f_t.shape
        assert hat_f.shape == (bs, L, D), "hat_f 的形状必须和 f_t 一致"

        # project
        x = self.noisy_proj(f_t) + self.cond_proj(hat_f)   # [bs, L, model_dim]

        # time emb
        t_emb = self.time_mlp(self.time_embed(t))          # [bs, model_dim]
        x = x + t_emb.unsqueeze(1)

        # norm + transformer
        x = self.in_norm(x)
        x = self.encoder(x)
        x = self.out_norm(x)

        eps_pred = self.out_proj(x)
        return eps_pred

# =========================================================
# 2) 时间步 t 的正弦时间嵌入（Sinusoidal Em    bedding）
# =========================================================
class SinusoidalTimeEmbedding(nn.Module):
    """
    将离散时间步 t 映射为连续向量（扩散模型标准做法）

    输入：
        t:   [bs]（long）
    输出：
        emb: [bs, time_dim]
    """
    def __init__(self, time_dim: int):
        super().__init__()
        self.time_dim = time_dim

    def forward(self, t: torch.Tensor):
        device = t.device
        half = self.time_dim // 2

        # 构造不同频率
        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(0, half, device=device).float()
            / (half - 1)
        )  # [half]

        # t: [bs] -> [bs, 1]
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # [bs, half]

        # sin + cos 拼接
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # [bs, time_dim]

        # 如果 time_dim 是奇数，补 0
        if self.time_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)

        return emb

@torch.no_grad()
def ddim_from_xt(model, schedule, x_T, hat_f, t_start: int):
    """
    DDIM 确定性反推：从给定的 x_{t_start} 开始 -> x_0
    schedule: alpha_bars [T+1], t in [1..T]
    """
    device = x_T.device
    bs = x_T.size(0)
    alpha_bars = schedule.alpha_bars.to(device)

    x = x_T
    for t in range(t_start, 0, -1):
        t_batch = torch.full((bs,), t, device=device, dtype=torch.long)

        a_t = alpha_bars[t].view(1, 1, 1)
        a_prev = alpha_bars[t-1].view(1, 1, 1)

        eps_pred = model(f_t=x, t=t_batch, hat_f=hat_f)
        x0_pred = (x - torch.sqrt(1.0 - a_t) * eps_pred) / torch.sqrt(a_t)

        x = torch.sqrt(a_prev) * x0_pred + torch.sqrt(1.0 - a_prev) * eps_pred

    return x



# 重写了扩散模型 用了UNet 并用了现有的库
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from diffusers import UNet1DModel, DDIMScheduler
from tqdm import tqdm
import os


# ==========================================
# 1. 归一化方案 A：标准正态分布归一化
# ==========================================
class FeatureScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit_transform(self, x):
        self.mean = x.mean()
        self.std = x.std()
        return (x - self.mean) / (self.std + 1e-6)

    def inverse_transform(self, x_norm):
        return x_norm * self.std + self.mean


# ==========================================
# 2. 条件扩散模型类定义
# ==========================================
class SemComDiffusion(nn.Module):
    def __init__(self, feat_dim=128, seq_len=31):
        super().__init__()
        self.feat_dim = feat_dim
        self.seq_len = seq_len

        # 定义 UNet1D
        # 输入维度: feat_dim * 2 (f_t 和 f_cond 在通道维度拼接)
        self.model = UNet1DModel(
            in_channels=feat_dim * 2,
            out_channels=feat_dim,
            block_out_channels=(128, 256, 512),
            layers_per_block=2,
            use_timestep_embedding=True
        )

        # 训练使用 DDPM 调度，推理使用 DDIM
        self.scheduler = DDIMScheduler(num_train_timesteps=1000)

    def forward(self, f_t, t, f_cond):
        """
        f_t: [bs, 128, 31]
        f_cond: [bs, 128, 31]
        """
        # 在通道维度拼接特征和条件
        net_input = torch.cat([f_t, f_cond], dim=1)
        return self.model(net_input, t).sample