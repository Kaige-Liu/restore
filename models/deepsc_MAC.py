from models.modules import Encoder, Decoder

import torch
import torch.nn as nn
from utlis.trainer import create_masks

# 设置随机种子以确保可重复性
torch.manual_seed(5)

def generate_key(args, data_size):  # 输入的data_size=[bs, 31] 输出形状是[bs, 10]
    k_range = [6, 8]
    # 使用torch.randint生成均匀分布的随机整数
    key = torch.randint(high=k_range[1], low=k_range[0], size=(data_size[0], 8), dtype=torch.int32)
    # 创建起始和结束索引的列
    start_column = torch.full((data_size[0], 1), args.start_idx, dtype=torch.int32)
    end_column = torch.full((data_size[0], 1), args.end_idx, dtype=torch.int32)
    # 沿着列方向拼接
    key = torch.cat([start_column, key, end_column], dim=1)
    return key

# 定义Key_net类
class Key_net(nn.Module):
    def __init__(self, args):
        super(Key_net, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=args.vocab_size, embedding_dim=args.encoder_d_model)

    def forward(self, key):
        key_ebd = self.embedding(key)
        return key_ebd

# 定义Channels类
class Channels(nn.Module):
    def __init__(self):
        super(Channels, self).__init__()

    def awgn(self, inputs, n_std=0.1):
        # 使用torch.randn生成正态分布的随机数
        noise = torch.randn(inputs.shape) * n_std
        # 将噪声添加到输入信号
        outputs = inputs + noise
        return outputs

import torch
import torch.nn.functional as F

# class PowerNorm(nn.Module):
#     def __init__(self):
#         super(PowerNorm, self).__init__()
#
#     def forward(self, x):
#         power = x.pow(2).mean(dim=1, keepdim=True)
#         norm_factor = power.rsqrt()
#         return x * norm_factor


class Channel_Encoder(nn.Module):
    def __init__(self, size1=256, size2=16):
        super(Channel_Encoder, self).__init__()

        # TensorFlow Dense layer corresponds to PyTorch Linear layer
        self.dense0 = nn.Linear(128, size1)  # 前面的128是encoder_d_model
        self.dense1 = nn.Linear(size1, size2)

        # Define a function for power normalization
        self.powernorm = nn.Module()

        def powernorm_function(x):  # 功率归一化
            # PyTorch does not have a direct equivalent to TensorFlow's reduce_mean, so we use mean
            # Also, PyTorch does not have a Lambda layer, so we define the function here
            norm = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True))
            return x / norm

        self.powernorm.forward = powernorm_function

    def forward(self, inputs):  # 输入是[bs, 30, 128] 输出是[bs, 30, 16]
        # Use PyTorch's functional API for ReLU activation
        outputs1 = F.relu(self.dense0(inputs))
        outputs2 = self.dense1(outputs1)
        # Apply the power normalization function
        power_norm_outputs = self.powernorm(outputs2)

        return power_norm_outputs

# 定义Channel_Decoder类
class Channel_Decoder(nn.Module):
    def __init__(self, size1=128, size2=512):  # size1=128, size2=512
        super(Channel_Decoder, self).__init__()
        self.dense1 = nn.Linear(16, size1)  # 信道完了就是[bs, 30, 16]
        self.dense2 = nn.Linear(size1, size2)
        self.dense3 = nn.Linear(size2, size1)
        self.layernorm1 = nn.LayerNorm(size1, eps=1e-6)  # 表示

    def forward(self, receives):  # 输入是[bs, 30, 16] 输出是[bs, 30, 128]
        x1 = F.relu(self.dense1(receives))
        x2 = F.relu(self.dense2(x1))
        x3 = self.dense3(x2)
        output = self.layernorm1(x1 + x3)
        return output

class MAC_generate(nn.Module):  # 输入是[bs, 10, 128]和[bs, 31, 128] 输出是[bs, 1, 128]
    def __init__(self):
        super(MAC_generate, self).__init__()
        # 定义第一个输入的卷积层
        self.conv1 = nn.Conv1d(in_channels=10, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)

        # 定义第二个输入的卷积层
        self.conv2 = nn.Conv1d(in_channels=31, out_channels=16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(16)

        # 定义融合后的卷积层
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(16)

        # 定义输出层
        self.conv4 = nn.Conv1d(in_channels=16, out_channels=1, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(1)

    def forward(self, x1, x2):
        # 第一个输入的前向传播
        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)

        # 第二个输入的前向传播
        x2 = self.conv2(x2)
        x2 = self.bn2(x2)
        x2 = F.relu(x2)

        # 将两个输入在通道维度上拼接
        x = torch.cat((x1, x2), dim=1)

        # 融合后的前向传播
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        # 输出层前向传播
        x = self.conv4(x)
        x = self.bn4(x)

        return x

class MAC_verify(nn.Module):  # 输入是[bs, 32, 128], [bs, 10 128] 输出为0到1之间
    def __init__(self):
        super(MAC_verify, self).__init__()  # 待定 下面都是乱写的，需要修改


    def forward(self, recovered_semantic_enc_output, recovered_mac, key_ebd):  # 输出验证结果
        result = 1


        return recovered_semantic_enc_output, result




class Transmitter(nn.Module):  # 改完了 从维度上看没问题
    def __init__(self, args):
        super(Transmitter, self).__init__()
        self.embedding = nn.Embedding(args.vocab_size, args.encoder_d_model)
        # semantic encoder
        self.semantic_encoder = Encoder(args.encoder_num_layer, args.encoder_num_heads,
                                        args.encoder_d_model, args.encoder_d_ff,
                                        args.vocab_size, dropout_pro=args.encoder_dropout)

        # self.mac_encoder = Encoder(args.encoder_num_layer, args.encoder_num_heads,
        #                               args.encoder_d_model, args.encoder_d_ff,
        #                               args.vocab_size, dropout=args.encoder_dropout)
        self.mac_encoder = MAC_generate()  # 输入是[bs, 10, 128]和[bs, 31, 128] 输出是[bs, 1, 128]

        self.channel_encoder = Channel_Encoder(256, 16)
        self.channel_layer = Channels()

    # def forward(self, inputs, key_ebd, n_std, training=False, enc_padding_mask=None):
    #     inputs_ebd = self.embedding(inputs)
    #
    #     semantic_enc_output = self.semantic_encoder(inputs_ebd, training, enc_padding_mask)
    #     semantic_enc_output_trim = semantic_enc_output[:, :-1, :]
    #
    #     semantic_enc_output_and_key_ebd = torch.cat([semantic_enc_output_trim, key_ebd], dim=1)  # 将秘钥和语义编码拼接
    #
    #     enc_padding_mask_pading = torch.zeros(
    #         (enc_padding_mask.size(0), enc_padding_mask.size(1), enc_padding_mask.size(2), key_ebd.size(1)),
    #         device=inputs.device)  # 创建指定维度的全0张量
    #     enc_padding_mask_expended = torch.cat([enc_padding_mask[:, :, :, :-1], enc_padding_mask_pading], dim=3)
    #
    #     mac_output = self.mac_encoder(semantic_enc_output_and_key_ebd, training, enc_padding_mask_expended)
    #     mac_output_trim = mac_output[:, :semantic_enc_output_trim.size(1), :]
    #
    #     channel_enc_output = self.channel_encoder(mac_output_trim)
    #
    #     received_channel_enc_output = self.channel_layer.awgn(channel_enc_output, n_std)
    #
    #     return received_channel_enc_output, channel_enc_output, mac_output_trim, semantic_enc_output_trim

    def forward(self, inputs, key_ebd, n_std, training=False, enc_padding_mask=None):
        inputs_ebd = self.embedding(inputs)

        semantic_enc_output = self.semantic_encoder(inputs_ebd, training, enc_padding_mask)
        # semantic_enc_output_trim = semantic_enc_output[:, :-1, :]


        mac_output = self.mac_encoder(key_ebd, semantic_enc_output)  # [bs, 1, 128]
        semantic_mac = torch.cat([semantic_enc_output, mac_output], dim=1)  # [bs, 11, 128]

        channel_enc_output = self.channel_encoder(semantic_mac)

        received_channel_enc_output = self.channel_layer.awgn(channel_enc_output, n_std)

        return received_channel_enc_output, channel_enc_output, semantic_mac


    def train_seman(self, inputs, training=False, enc_padding_mask=None):
        inputs_ebd = self.embedding(inputs)
        semantic_enc_output = self.semantic_encoder(inputs_ebd, training, enc_padding_mask)
        return semantic_enc_output

    # def train_mac(self, inputs, key_ebd, training=False, enc_padding_mask=None):
    #     inputs_ebd = self.embedding(inputs)
    #     inp_and_key_ebd = torch.cat([inputs_ebd, key_ebd], dim=1)
    #
    #     enc_padding_mask_pading = torch.zeros(
    #         (enc_padding_mask.size(0), enc_padding_mask.size(1), enc_padding_mask.size(2), key_ebd.size(1)),
    #         device=inputs.device)
    #     enc_padding_mask_expended = torch.cat([enc_padding_mask, enc_padding_mask_pading], dim=3)
    #
    #     mac_output = self.cipher_encoder(inp_and_key_ebd, training, enc_padding_mask_expended)
    #     return mac_output
    def train_mac(self, semantic_enc_output, key_ebd):
        return self.mac_encoder(key_ebd, semantic_enc_output)  # 先秘钥再特征 输出是[bs, 1, 128]

    def train_channel(self, inputs, n_std):
        inputs_ebd = self.embedding(inputs)
        channel_enc_output = self.channel_encoder(inputs_ebd)
        received_channel_enc_output = self.channel_layer.awgn(channel_enc_output, n_std)
        return received_channel_enc_output


class Receiver(nn.Module):
    def __init__(self, args):
        super(Receiver, self).__init__()
        self.channel_decoder = Channel_Decoder(args.decoder_d_model, 512)  # 假设Channel_Decoder已转换为PyTorch类

        self.embedding = nn.Embedding(num_embeddings=args.vocab_size, embedding_dim=args.encoder_d_model)

        # self.cipher_decoder = Decoder(args.decoder_num_layer, args.decoder_d_model, args.decoder_num_heads,
        #                               args.decoder_d_ff, args.vocab_size, dropout=args.decoder_dropout)
        self.mac_decoder = MAC_verify()

        # self.semantic_decoder = Decoder(args.decoder_num_layer, args.decoder_d_model, args.decoder_num_heads,
        #                                 args.decoder_d_ff, args.vocab_size, dropout_pro=args.decoder_dropout)
        self.semantic_decoder = Decoder(args.decoder_num_layer, args.vocab_size, args.MAX_LENGTH,
                args.decoder_d_model, args.decoder_num_heads, args.decoder_d_ff, args.decoder_dropout)

        self.dense = nn.Linear(args.decoder_d_model, args.vocab_size)
        self.final_layer = nn.Linear(in_features=args.encoder_d_model, out_features=args.vocab_size)

    def forward(self, received_channel_enc_output, tar_inp, key_ebd, look_ahead_mask, src_mask,
                                                            training=False, combined_mask=None):
        channel_dec_output = self.channel_decoder(received_channel_enc_output)
        tar_inp_ebd = self.embedding(tar_inp)

        # 首先分离
        semantic_enc_output = channel_dec_output[:, :31, :]  # 前31个通道
        mac = channel_dec_output[:, 31:, :]

        # 验证mac
        result = self.mac_decoder(semantic_enc_output, mac, key_ebd)

        # 对recovered_semantic_enc_output进行语义解码
        semantic_decoder_output = self.semantic_decoder(tar_inp, channel_dec_output, look_ahead_mask, src_mask)
        pred = self.dense(semantic_decoder_output)  # 和tf代码中的pred一样的维度 直接用



        # received_cipher_enc_output_and_key_ebd = torch.cat((received_channel_dec_output, key_ebd), dim=1)
        #
        # cipher_dec_output, _ = self.cipher_decoder(tar_inp_ebd, received_cipher_enc_output_and_key_ebd, training,
        #                                           combined_mask, None)
        #
        # semantic_decoder_output, _ = self.semantic_decoder(cipher_dec_output,
        #                                                   received_cipher_enc_output_and_key_ebd, training,
        #                                                   combined_mask, None)
        # predictions = self.final_layer(semantic_decoder_output)

        return pred, result

    def train_cipher(self, inps, tar_inp, key_ebd, training=False, combined_mask=None):
        tar_inp_ebd = self.embedding(tar_inp)
        inps_and_key_ebd = torch.cat((inps, key_ebd), dim=1)

        cipher_dec_output, _ = self.cipher_decoder(tar_inp_ebd, inps_and_key_ebd, training, combined_mask, None)
        predictions = self.final_layer(cipher_dec_output)
        return predictions

    def train_seman(self, inps, tar_inp, training=False, combined_mask=None):
        tar_inp_ebd = self.embedding(tar_inp)
        semantic_decoder_output, _ = self.semantic_decoder(tar_inp_ebd, inps, training, combined_mask, None)
        predictions = self.final_layer(semantic_decoder_output)
        return predictions

    def train_channel(self, received_channel_enc_output):
        received_channel_dec_output = self.channel_decoder(received_channel_enc_output)
        predictions = self.final_layer(received_channel_dec_output)
        return predictions


