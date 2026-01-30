import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transceiver import MultiHeadedAttention, PositionwiseFeedForward, PositionalEncoding


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_angles(pos, i, d_model):
    angle_rates = pos / torch.pow(10000, ((2 * i) / torch.tensor(d_model, dtype=torch.float32)))
    return angle_rates

def postional_encoder(position, d_model):
    # 使用torch.arange生成位置序列
    angle_set = get_angles(torch.arange(position, dtype=torch.float32).unsqueeze(1),
                           torch.arange(d_model, dtype=torch.float32).unsqueeze(0),
                           d_model)

    # 使用torch.sin和torch.cos生成正弦和余弦值
    angle_set[:, 0::2] = torch.sin(angle_set[:, 0::2])
    angle_set[:, 1::2] = torch.cos(angle_set[:, 1::2])

    # 增加一个维度作为序列的表示
    pos_encoding = angle_set.unsqueeze(0)

    return pos_encoding.float()

class Sublayer1(nn.Module):
    def __init__(self, d_model, num_heads):
        super(Sublayer1, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads

        assert self.d_model % self.num_heads == 0

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.dense = nn.Linear(d_model, d_model)

    def scale_dot_product_attention(self, q, k, v, mask):
        matmul_qk = torch.matmul(q, k.transpose(-1, -2))  # (..., seq_len_q, seq_len_k)
        dk = q.size(-1)  # size of the last dimension of k
        scaled_dot_logits = matmul_qk / torch.sqrt(torch.tensor(dk, dtype=q.dtype))

        if mask is not None:
            scaled_dot_logits += (mask * -1e9)

        attention_weights = F.softmax(scaled_dot_logits, dim=-1)
        outputs = torch.matmul(attention_weights, v)
        return outputs, attention_weights

    def split_heads(self, x):
        batch_size = x.size(0)
        length = x.size(1)
        depth = self.d_model // self.num_heads

        x = x.view(batch_size, length, self.num_heads, depth)
        x = x.permute(0, 2, 1, 3)
        return x

    def combined_heads(self, x):
        batch_size = x.size(0)
        length = x.size(2)
        x = x.permute(0, 2, 1, 3)
        x = x.contiguous().view(batch_size, length, self.d_model)
        return x

    def forward(self, v, k, q, mask):
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        scaled_attention, attention_weights = self.scale_dot_product_attention(q, k, v, mask)
        attention_output = self.combined_heads(scaled_attention)
        multi_outputs = self.dense(attention_output)

        return multi_outputs, attention_weights



class Sublayer2(nn.Module):
    def __init__(self, d_model, dff):
        super(Sublayer2, self).__init__()
        self.d_model = d_model
        self.dff = dff
        self.linear1 = nn.Linear(d_model, dff)
        self.linear2 = nn.Linear(dff, d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)  # Activation function
        x = self.linear2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, drop_pro=0.1):
        super(EncoderLayer, self).__init__()

        self.sl1 = Sublayer1(d_model, num_heads)
        self.sl2 = Sublayer2(d_model, dff)

        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)

        self.dropout1 = nn.Dropout(drop_pro)
        self.dropout2 = nn.Dropout(drop_pro)

    def forward(self, x, training=True, mask=None):
        # Assuming that Sublayer1's forward method accepts mask as an argument
        attn_output, _ = self.sl1(x, x, x, mask)
        attn_output = self.dropout1(attn_output) if training else attn_output
        output1 = self.layernorm1(x + attn_output)

        ffn_output = self.sl2(output1)
        ffn_output = self.dropout2(ffn_output) if training else ffn_output
        output2 = self.layernorm2(output1 + ffn_output)

        return output2

class Encoder(nn.Module):
    def __init__(self, num_layers, num_heads, d_model, dff, input_vocab_size,
                 maximum_position_encoding=512, dropout_pro=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_encoding = postional_encoder(maximum_position_encoding, self.d_model)

        self.encoder = nn.ModuleList([EncoderLayer(d_model, num_heads, dff, dropout_pro)
                                             for _ in range(num_layers)])

        self.dropout = nn.Dropout(dropout_pro)

    def forward(self, x, training, mask):
        seq_len = x.size(1)
        x *= torch.sqrt(torch.tensor(self.d_model, dtype=torch.float))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x) if training else x

        for layer in self.encoder:
            x = layer(x, training, mask)

        return x

# class DecoderLayer(nn.Module):
#     def __init__(self, d_model, num_heads, dff, drop_pro=0.1):
#         super(DecoderLayer, self).__init__()
#
#         # 假设sublayer1和sublayer2已经在PyTorch中定义好了
#         self.sl11 = Sublayer1(d_model, num_heads)
#         self.sl12 = Sublayer1(d_model, num_heads)
#
#         self.ffn = Sublayer2(d_model, dff)
#
#         self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
#         self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
#         self.layernorm3 = nn.LayerNorm(d_model, eps=1e-6)
#
#         self.dropout1 = nn.Dropout(drop_pro)
#         self.dropout2 = nn.Dropout(drop_pro)
#         self.dropout3 = nn.Dropout(drop_pro)
#
#     def forward(self, x, enc_output, training, look_ahead_mask, padding_mask):
#         attn1, attn_weights1 = self.sl11(x, x, x, look_ahead_mask)
#         attn1 = self.dropout1(attn1) if training else attn1
#         output1 = self.layernorm1(x + attn1)
#
#         attn2, attn_weights2 = self.sl12(enc_output, enc_output, output1, padding_mask)
#         attn2 = self.dropout2(attn2) if training else attn2
#         output2 = self.layernorm2(attn2 + output1)
#
#         ffn_output = self.ffn(output2)
#         ffn_output = self.dropout3(ffn_output) if training else ffn_output
#         output3 = self.layernorm3(ffn_output + output2)
#
#         return output3, attn_weights1, attn_weights2
#
# class Decoder(nn.Module):
#     def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
#                  maximum_position_encoding=512, dropout_pro=0.1):
#         super(Decoder, self).__init__()
#
#         self.d_model = d_model
#         self.num_layers = num_layers
#
#         # 假设postional_encoder已经在PyTorch中定义好了
#         self.pos_encoding = postional_encoder(maximum_position_encoding, d_model)
#
#         self.dec_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, dff, dropout_pro)
#                                           for _ in range(num_layers)])
#         self.dropout = nn.Dropout(dropout_pro)
#
#     def forward(self, x, enc_output, training, look_ahead_mask, padding_mask):
#         seq_len = x.size(1)
#         attention_weights = {}
#         x *= torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
#         x += self.pos_encoding[:, :seq_len, :]
#
#         x = self.dropout(x) if training else x
#
#         for i in range(self.num_layers):
#             x, block1, block2 = self.dec_layers[i](x, enc_output, training,
#                                                    look_ahead_mask, padding_mask)
#
#         attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
#         attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2
#         return x, attention_weights

# 定义损失函数

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

class Decoder(nn.Module):
    def __init__(self, num_layers, trg_vocab_size, max_len,
                 d_model, num_heads, dff, dropout=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.embedding = nn.Embedding(trg_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len)
        self.dec_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, dff, dropout)
                                         for _ in range(num_layers)])

    def forward(self, tar_inp, channel_dec_output, look_ahead_mask, trg_padding_mask):
        x = self.embedding(tar_inp)
        x = self.pos_encoding(x)

        for dec_layer in self.dec_layers:
            x = dec_layer(x, channel_dec_output, look_ahead_mask, trg_padding_mask)

        return x




criterion = nn.CrossEntropyLoss(reduction='none')

def loss_function(real, pred):
    mask = torch.logical_not(torch.eq(real, 0))
    loss_ = criterion(real, pred)
    mask = mask.type(torch.float32)  # 如果mask不是布尔类型，需要先转换为布尔类型
    loss_ *= mask
    return torch.mean(loss_)

# 创建填充掩码
def create_padding_mask(seq):
    seq = (seq == 0).float()
    return seq.unsqueeze(1).unsqueeze(2)


def create_look_ahead_mask(size):
    mask = torch.ones(size, size).triu(1)  # 创建一个上三角矩阵
    return mask

# 创建掩码集合
def create_masks(inp, tar):
    enc_padding_mask = create_padding_mask(inp)
    dec_padding_mask = create_padding_mask(inp)
    look_ahead_mask = create_look_ahead_mask(tar.size(1))
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = torch.max(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask


