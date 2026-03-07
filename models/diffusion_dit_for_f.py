import torch
import torch.nn as nn
import math


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#  DiT Block (集成 AdaLN-Zero)

class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_size)
        )

        # AdaLN 调制网络：生成 6 个参数 (缩放、平移、门控 各一对)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        # 提取当前层需要的条件控制参数
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)

        # 自注意力层 + AdaLN 控制
        norm1_x = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(norm1_x, norm1_x, norm1_x)
        x = x + gate_msa.unsqueeze(1) * attn_out

        # MLP 层 + AdaLN 控制
        norm2_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_out = self.mlp(norm2_x)
        x = x + gate_mlp.unsqueeze(1) * mlp_out

        return x


# 主架构：特征恢复 DiT
class FeatureRestorationDiT(nn.Module):  # 创新点
    def __init__(
            self,
            feature_dim=16,  # f 的维度 (通信信道物理特征 16 维)
            seq_len=32,  # 序列长度
            hidden_size=256,  # DiT 内部隐藏层维度
            depth=4,  # Transformer 层数
            num_heads=8,
            snr_embed_dim=128
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.seq_len = seq_len

        self.x_embedder = nn.Linear(feature_dim * 2, hidden_size)

        # 绝对位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, hidden_size))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.t_embedder = nn.Sequential(
            SinusoidalPosEmb(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.snr_embedder = nn.Sequential(
            SinusoidalPosEmb(snr_embed_dim),
            nn.Linear(snr_embed_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.null_feature = nn.Parameter(torch.zeros(1, seq_len, feature_dim))
        self.null_snr = nn.Parameter(torch.zeros(1, hidden_size))

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads) for _ in range(depth)
        ])

        # 最后一层也需要 AdaLN 调制
        self.final_layer = nn.Sequential(
            nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6),
            nn.Linear(hidden_size, feature_dim)
        )
        self.adaLN_modulation_final = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # AdaLN 零初始化 (Zero-Initialization)
        # 强迫网络刚开始就像透明的一样，极大提升训练稳定性
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.adaLN_modulation_final[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation_final[-1].bias, 0)
        nn.init.constant_(self.final_layer[-1].weight, 0)
        nn.init.constant_(self.final_layer[-1].bias, 0)

    def forward(self, x_t, f_hat, t, snr, context_mask=None):
        bs = x_t.shape[0]
        snr = snr + 15.0

        if context_mask is not None:
            mask_expanded = context_mask.view(-1, 1, 1).expand(-1, self.seq_len, self.feature_dim)
            f_hat = f_hat.clone()
            f_hat[mask_expanded] = self.null_feature.expand(bs, -1, -1)[mask_expanded]

        x = torch.cat([x_t, f_hat], dim=-1)
        x = self.x_embedder(x)
        x = x + self.pos_embed

        t_emb = self.t_embedder(t)
        snr_emb = self.snr_embedder(snr)

        if context_mask is not None:
            mask_snr = context_mask.view(-1, 1).expand_as(snr_emb)
            snr_emb = snr_emb.clone()
            snr_emb[mask_snr] = self.null_snr.expand(bs, -1)[mask_snr]

        c = t_emb + snr_emb

        for block in self.blocks:
            x = block(x, c)

        # 最后一层的去噪调制
        shift, scale = self.adaLN_modulation_final(c).chunk(2, dim=1)
        x = modulate(self.final_layer[0](x), shift, scale)
        x = self.final_layer[1](x)

        return x