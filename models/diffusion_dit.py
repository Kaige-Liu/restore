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
        
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
       
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        # c 是融合了时间 t 和 SNR 的条件向量: [bs, hidden_size]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        
        # Attention 分支
        x_norm1 = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(x_norm1, x_norm1, x_norm1)
        x = x + gate_msa.unsqueeze(1) * attn_out
        
        # FFN 分支
        x_norm2 = modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_out = self.mlp(x_norm2)
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        
        return x


class FeatureRestorationDiT(nn.Module):
    def __init__(
        self, 
        feature_dim=128,      # f 的维度 (128)
        seq_len=32,           # 序列长度 (31 + 1 IBSID = 32)
        hidden_size=256,      # DiT 内部隐藏层维度
        depth=4,              # Transformer 层数
        num_heads=8,
        snr_embed_dim=128
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.seq_len = seq_len
        
        self.x_embedder = nn.Linear(feature_dim * 2, hidden_size)
 
        # 让扩散模型知道它在处理的是第几个词的特征
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
        
        self.final_layer = nn.Sequential(
            nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6),
            nn.Linear(hidden_size, feature_dim)
        )
        
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        nn.init.constant_(self.final_layer[-1].weight, 0)
        nn.init.constant_(self.final_layer[-1].bias, 0)

    def forward(self, x_t, f_hat, t, snr, context_mask=None):
        bs = x_t.shape[0]
        
        # 将 SNR 强行平移为正数，防止正弦编码崩溃
        snr = snr + 15.0 
        
        if context_mask is not None:
            mask_expanded = context_mask.view(-1, 1, 1).expand(-1, self.seq_len, self.feature_dim)
            f_hat = f_hat.clone() 
            f_hat[mask_expanded] = self.null_feature.expand(bs, -1, -1)[mask_expanded]
        
        x = torch.cat([x_t, f_hat], dim=-1) # [bs, 32, 256]
        x = self.x_embedder(x)              # [bs, 32, hidden_size]
        
        # 将位置编码加到特征序列中 
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
            
        x = self.final_layer(x)             
        return x