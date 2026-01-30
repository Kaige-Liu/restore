import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv1d(nn.Module):
    """深度可分离卷积"""

    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(DepthwiseSeparableConv1d, self).__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size,
                                   padding=padding, groups=in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class LinearMultiHeadAttention(nn.Module):
    """线性复杂度的多头注意力机制"""

    def __init__(self, embed_dim, num_heads):
        super(LinearMultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Conv1d(embed_dim, embed_dim, kernel_size=1)
        self.k_proj = nn.Conv1d(embed_dim, embed_dim, kernel_size=1)
        self.v_proj = nn.Conv1d(embed_dim, embed_dim, kernel_size=1)
        self.out_proj = nn.Conv1d(embed_dim, embed_dim, kernel_size=1)

    def forward(self, x):
        batch_size, channels, seq_len = x.size()

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch_size, self.num_heads, self.head_dim, seq_len)
        k = k.view(batch_size, self.num_heads, self.head_dim, seq_len)
        v = v.view(batch_size, self.num_heads, self.head_dim, seq_len)

        k_prime = F.elu(k) + 1
        context = torch.matmul(k_prime, v.transpose(2, 3))
        attention = torch.matmul(q.transpose(2, 3), context)

        attention = attention.transpose(2, 3).contiguous().view(batch_size, channels, seq_len)
        output = self.out_proj(attention)

        return output


class BinaryDecisionNetwork(nn.Module):
    def __init__(self, key_channels=10, semantic_channels=31,
                 knowledge_channels=8, output_channels=1, seq_length=128, num_heads=4):
        super(BinaryDecisionNetwork, self).__init__()

        # 密钥处理
        self.key_processor = DepthwiseSeparableConv1d(key_channels, 8, kernel_size=3, padding=1)

        # 语义特征处理
        self.semantic_processor = nn.Sequential(
            LinearMultiHeadAttention(semantic_channels, num_heads),
            nn.LayerNorm([semantic_channels, seq_length]),
            nn.Conv1d(semantic_channels, 8, kernel_size=1)
        )

        # 知识库和映射处理
        self.knowledge_reduce = nn.Conv1d(knowledge_channels, 4, kernel_size=1)
        self.mapping_reduce = nn.Conv1d(knowledge_channels, 4, kernel_size=1)

        # 知识库组合处理
        self.knowledge_combiner = nn.Sequential(
            DepthwiseSeparableConv1d(8, 8, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # 新增输入处理
        self.output_processor = nn.Sequential(
            nn.Conv1d(output_channels, 8, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # 特征融合：对五个特征计算注意力权重
        self.attention_projection = nn.Conv1d(8 * 4, 4, kernel_size=1)

        # 输出层
        self.output_layer = nn.Sequential(
            nn.Conv1d(8, 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(4, 1, kernel_size=1),
            nn.Sigmoid()  # 输出0-1之间的概率值
        )

    def forward(self, key, semantic, knowledge, mapping, original_output):
        # 处理前四个输入特征
        key_processed = self.key_processor(key)
        semantic_processed = self.semantic_processor(semantic)

        # 知识库和映射降维后拼接
        knowledge_reduced = self.knowledge_reduce(knowledge)
        mapping_reduced = self.mapping_reduce(mapping)
        knowledge_combined = torch.cat([knowledge_reduced, mapping_reduced], dim=1)
        knowledge_processed = self.knowledge_combiner(knowledge_combined)

        # 处理新增输入
        output_processed = self.output_processor(original_output)

        # 拼接所有特征
        all_features = torch.cat([
            key_processed,
            semantic_processed,
            knowledge_processed,
            output_processed
        ], dim=1)  # [batch_size, 8*4, seq_length]

        # 计算注意力得分并应用softmax
        attention_scores = self.attention_projection(all_features)  # [batch_size, 4, seq_length]
        attention_weights = F.softmax(attention_scores, dim=1)  # 沿通道维度归一化

        # 分离注意力权重
        key_weight, semantic_weight, knowledge_weight, output_weight = torch.chunk(attention_weights, 4, dim=1)

        # 加权融合
        weighted_sum = (key_processed * key_weight +
                        semantic_processed * semantic_weight +
                        knowledge_processed * knowledge_weight +
                        output_processed * output_weight)

        # 全局平均池化，将序列压缩为一个特征向量
        pooled = F.avg_pool1d(weighted_sum, kernel_size=weighted_sum.size(2))

        # 输出层
        output = self.output_layer(pooled).squeeze(2)  # [batch_size, 1]

        return output


# 测试模型
def test_decision_model():
    model = BinaryDecisionNetwork(
        key_channels=10,
        semantic_channels=31,
        knowledge_channels=8,
        output_channels=1,
        seq_length=128,
        num_heads=4
    )

    batch_size = 16
    key = torch.randn(batch_size, 10, 128)
    semantic = torch.randn(batch_size, 31, 128)
    knowledge = torch.randn(batch_size, 8, 128)
    mapping = torch.randn(batch_size, 8, 128)
    original_output = torch.randn(batch_size, 1, 128)

    output = model(key, semantic, knowledge, mapping, original_output)

    print(f"输入形状:")
    print(f"  密钥: {key.shape}")
    print(f"  语义特征: {semantic.shape}")
    print(f"  知识库: {knowledge.shape}")
    print(f"  知识库映射: {mapping.shape}")
    print(f"  原始输出: {original_output.shape}")
    print(f"输出形状: {output.shape}")  # 应输出 [batch_size, 1]
    print(f"决策模型参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")


if __name__ == "__main__":
    test_decision_model()