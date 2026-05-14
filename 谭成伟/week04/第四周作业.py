import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """多头自注意力机制"""

    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.hidden_size = hidden_size  # 隐藏层维度，如768
        self.num_heads = num_heads  # 注意力头数量，如12
        self.head_size = hidden_size // num_heads  # 每个头的维度，如64

        # Q、K、V的线性变换层
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)

        # 输出线性变换层
        self.output_linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        # x: [batch_size, seq_len, hidden_size]
        batch_size, seq_len, _ = x.size()

        # 线性变换得到Q、K、V
        q = self.q_linear(x)  # [batch, seq_len, hidden_size]
        k = self.k_linear(x)
        v = self.v_linear(x)

        # 分割成多个头：[batch, num_heads, seq_len, head_size]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)

        # 计算注意力分数：Q @ K^T / sqrt(head_size)
        # 输出: [batch, num_heads, seq_len, seq_len]
        # k 最后两维交换
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_size, dtype=torch.float32))

        # Softmax归一化
        attn_weights = F.softmax(scores, dim=-1)

        # 加权求和得到输出
        # 输出: [batch, num_heads, seq_len, head_size]
        output = torch.matmul(attn_weights, v)

        # 拼接多头输出：[batch, seq_len, hidden_size]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

        # 最后线性变换
        output = self.output_linear(output)
        return output, attn_weights


class TransformerLayer(nn.Module):
    """完整的Transformer层"""

    def __init__(self, hidden_size, num_heads, intermediate_size=3072, dropout=0.1):
        super().__init__()

        # 多头注意力
        self.attention = MultiHeadAttention(hidden_size, num_heads)

        # LayerNorm层
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),  # 升维
            nn.GELU(),  # GELU激活
            nn.Linear(intermediate_size, hidden_size)  # 降维
        )

        # Dropout层
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch_size, seq_len, hidden_size]

        # 多头注意力 + 残差连接 + LayerNorm
        attn_output, attn_weights = self.attention(x)
        # 这里也可以直接 x + attn_output
        x = self.norm1(x + self.dropout(attn_output))  # 残差连接

        # 前馈网络 + 残差连接 + LayerNorm
        ff_output = self.feed_forward(x)
        # 这里也可以直接 x + ff_output
        x = self.norm2(x + self.dropout(ff_output))  # 残差连接

        return x, attn_weights


# ------------------------------
# 测试代码
# ------------------------------
if __name__ == "__main__":
    # 参数设置（与bert-base-chinese一致）
    batch_size = 2
    seq_len = 10
    hidden_size = 768
    num_heads = 12

    # 创建Transformer层
    transformer_layer = TransformerLayer(hidden_size, num_heads)

    # 创建随机输入：[batch, seq_len, hidden_size]
    input_tensor = torch.randn(batch_size, seq_len, hidden_size)

    # 前向传播
    output, attn_weights = transformer_layer(input_tensor)

    # 打印输出形状
    print(f"输入形状: {input_tensor.shape}")
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {attn_weights.shape}")
    print("\nTransformer层实现成功！")
