import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerEncoderLayer(nn.Module):
    """
    单个 Transformer 编码器层
    参数:
        d_model: 输入/输出的特征维度 (例如 512)
        nhead: 多头注意力的头数 (需要能整除 d_model)
        dim_feedforward: 前馈网络隐藏层维度 (通常为 2048)
        dropout: Dropout 比率 (默认 0.1)
        activation: 激活函数，支持 'relu' 或 'gelu' (默认 'relu')
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu'):
        super(TransformerEncoderLayer, self).__init__()
        
        # 多头自注意力
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # 前馈网络: 两个线性层 + 激活 + Dropout
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # 层归一化 (注意: 通常在残差连接之前或之后，这里采用 Pre-LN 或 Post-LN? 
        # 常见实现为 Post-LN (原始 Transformer)，但 Pre-LN 更稳定。下面实现 Post-LN)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # 激活函数
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'gelu':
            self.activation = F.gelu
        else:
            raise ValueError("activation 参数必须是 'relu' 或 'gelu'")
        
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        参数:
            src: 输入序列, shape (batch_size, seq_len, d_model)
            src_mask: 注意力掩码, shape (seq_len, seq_len) 或 (batch_size, seq_len, seq_len)
            src_key_padding_mask: padding 掩码, shape (batch_size, seq_len), True 表示需要屏蔽的位置
        返回:
            output: 输出张量, shape (batch_size, seq_len, d_model)
        """
        # 多头自注意力 + 残差连接 + 层归一化
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # 前馈网络 + 残差连接 + 层归一化
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src

# 使用示例
if __name__ == "__main__":
    # 超参数
    batch_size = 2
    seq_len = 10
    d_model = 512
    nhead = 8
    dim_feedforward = 2048
    
    # 创建层
    encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward)
    
    # 随机输入
    src = torch.rand(batch_size, seq_len, d_model)
    src_mask = None
    src_key_padding_mask = None  # 形状可为 (batch_size, seq_len) 的布尔张量
    
    # 前向传播
    output = encoder_layer(src, src_mask, src_key_padding_mask)
    print(f"输入形状: {src.shape} -> 输出形状: {output.shape}")
