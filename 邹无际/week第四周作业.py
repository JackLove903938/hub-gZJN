import torch
import torch.nn as nn
import math


class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层：多头自注意力 + 前馈网络"""
    
    def __init__(self, hidden_size=768, num_attention_heads=12, intermediate_size=3072, dropout_rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        
        # 多头自注意力层
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.attention_dropout = nn.Dropout(dropout_rate)
        self.attention_output = nn.Linear(hidden_size, hidden_size)
        self.attention_layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.attention_output_dropout = nn.Dropout(dropout_rate)
        
        # 前馈神经网络层
        self.intermediate_dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = nn.GELU()
        self.output_dense = nn.Linear(intermediate_size, hidden_size)
        self.ffn_dropout = nn.Dropout(dropout_rate)
        self.ffn_layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.ffn_output_dropout = nn.Dropout(dropout_rate)
    
    def transpose_for_scores(self, x, batch_size):
        """将张量重塑为多头格式: [batch, seq, hidden] -> [batch, heads, seq, head_size]"""
        seq_len = x.size(1)
        x = x.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states, attention_mask=None):
        """前向传播: [batch, seq, hidden] -> [batch, seq, hidden]"""
        batch_size = hidden_states.size(0)
        seq_len = hidden_states.size(1)
        
        # 自注意力机制
        residual_input = hidden_states
        query_layer = self.transpose_for_scores(self.query(hidden_states), batch_size)
        key_layer = self.transpose_for_scores(self.key(hidden_states), batch_size)
        value_layer = self.transpose_for_scores(self.value(hidden_states), batch_size)
        
        # 计算注意力分数 Q*K^T
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.attention_dropout(attention_probs)
        
        # 注意力输出
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.view(batch_size, seq_len, self.hidden_size)
        
        attention_output = self.attention_output_dropout(self.attention_output(context_layer))
        attention_output = self.attention_layer_norm(residual_input + attention_output)
        
        # 前馈网络
        residual_input = attention_output
        ffn_output = self.intermediate_act_fn(self.intermediate_dense(attention_output))
        ffn_output = self.ffn_output_dropout(self.output_dense(ffn_output))
        layer_output = self.ffn_layer_norm(residual_input + ffn_output)
        
        return layer_output


class TransformerEncoder(nn.Module):
    """完整的Transformer编码器，由多层TransformerEncoderLayer堆叠而成"""
    
    def __init__(self, num_layers=12, hidden_size=768, num_attention_heads=12, 
                 intermediate_size=3072, dropout_rate=0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_size, num_attention_heads, intermediate_size, dropout_rate)
            for _ in range(num_layers)
        ])
    
    def forward(self, hidden_states, attention_mask=None):
        """前向传播，依次通过每一层"""
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states


# 测试
if __name__ == "__main__":
    torch.manual_seed(42)
    
    # 创建测试数据
    batch_size, seq_len, hidden_size = 2, 10, 768
    input_tensor = torch.randn(batch_size, seq_len, hidden_size)
    print(f"输入形状: {input_tensor.shape}")
    
    # 测试单个Transformer层
    transformer_layer = TransformerEncoderLayer(hidden_size, 12, 3072, 0.1)
    transformer_layer.eval()
    with torch.no_grad():
        output = transformer_layer(input_tensor)
    print(f"单层输出形状: {output.shape}")
    
    # 测试12层完整编码器
    transformer_encoder = TransformerEncoder(12, hidden_size, 12, 3072, 0.1)
    transformer_encoder.eval()
    with torch.no_grad():
        output = transformer_encoder(input_tensor)
    print(f"12层输出形状: {output.shape}")
    
    total_params = sum(p.numel() for p in transformer_encoder.parameters())
    print(f"总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # 测试带注意力掩码
    attention_mask = torch.ones(batch_size, 1, 1, seq_len)
    attention_mask[1, 0, 0, 7:] = -10000.0
    with torch.no_grad():
        output_masked = transformer_encoder(input_tensor, attention_mask)
    print(f"带掩码输出形状: {output_masked.shape}")

