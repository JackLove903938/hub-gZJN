"""
自己实现一个transform算子:encoder
"""
import torch
import torch.nn as nn
import random
import math


class SelfTransform():
    def __init__(self, model: nn.Module):
        ## 初始化参数
        # 基本参数
        layer = model.transform
        self.d_model = layer.self_attn.embed_dim
        self.nhead = layer.self_attn.num_heads
        self.dim_feedforward = layer.linear1.out_features
        self.activation = layer.activation

        # 多头注意力：qkv参数
        self.q_w = layer.self_attn.in_proj_weight[: self.d_model, :]
        self.k_w = layer.self_attn.in_proj_weight[self.d_model : 2 * self.d_model, :]
        self.v_w = layer.self_attn.in_proj_weight[2 * self.d_model :, :]
        self.q_b = layer.self_attn.in_proj_bias[: self.d_model]
        self.k_b = layer.self_attn.in_proj_bias[self.d_model : 2 * self.d_model]
        self.v_b = layer.self_attn.in_proj_bias[2 * self.d_model :]

        # 多头注意力：全连接层参数
        self.out_proj_w = layer.self_attn.out_proj.weight
        self.out_proj_b = layer.self_attn.out_proj.bias

        # 前馈神经网络：全连接层参数
        self.linear1_w = layer.linear1.weight
        self.linear1_b = layer.linear1.bias
        self.linear2_w = layer.linear2.weight
        self.linear2_b = layer.linear2.bias

        # 残差：LayerNorm参数
        self.norm1_w = layer.norm1.weight
        self.norm1_b = layer.norm1.bias
        self.norm2_w = layer.norm2.weight
        self.norm2_b = layer.norm2.bias

    def forward(self, x):
        """多头自注意力"""
        assert x.dim() == 3, f"x 应为 [B, H, W]，实际 {x.shape}"
        q = torch.nn.functional.linear(x, self.q_w, self.q_b)
        k = torch.nn.functional.linear(x, self.k_w, self.k_b)
        v = torch.nn.functional.linear(x, self.v_w, self.v_b)

        B, H, W = q.shape
        assert self.nhead > 0, f"nhead 必须 > 0,当前为 {self.nhead}"
        assert W % self.nhead == 0, f"W={W} 必须能整除 nhead={self.nhead}"

        # qkv矩阵
        head_dim = W // self.nhead
        q = q.reshape(B,H,self.nhead,head_dim).permute(0,2,1,3) # 整理后(B,nhead,H,head_dim)
        k = k.reshape(B,H,self.nhead,head_dim).permute(0,2,1,3)
        v = v.reshape(B,H,self.nhead,head_dim).permute(0,2,1,3)

        # k.transpose(-2,-1):交换最后2维
        # @：矩阵乘法，默认最后两维
        qk = q @ k.transpose(-2,-1)    # (B,nhead,H,H)

        # 归一化
        qk = torch.softmax(qk /math.sqrt(head_dim),dim=-1)

        qkv = qk @ v # (B,nhead,H,head_dim)

        # 拼接多头结果:(B,nhead,H,head_dim)-->(B,H,nhead,head_dim)-->(B,H,W)
        qkv = qkv.permute(0,2,1,3).reshape(B,H,W)

        # 输出多头结果:(B,H,W)
        multi_head_out = torch.nn.functional.linear(qkv, self.out_proj_w, self.out_proj_b)

        """残差计算1"""
        # normalized_shape=(W,)：对每个 x[b, h, :] 归一化
        residual_out1 = torch.nn.functional.layer_norm(
            x + multi_head_out,
            normalized_shape=(W,),
            weight=self.norm1_w,
            bias=self.norm1_b,
        )

        """前馈网络"""
        # (B,H,W)-->(B,H,hidden)
        feed_forward_out = torch.nn.functional.linear(residual_out1,self.linear1_w,self.linear1_b)
        feed_forward_out = self.activation(feed_forward_out)
        # (B,H,hidden)-->(B,H,W)
        feed_forward_out = torch.nn.functional.linear(feed_forward_out,self.linear2_w,self.linear2_b)

        """残差计算2"""
        residual_out2 = torch.nn.functional.layer_norm(
            residual_out1 + feed_forward_out,
            normalized_shape=(W,),
            weight=self.norm2_w,
            bias=self.norm2_b,
        )

        return residual_out2

class Transform(nn.Module):
    def __init__(self, batch_first, d_model, nhead, dim_feedforward, activation):
        super().__init__()
        ## TransformerEncoder
        # d_model：QKV参数
        # nhead：多头注意力机制的头数
        # dim_feedforward：前馈神经网络的隐藏层维度
        # activation：前馈神经网络的激活函数
        self.transform = nn.TransformerEncoderLayer(
            batch_first=batch_first,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation=activation,
        )

    def forward(self, x):
        return self.transform(x)


# 比较两个tensor是否相同，并打印误差
def compare_tensors(y_torch, y_self):
    diff = torch.abs(y_torch - y_self)
    max_error = diff.max().item()
    total_error = diff.sum().item()
    if torch.allclose(y_torch, y_self, atol=1e-6):
        print("两个transform算子输出结果相同！")
    else:
        print("两个transform算子输出结果不同！")
    print(f"最大误差: {max_error:.10f}")
    print(f"累计误差: {total_error:.10f}")

if __name__ == '__main__':
    random.seed(42)
    batch_size = 3
    seq_len = 128
    d_model = 768
    nhead = 12
    dim_feedforward = 3072
    activation = "gelu"
    batch_first = True

    x = torch.randn(batch_size, seq_len, d_model)

    # torch transform算子
    transform = Transform(batch_first, d_model, nhead, dim_feedforward, activation)
    transform.eval()
    y_torch = transform(x)

    # self transform算子
    self_transform = SelfTransform(transform)
    y_self = self_transform.forward(x)

    # 比较结果
    compare_tensors(y_torch, y_self)