
#练习transformaer
import math

import torch
from torch import nn
from torch.nn import functional

#多头注意层
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, head_num):
        super().__init__()
        #断言，必须整数，否则直接报错
        assert hidden_size % head_num == 0
        #定义字符的维度，一般是768，embedding的特征值长度
        self.hidden_size = hidden_size
        #定义头数，768被切分成多少份
        self.head_num = head_num
        #每个头的维度
        self.d_k = hidden_size // head_num
        #q、k、v放在一个矩阵中，特征值、权重依旧分别计算
        self.qkv=nn.Linear(hidden_size, hidden_size*3)
        #输入维度、输出维度保持一致
        self.out = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, mask=None):
        #多少句话，一句话多少字，一个字多少维度
        Batch_size, T, H = x.shape
        #dim=-1代表最后一维，输入768，生成3*768，又分成3个768
        q,k,v=self.qkv(x).chunk(3,dim=-1)
        #形状[128,5,12,64]->[128,12,5,64]
        q=q.view(Batch_size,T,self.head_num,self.d_k).transpose(1,2)
        k=k.view(Batch_size,T,self.head_num,self.d_k).transpose(1,2)
        v=v.view(Batch_size,T,self.head_num,self.d_k).transpose(1,2)

        #k的形状[128,12,5,64]->[128,12,64,5]
        #矩阵乘法只算最后两个维度，[5,5]就是最核心的注意力矩阵
        scores=q@k.transpose(-2,-1)/math.sqrt(self.d_k)#输出是[128,12,5,5]

        #mask用来随机替换词，让模型预测被遮盖的原始词
        if mask is not None:
            scores = scores.masked_fill(mask==0, -1e9)
        #softmax函数
        attention=functional.softmax(scores,dim=-1)

        #[128,12,5,5]@[128,12,5,64]->[128,12,5,64]
        #qkv计算的最终结果
        out=scores@v

        #形状[128,12,5,64]->[128,5,12,64]->[128.5.768]
        out=out.transpose(1,2).contiguous().view(Batch_size,T,H)
        #底层call方法，执行线性层forward，输入多少维度，输出多少维度，这里是768
        return self.out(out)

#Transformer （流程）编码器层实现
class EncoderLayer(nn.Module):
    def __init__(self, hidden_size,n_head,ff):
        super().__init__()
        #生成多头注意层类
        self.attn = MultiHeadAttention(hidden_size,n_head)
        #归一化函数
        self.ln1 = nn.LayerNorm(hidden_size)
        #前馈网络计算，2个线性层，一个GELU激活函数
        self.ffn=nn.Sequential(nn.Linear(hidden_size,ff),
                               nn.GELU(),
                               nn.Linear(ff,hidden_size))
        #归一化函数
        self.ln2=nn.LayerNorm(hidden_size)

    #1、多头注意层输出+原始数据，归一化函数，第一次残差
    #2、前馈网络+多头注册层的输出，归一化函数，第二次残差
    def forward(self,x,mask=None):
        x=self.ln1(x+self.attn(x,mask))
        x=self.ln2(x+self.ffn(x))
        return x

#完整 Transformer 编码器的堆叠实现
#多层的EncoderLayer
class TransformerEncoder(nn.Module):
    #n_layer为编译器层数，即执行12遍transformer
    def __init__(self,hidden_size=768,n_layer=12,n_head=12,ff=3072):
        super().__init__()
        #用 nn.ModuleList 包裹 n_layer 个 EncoderLayer，实现多层编码器堆叠。
        self.layers=nn.ModuleList([EncoderLayer(hidden_size,n_head,ff) for _ in range(n_layer)])

    def forward(self,x,mask=None):
        #执行12层transformer
        for layer in self.layers:
            x=layer(x,mask)
        return x

if __name__=='__main__':
    model=TransformerEncoder(hidden_size=768,n_layer=12,n_head=12,ff=3072)
    #torch.rand是[0,1]的均匀分布随机张量，torch.randn是[-1,1]的正态分布随机张量，大多数接近0
    x=torch.randn(2,5,768)
    #输出形状
    print(model(x).shape)

