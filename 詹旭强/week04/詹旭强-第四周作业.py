import torch
import torch.nn as nn
import torch.nn.functional as fel
import math



class zxqMultiHeadAttention(nn.Module):

    def __init__(self,hidden_size,n_head,ff,dropout=0.1):
        super().__init__()
        assert hidden_size % n_head == 0, "hidden 维度必须能被 n_head 整除"
        self.n_head = n_head
        self.d_k = hidden_size//n_head #每个注意力头的维度

        #最底层注意力部分初始化
        self.qkv = nn.Linear(hidden_size,hidden_size*3)  #使用一个 Linear 一次性生成 Q, K, V，比拆分成3个 Linear 计算效率更高
        self.out = nn.Linear(hidden_size,hidden_size)
        self.dropt1 = nn.Dropout(dropout)


        #前馈网络部分初始化
        self.ln1 = nn.LayerNorm(hidden_size)  #前馈网络 第一个归一化层
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size,ff),  #升维度，为了补偿激活过程中丢失的信息
            nn.GELU(), 
            nn.Linear(ff,hidden_size)   #还原维度，保证输入输出一致

        )  #前馈网络 两个线性层，中间加个激活
        self.ln2 = nn.LayerNorm(hidden_size) # 第二个层归一化


    def forward(self, x):

        #输入的形状
        B, Q, H = x.shape
        # b,q,h -> b,q,n_head,d_k -> b,n_head,q,h ,view: 相当于reshape ，不改变底层结构 transpose(1,2):将第2、3维进行交换位置
        q,k,v = self.qkv(x).chunk(3,dim=-1) #将一次性初始化的大矩阵按最后一维拆为3份

        q = q.view(B,Q,self.n_head,self.d_k).transpose(1,2)
        k = k.view(B,Q,self.n_head,self.d_k).transpose(1,2)
        v = v.view(B,Q,self.n_head,self.d_k).transpose(1,2)

        scores = q @  k.transpose(-2, -1) / math.sqrt(self.d_k)
        attn =  fel.softmax(scores,dim=-1) 

        attn = self.dropt1(attn)

        out = attn @ v #加权求和

        #b,n_head,q,d_k -> b,q,n_head,d_k -> b,q,h
        out = out.transpose(1,2).contiguous().view(B, Q, H)
        result = self.out(out) #过一个线性层

        #前馈网络 第一个归一化层
        ln1_res = self.ln1(x+result) 
        ffn_res = self.ffn(ln1_res)
        #前馈网络 第一个归一化层 输出最终结果
        ln2_res = self.ln2(ln1_res+ffn_res)

        return ln2_res
    

if  __name__ == "__main__":

    model = zxqMultiHeadAttention(hidden_size = 768,n_head = 12,ff = 3072 ,dropout=0.1)

    x = torch.randn(2, 16, 768)
    #循环堆叠12次
    output = model(x)
    for _ in range(11):
        output = model(output)

print(f"输入形状: {x.shape}")   # torch.Size()
print(f"输出形状: {output.shape}") # torch.Size()
