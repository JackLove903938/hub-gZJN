"""
BERT-Transformer 模型实现
基于标准BERT架构，包含12层Transformer Block叠加

架构流程:
    → Token Embedding + Position Embedding + Segment Embedding
    → 12 × Transformer Block (Multi-Head Attention → Add&Norm → Feed Forward → Add&Norm)
    → Output
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力机制"""

    def __init__(self, hidden_size=768, num_heads=12):
        # 调用父类nn.Module的初始化方法
        super(MultiHeadSelfAttention, self).__init__()
        # 断言hidden_size必须能被num_heads整除，确保每个头能分配到整数维度的特征
        assert hidden_size % num_heads == 0, "hidden_size必须能被num_heads整除"

        # 保存模型的隐藏层维度大小（例如768）
        self.hidden_size = hidden_size
        # 保存注意力头的数量（例如12）
        self.num_heads = num_heads
        # 计算每个注意力头的维度大小，即 768 / 12 = 64
        self.head_dim = hidden_size // num_heads

        # 定义Query的线性变换层，将输入映射为Q向量，维度保持不变(768 -> 768)
        self.W_q = nn.Linear(hidden_size, hidden_size)
        # 定义Key的线性变换层，将输入映射为K向量，维度保持不变(768 -> 768)
        self.W_k = nn.Linear(hidden_size, hidden_size)
        # 定义Value的线性变换层，将输入映射为V向量，维度保持不变(768 -> 768)
        self.W_v = nn.Linear(hidden_size, hidden_size)
        # 定义输出的线性变换层，将多头拼接后的结果映射回原始维度(768 -> 768)
        self.W_o = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, attention_mask=None):
        # 获取输入张量x的批次大小(batch_size)和序列长度(seq_len)，_代表hidden_size
        batch_size, seq_len, _ = x.size()

        # 将输入x通过W_q线性层，得到Query张量，形状为 [batch, seq_len, hidden_size]
        Q = self.W_q(x)
        # 将输入x通过W_k线性层，得到Key张量，形状为 [batch, seq_len, hidden_size]
        K = self.W_k(x)
        # 将输入x通过W_v线性层，得到Value张量，形状为 [batch, seq_len, hidden_size]
        V = self.W_v(x)

        # 将Q重塑并转置，拆分为多头: [batch, seq_len, hidden_size] -> [batch, seq_len, num_heads, head_dim] -> [batch, num_heads, seq_len, head_dim]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # 将K重塑并转置，拆分为多头，形状同Q
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # 将V重塑并转置，拆分为多头，形状同Q
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力得分: Q和K的转置进行矩阵乘法，并除以sqrt(head_dim)进行缩放，防止梯度消失
        # 形状: [batch, num_heads, seq_len, head_dim] x [batch, num_heads, head_dim, seq_len] -> [batch, num_heads, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 如果提供了注意力掩码（用于屏蔽padding部分或未来信息）
        if attention_mask is not None:
            # 扩展掩码维度以匹配scores的形状: [batch, seq_len] -> [batch, 1, 1, seq_len]
            extended_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # 将掩码转换为加法掩码：原本为1(有效)的位置变成0，原本为0(无效/padding)的位置变成-10000.0
            # 这样在softmax时，-10000的位置概率会接近0，从而实现屏蔽
            extended_mask = (1.0 - extended_mask.float()) * -10000.0
            # 将加法掩码加到注意力得分上
            scores = scores + extended_mask

        # 对最后一个维度(seq_len)进行softmax操作，得到注意力权重分布，概率和为1
        attn_weights = F.softmax(scores, dim=-1)
        # 将注意力权重与Value相乘，得到上下文向量
        # 形状: [batch, num_heads, seq_len, seq_len] x [batch, num_heads, seq_len, head_dim] -> [batch, num_heads, seq_len, head_dim]
        context = torch.matmul(attn_weights, V)

        # 将多头结果转置并拼接: [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, num_heads, head_dim] -> [batch, seq_len, hidden_size]
        # contiguous()确保张量在内存中是连续的，以便后续的view操作
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        # 将拼接后的结果通过输出线性层W_o，得到最终的多头注意力输出，形状为 [batch, seq_len, hidden_size]
        output = self.W_o(context)

        # 返回多头自注意力的输出
        return output


class FeedForward(nn.Module):
    """前馈神经网络"""

    def __init__(self, hidden_size=768, intermediate_size=3072):
        # 调用父类nn.Module的初始化方法
        super(FeedForward, self).__init__()
        # 第一层线性变换：升维，从隐藏层维度(768)映射到中间维度(3072)
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        # 第二层线性变换：降维，从中间维度(3072)映射回隐藏层维度(768)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x):
        # 输入x通过第一层线性层，维度变为 [batch, seq_len, 3072]
        x = self.fc1(x)
        # 使用GELU激活函数（BERT默认激活函数，比ReLU更平滑），增加非线性
        x = F.gelu(x)
        # 通过第二层线性层，维度变回 [batch, seq_len, 768]
        x = self.fc2(x)
        # 返回前馈神经网络的输出
        return x


class TransformerBlock(nn.Module):
    """
    单个Transformer Block
    流程: Multi-Head Attention → Add & Norm → Feed Forward → Add & Norm
    """

    def __init__(self, hidden_size=768, num_heads=12, intermediate_size=3072, dropout=0.1):
        # 调用父类nn.Module的初始化方法
        super(TransformerBlock, self).__init__()

        # 初始化多头自注意力层
        self.attention = MultiHeadSelfAttention(hidden_size, num_heads)
        # 初始化注意力层后的LayerNorm层，eps为防除零的极小值
        self.attention_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        # 初始化注意力层后的Dropout层，防止过拟合
        self.attention_dropout = nn.Dropout(dropout)

        # 初始化前馈神经网络层
        self.feed_forward = FeedForward(hidden_size, intermediate_size)
        # 初始化前馈网络后的LayerNorm层
        self.ff_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        # 初始化前馈网络后的Dropout层
        self.ff_dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        # 1. Multi-Head Self-Attention: 输入x通过注意力层
        attn_output = self.attention(x, attention_mask)
        # 对注意力输出进行Dropout
        attn_output = self.attention_dropout(attn_output)
        # 2. Add & Norm (残差连接 + LayerNorm): 将注意力输出与原始输入x相加，然后进行LayerNorm

        x = self.attention_norm(x + attn_output)

        # 3. Feed Forward: 将Add&Norm的输出送入前馈神经网络
        ff_output = self.feed_forward(x)
        # 对前馈网络输出进行Dropout
        ff_output = self.ff_dropout(ff_output)

        # 4. Add & Norm (残差连接 + LayerNorm): 将前馈输出与输入x相加，然后进行LayerNorm
        x = self.ff_norm(x + ff_output)
        # 返回Transformer Block的输出
        return x


class BertEmbeddings(nn.Module):
    """
    BERT嵌入层
    流程: Token Embedding + Position Embedding + Segment Embedding → LayerNorm → Dropout
    """

    def __init__(self, vocab_size=30522, hidden_size=768, max_position_embeddings=512, dropout=0.1):
        # 调用父类nn.Module的初始化方法
        super(BertEmbeddings, self).__init__()

        # 词嵌入层：将词表中的token ID映射为维度为hidden_size的向量
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        # 位置嵌入层：将位置索引(0~511)映射为维度为hidden_size的向量，提供位置信息
        self.position_embedding = nn.Embedding(max_position_embeddings, hidden_size)
        # 句子编号嵌入层（Segment/Type）：区分句子对中的前一句(0)和后一句(1)
        self.segment_embedding = nn.Embedding(2, hidden_size)

        # LayerNorm层：对三种嵌入相加后的结果进行归一化
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        # Dropout层：防止过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, token_type_ids=None):
        # 获取输入token ID的批次大小和序列长度
        batch_size, seq_len = input_ids.size()

        # 将输入的token ID转换为词嵌入向量，形状: [batch, seq_len, hidden_size]
        token_embeds = self.token_embedding(input_ids)

        # 生成位置ID序列：从0到seq_len-1，并扩展到batch_size大小，形状: [batch, seq_len]
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        # 将位置ID转换为位置嵌入向量，形状: [batch, seq_len, hidden_size]
        position_embeds = self.position_embedding(position_ids)

        # 如果没有提供句子编号ID，则默认全为0（即单句场景）
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        # 将句子编号ID转换为句子嵌入向量，形状: [batch, seq_len, hidden_size]
        segment_embeds = self.segment_embedding(token_type_ids)

        # 将三种嵌入向量按元素相加，得到最终的综合嵌入表示
        embeddings = token_embeds + position_embeds + segment_embeds
        # 对综合嵌入进行LayerNorm归一化
        embeddings = self.layer_norm(embeddings)
        # 对归一化后的结果进行Dropout
        embeddings = self.dropout(embeddings)

        # 返回嵌入层的输出
        return embeddings


class BertTransformer(nn.Module):
    """
    BERT-Transformer 完整模型
    流程: Input → Embeddings → 12 × TransformerBlock → Output

    标准BERT-Base配置:
    - 12层Transformer Block
    - 隐藏维度 768
    - 12个注意力头
    - 前馈网络中间维度 3072
    """

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_heads=12,
        num_layers=12,
        intermediate_size=3072,
        max_position_embeddings=512, # 最大处理句子长度
        dropout=0.1
    ):
        super(BertTransformer, self).__init__()

        self.hidden_size = hidden_size
        # Transformer Block的层数
        self.num_layers = num_layers

        # 初始化嵌入层
        self.embeddings = BertEmbeddings(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            max_position_embeddings=max_position_embeddings,
            dropout=dropout
        )

        # 使用ModuleList初始化12层 Transformer Block，这样可以方便地进行迭代
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                dropout=dropout
            )
            for _ in range(num_layers) # 循环num_layers次创建指定数量的Block
        ])

        # 初始化最终的LayerNorm层，用于在所有Transformer Block输出后进行最终归一化
        self.final_layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)


    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        # 1. Embedding层: 将输入的ID序列转换为连续的向量表示
        hidden_states = self.embeddings(input_ids, token_type_ids)

        # 2. 12层 Transformer Block 依次叠加: 将上一层的输出作为下一层的输入
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, attention_mask)

        # 3. 最终LayerNorm: 对最后一层Block的输出进行归一化
        sequence_output = self.final_layer_norm(hidden_states)

        # 4. 提取[CLS] token表示: 取每个序列第0个位置的向量，通常用于分类任务
        pooled_output = sequence_output[:, 0, :]

        # 返回完整的序列输出（用于序列标注等）和[CLS]的池化输出（用于分类等）
        return sequence_output, pooled_output


if __name__ == "__main__":
    # 实例化BERT-Transformer模型，使用标准BERT-Base参数(12层)
    model = BertTransformer(
        vocab_size=30522,          # 词表大小
        hidden_size=768,           # 隐藏层维度
        num_heads=12,              # 注意力头数
        num_layers=12,             # Transformer Block层数
        intermediate_size=3072,    # 前馈网络中间维度
        max_position_embeddings=512, # 最大序列长度
        dropout=0.1                # Dropout比例
    )

    # 打印分隔线
    print("=" * 60)
    # 打印标题
    print("BERT-Transformer 模型结构")
    # 打印分隔线
    print("=" * 60)
    # 打印模型的详细结构信息
    print(model)
    # 打印空行
    print()

    # 统计模型总参数量：遍历所有参数，计算元素个数并求和
    total_params = sum(p.numel() for p in model.parameters())
    # 打印总参数量，以千位分隔符显示，并换算为百万单位显示
    print(f"总参数量: {total_params:,} ({total_params / 1e6:.1f}M)")
    # 打印空行
    print()

    # 设置模拟输入的批次大小
    batch_size = 2
    # 设置模拟输入的序列长度
    seq_len = 32
    # 随机生成输入token ID，范围在0到30521之间，形状为 [batch_size, seq_len]
    input_ids = torch.randint(0, 30522, (batch_size, seq_len))
    # 生成全1的注意力掩码，表示所有token都是有效的（非padding），形状为 [batch_size, seq_len]
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    # 生成全0的句子编号ID，表示所有token都属于句子A（单句任务），形状为 [batch_size, seq_len]
    token_type_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)

    # 将模型设置为评估模式，这会关闭Dropout等训练时才使用的层
    model.eval()
    # 使用torch.no_grad()上下文管理器，在前向传播时不计算梯度，节省内存和计算资源
    with torch.no_grad():
        # 执行前向传播，获取序列输出和池化输出
        sequence_output, pooled_output = model(input_ids, attention_mask, token_type_ids)

    # 打印输入张量的形状
    print(f"输入 shape: {input_ids.shape}")
    # 打印序列输出张量的形状，应为 [batch_size, seq_len, hidden_size]
    print(f"序列输出 shape: {sequence_output.shape}")
    # 打印池化输出张量的形状，应为 [batch_size, hidden_size]
    print(f"池化输出 shape: {pooled_output.shape}")
    # 打印成功运行的信息
    print("BERT-Transformer 12层模型运行成功!")

