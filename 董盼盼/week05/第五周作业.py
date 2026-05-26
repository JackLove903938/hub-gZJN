import torch
import torch.nn as nn
import torch.optim as optim
from glob import glob
import os

# ===================== 超参数配置 =====================
batch_size = 8
seq_len = 100
d_model = 128
nhead = 8
num_layers = 2 #transformer 两层
lr = 1e-3
epochs = 200
gen_max_len = 200#生成文本长度
temperature = 0.75    # 采样温度 0~1越小越保守
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_save_path = "transformer_text_gen.pth"

# ===================== 1. 读取语料 + 构建词表 =====================
def load_corpus(pattern="corpus.txt"):
    texts = []
    for path in glob(pattern):#glob (pattern) = 按文件名规则找文件
        with open(path, encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
    return "".join(texts)

# 加载文本
full_text = load_corpus()
print(f"加载完成，总文本长度：{len(full_text)}")

# 构建字符映射
chars = sorted(set(full_text))
vocab_size = len(chars)
print(f"字表大小：{vocab_size}")
char2idx = {c: i for i, c in enumerate(chars)}
idx2char = {i: c for c, i in char2idx.items()}#从id找到字


# 全文转为数字序列
data = torch.tensor([char2idx[c] for c in full_text], dtype=torch.long)

# ===================== 2. 批量采样函数 =====================
def get_batch():
    batch_data = []
    for _ in range(batch_size):
        start = torch.randint(0, len(data) - seq_len - 1, (1,)).item()#随机一个数字作为起始位置
        batch_data.append(data[start:start + seq_len + 1])#从start开始在文本截取一段文字
    batch_data = torch.stack(batch_data)#把batch_size个文字拼在一起变成一个张量
    x = batch_data[:, :-1].to(device)#输入X去掉最后一个字
    y = batch_data[:, 1:].to(device)#输入y去掉第一个字 标签
    return x, y

# ===================== 3. 单向Transformer模型 =====================
class UniTransformerGen(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            batch_first=True,
            dropout=0.1,
            activation="gelu" 
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        b, s = x.shape
        # 因果掩码 屏蔽未来位置
        causal_mask = nn.Transformer.generate_square_subsequent_mask(s).to(device)#下三角 确保单向注意力
        # 位置编码 形状（b,s）
        pos = torch.arange(s, device=device).unsqueeze(0).expand(b, s)
        x_emb = self.embedding(x) + self.pos_emb(pos)#计算带位置信息的词嵌入
        feat = self.encoder(x_emb, mask=causal_mask)
        logits = self.fc_out(feat)
        return logits

# 模型、损失、优化器初始化
model = UniTransformerGen().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# ===================== 4. 模型保存 & 加载 =====================
def save_model():
    torch.save({
        "model_state_dict": model.state_dict(),
        "opt_state_dict": optimizer.state_dict(),
        "epoch": epochs
    }, model_save_path)
    print(f"模型已保存至 {model_save_path}")

def load_model():
    if os.path.exists(model_save_path):
        ckpt = torch.load(model_save_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["opt_state_dict"])
        print("成功加载预训练模型")
    else:
        print("无保存模型，从头训练")

# ===================== 5. 训练流程 =====================
def train():
    model.train()
    for epoch in range(epochs):
        x_batch, y_batch = get_batch()
        optimizer.zero_grad()
        pred_logits = model(x_batch)
        #模型预测输出形状: (batch_size, seq_len, vocab_size)
        ## 真实标签（下一个token），形状: (batch_size, seq_len)
        loss = criterion(pred_logits.reshape(-1, vocab_size), y_batch.reshape(-1))
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch:{epoch:3d}  Loss:{loss.item():.4f}")
    save_model()

# ===================== 6. 两种文本生成方式 =====================
# 6.1 贪心生成
def generate_greedy(start_txt, max_len=gen_max_len):
    model.eval()
    with torch.no_grad():
        seq = [char2idx[c] for c in start_txt if c in char2idx]
        seq_tensor = torch.tensor([seq], dtype=torch.long).to(device)
        for _ in range(max_len):
            if seq_tensor.shape[1] > seq_len:#seq_tensor.shape[1]当前序列长度
                seq_tensor = seq_tensor[:, -seq_len:]
            logits = model(seq_tensor)
            next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)#选择最大的概率
            seq_tensor = torch.cat([seq_tensor, next_id], dim=1)#新生成的token加在序列后面
        res = "".join([idx2char[i.item()] for i in seq_tensor[0]])#返回文本
    return res

# 6.2 温度+top_p采样生成
def generate_temp_top_p(start_txt, max_len=gen_max_len, temp=temperature, top_p=0.9):
    """
    温度 + Top-p (核采样) 生成
    - temp: 温度参数，控制分布平滑度
    - top_p: 累积概率阈值，只从累积概率达到 top_p 的最小词集中采样
    """
    model.eval()
    with torch.no_grad():
        seq = [char2idx[c] for c in start_txt if c in char2idx]
        seq_tensor = torch.tensor([seq], dtype=torch.long).to(device)
        
        for _ in range(max_len):
            if seq_tensor.shape[1] > seq_len:
                seq_tensor = seq_tensor[:, -seq_len:]
            
            logits = model(seq_tensor)[:, -1, :]  # 取最后一个位置的预测
            # 温度缩放
            logits = logits / temp
            probs = torch.softmax(logits, dim=-1)
            
            # === Top-p 实现 ===
            # 1. 按概率降序排序
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            
            # 2. 计算累积概率
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
            
            # 3. 找到累积概率超过 top_p 的位置
            mask = cumsum_probs > top_p
            # 保留至少一个 token（否则可能为空）
            mask[..., 1:] = mask[..., :-1].clone()# 把判断向后错一位，对齐正确位置
            mask[..., 0] = False # 保证概率最高的第一个字一定保留，不被杀
            sorted_probs[mask] = 0.0#把累计超过阈值之后的位置全部设为0
            
            # 4. 重新归一化概率
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            
            # 5. 采样
            sampled_idx = torch.multinomial(sorted_probs, num_samples=1)
            next_id = sorted_indices.gather(dim=-1, index=sampled_idx)
            
            seq_tensor = torch.cat([seq_tensor, next_id], dim=1)
        
        res = "".join([idx2char[i.item()] for i in seq_tensor[0]])
    return res

# ===================== 主执行入口 =====================
if __name__ == "__main__":
    # 加载已有模型，没有则训练
    load_model()
    # 开始训练
    train()

    # 测试两种生成方式
    prompt = "今日马来西亚"
    print("\n===== 贪心生成 =====")
    print(generate_greedy(prompt))

    print("\n===== 温度+top_p采样生成 =====")
    print(generate_temp_top_p(prompt))
