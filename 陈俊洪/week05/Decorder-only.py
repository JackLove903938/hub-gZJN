import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import Dataset, DataLoader
import glob
import os

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int = 512, num_heads: int = 12,
                   dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads

        self.W_Q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_K = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_V = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_O = nn.Linear(embed_dim, embed_dim, bias=False)

        # 因果 mask：下三角为 True，表示"可见"
        causal_mask = torch.tril(torch.ones(max_len, max_len)).bool() # 这是bool矩阵,为下面的掩膜矩阵填充值进行一个预填充
        self.register_buffer("masked", causal_mask)

        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape

        # (B, N, D) -> (B, num_heads, N, head_dim)
        Q = self.W_Q(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)    # from (B, N, num_heads, head_dim) -> (B, num_heads, N, head_dim)
        K = self.W_K(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)        
        V = self.W_V(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)        

        # scores: (B, num_heads, N, N)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)     # K^T = (B, num_heads, head_dim, N)

        # 因果 mask：屏蔽未来 token
        mask = self.masked[:N, :N]                          # (N, N) 灵活调整句子的维度embeddings
        scores = scores.masked_fill(~mask, float('-inf'))   # 上三角填 -inf

        attn = F.softmax(scores, dim=-1)                    
        attn = self.attn_dropout(attn)

        context = torch.matmul(attn, V)                     # (B, h, N, head_dim)        
        context = context.transpose(1, 2).contiguous().view(B, N, D)  

        out = self.W_O(context)
        out = self.proj_dropout(out)
        return out

# 前馈神经网络层 
class FeedForward(nn.Module):
    def __init__(self, embed_dim: int = 768, hidden_dim: int = 3072, dropout: float = 0.1):
        super().__init__()

        # 第一层:升维 D → D_ff,扩大特征空间
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        # 第二层:降维 D_ff → D,投影回原维度
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 公式: FFN(x) = W_2 · Dropout(GELU(W_1·x + b_1)) + b_2
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))
    
# 完成的Transformer层
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int = 768, num_heads: int = 12,
                 ffn_dim: int = 3072, dropout: float = 0.1):
        super().__init__()
        # 残差神经网络 1
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        # 残差神经网络 2
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.ffn(x))
        return x

# Decoder-only
class DecoderOnlyLM(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 256, num_heads: int = 8,
                 ffn_dim: int = 1024, num_layers: int = 4,
                 max_len: int = 128, dropout: float = 0.1):
        super().__init__()
        # token 的 embeddings 把语料库中的字符增大维度包含embeddings
        self.tok_emb = nn.Embedding(vocab_size, embed_dim)
        # 词句embeddings, 能处理的语句长度的token的位置编码
        self.pos_emb = nn.Embedding(max_len, embed_dim)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ]) # 四个transformer块
        self.norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.max_len = max_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N = x.size(1)
        pos = torch.arange(N, device=x.device).unsqueeze(0)        # (1, N)
        h = self.drop(self.tok_emb(x) + self.pos_emb(pos))         # (B, N, D)
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h)
        return self.lm_head(h)                                     # (B, N, V)

def load_text(pattern ="*.txt"):
    texts = []
    for path in glob.glob(pattern):
        with open(path, encoding='utf-8', errors='ignore') as f:
            texts.append(f.read())
    return "".join(texts)

def build_vocab(text):
    """ 构建vocab字典 """
    chars = sorted(set(text))
    char2dix = {c: i for i, c in enumerate(chars)}
    dix2char = {i: c for i, c in enumerate(chars)}
    return char2dix, dix2char

class CharDataset(Dataset):
    def __init__(self, text, char2idx, seq_len):
        self.seq_len = seq_len                              # 语句长度
        ids = [char2idx[c] for c in text if c in char2idx]  
        self.data = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + 1: idx + self.seq_len + 1]
        return x, y
    
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)                                          # (B, N, V)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            reduction="sum",
        )
        total_loss += loss.item()
        total_tokens += y.numel()
    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(avg_loss)
    return avg_loss, ppl

@torch.no_grad()
def generate(model: nn.Module, prompt: str,
            char2idx: dict, idx2char: dict,
            device: torch.device,
            max_new_tokens: int = 200,
            temperature: float = 1.0,
            top_k: int = 0) -> str:
    """ 给定 prompt,自回归生成 max_new_tokens 个字符 """
    model.eval()
    # prompt 里没出现在词表的字符直接丢掉
    ids = [char2idx[c] for c in prompt if c in char2idx]
    if not ids:
        # prompt 全是未知字符,从词表第 0 个字符起头
        ids = [0]
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)

    for _ in range(max_new_tokens):
        # 截断到模型最大支持长度
        x_cond = x[:, -model.max_len:]
        logits = model(x_cond)                       # (1, T, V)
        logits = logits[:, -1, :] / max(temperature, 1e-8)  # 只看最后一步: (1, V)

        # top-k 过滤:除前 k 大的 logit 外全部置 -inf
        if top_k and top_k > 0:
            v, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
            logits[logits < v[:, -1:]] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)   # (1, 1)
        x = torch.cat([x, next_id], dim=1)

    out_ids = x[0].tolist()
    return "".join(idx2char[i] for i in out_ids)

def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
          device: torch.device, epochs: int = 5, lr: float = 3e-4):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "val_loss": [], "val_ppl": []}

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, running_tokens = 0.0, 0
        for step, (x, y) in enumerate(train_loader, 1):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
            )
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            # 计算token级的loss平均
            running_loss += loss.item() * y.numel()
            running_tokens += y.numel()
            if step % 50 == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f}")

        train_loss = running_loss / max(running_tokens, 1)
        val_loss, val_ppl = evaluate(model, val_loader, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_ppl"].append(val_ppl)
        print(f"[epoch {epoch}] train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} val_ppl={val_ppl:.2f}")

    return history

def main():
    seq_len = 64
    batch_size = 64
    epochs = 1
    lr = 5e-4
    val_ratio = 0.05

    file_path = os.path.join("./corpus.txt")
    text = load_text(file_path)
    # 构建字典
    char2idx, idx2char = build_vocab(text)
    # 按时间先后切分 train / val 文本(语言模型对位置敏感,不要随机打乱后再切)
    split = int(len(text) * (1 - val_ratio))
    train_text, val_text = text[:split], text[split:]

    train_set = CharDataset(text=train_text, char2idx=char2idx, seq_len=seq_len)
    val_set = CharDataset(text=val_text, char2idx=char2idx, seq_len=seq_len)
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size,
                            shuffle=False, drop_last=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DecoderOnlyLM(
        vocab_size=len(char2idx),
        embed_dim=128, num_heads=4, ffn_dim=512,
        num_layers=2, max_len=seq_len, dropout=0.1,
    )
    train(model, train_loader, val_loader, device, epochs=epochs, lr=lr)

    # 训练完跑一段生成,直观看模型有没有学到东西
    prompt = train_text[:20]
    print("\n========== generation demo ==========")
    print(f"prompt: {prompt!r}")
    sample = generate(model, prompt, char2idx, idx2char, device,
                      max_new_tokens=200, temperature=0.8, top_k=20)
    print(f"sample:\n{sample}")

if __name__ == "__main__":
    main()
