import torch
import torch.nn as nn
import torch.nn.functional as F


# ========== 模型 ==========

class CausalLM(nn.Module):
    def __init__(self, vocab_size, hidden_size=128, num_heads=4, num_layers=4, max_len=32):
        super().__init__()
        self.max_len = max_len
        self.token_emb = nn.Embedding(vocab_size, hidden_size)
        self.pos_emb = nn.Embedding(max_len, hidden_size)
        # 直接用PyTorch内置的TransformerDecoderLayer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size, nhead=num_heads,
            dim_feedforward=hidden_size * 2, batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        seq_len = x.shape[1]
        # Embedding = token + position
        pos = torch.arange(seq_len, device=x.device)
        x = self.token_emb(x) + self.pos_emb(pos)
        # 因果mask：禁止看到未来的token
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)
        x = self.transformer(x, x, tgt_mask=mask, memory_mask=mask)
        return self.output_head(x)  # [batch, seq_len, vocab_size]


# ========== 数据准备 ==========

def build_dataset(corpus, max_len):
    vocab = ["<pad>"] + sorted(set(corpus))
    char_to_id = {c: i for i, c in enumerate(vocab)}
    id_to_char = {i: c for i, c in enumerate(vocab)}
    ids = [char_to_id[c] for c in corpus]
    # 滑动窗口构造样本：x=[0:n], y=[1:n+1]
    x_data, y_data = [], []
    for i in range(len(ids) - max_len):
        x_data.append(ids[i:i + max_len])
        y_data.append(ids[i + 1:i + max_len + 1])
    return torch.LongTensor(x_data), torch.LongTensor(y_data), vocab, char_to_id, id_to_char


# ========== 文本生成 ==========

def generate(model, start_text, char_to_id, id_to_char, max_gen_len=40):
    model.eval()
    ids = [char_to_id[c] for c in start_text]
    with torch.no_grad():
        for _ in range(max_gen_len):
            x = torch.LongTensor([ids[-model.max_len:]])
            logits = model(x)[0, -1] / 0.6  # temperature=0.6
            # top-k采样
            top_vals, top_ids = logits.topk(5)
            probs = F.softmax(top_vals, dim=-1)
            next_id = top_ids[torch.multinomial(probs, 1)].item()
            ids.append(next_id)
    return "".join(id_to_char[i] for i in ids)


# ========== 训练 ==========

def train():
    max_len = 32
    epochs = 50
    batch_size = 64

    corpus = ("天青色等烟雨而我在等你炊烟袅袅升起隔江千万里\n"
              "在瓶底书汉隶仿前朝的飘逸就当我为遇见你伏笔\n"
              "天青色等烟雨而我在等你月色被打捞起晕开了结局\n"
              "如传世的青花瓷自顾自美丽你眼带笑意\n"
              "色白花青的锦鲤跃然于碗底临摹宋体落款时却惦记着你\n"
              "你隐藏在窑烧里千年的秘密极细腻犹如绣花针落地\n"
              "帘外芭蕉惹骤雨门环惹铜绿而我路过那江南小镇惹了你\n"
              "在泼墨山水画里你从墨色深处被隐去\n"
              "白月光心里某个地方那么亮却那么冰凉\n"
              "每个人都有一段悲伤想隐藏却在生长\n"
              "白月光照天涯的两端在心上却不在身旁\n"
              "如果你的回忆不是对我的牢笼我愿意化作那道光\n"
              "从前从前有个人爱你很久但偏偏风渐渐把距离吹得好远\n"
              "好不容易又能再多爱一天但故事的最后你好像还是说了拜拜")

    x_data, y_data, vocab, char_to_id, id_to_char = build_dataset(corpus, max_len)
    print(f"词表大小: {len(vocab)}, 样本数: {len(x_data)}")

    model = CausalLM(vocab_size=len(vocab), max_len=max_len)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        idx = torch.randperm(len(x_data))
        total_loss, n = 0, 0
        for i in range(0, len(x_data), batch_size):
            batch_x = x_data[idx[i:i + batch_size]]
            batch_y = y_data[idx[i:i + batch_size]]
            logits = model(batch_x)
            loss = loss_fn(logits.view(-1, len(vocab)), batch_y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n += 1
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss/n:.4f}")
            print(f"  生成: {generate(model, '天青色', char_to_id, id_to_char)}")

    print("\n===== 最终生成 =====")
    for start in ["天青色", "白月光", "从前", "如果"]:
        print(f"  「{start}」→ {generate(model, start, char_to_id, id_to_char)}")


if __name__ == "__main__":
    train()
