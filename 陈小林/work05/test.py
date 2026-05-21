import torch
import torch.nn.functional as F


def generate_text(model, prompt_chars, char_to_idx, idx_to_char,
                  max_length=50, top_k=5, temperature=0.8):
    """使用全量模型生成文本"""
    model.eval()

    # 转换输入
    input_ids = [char_to_idx.get(ch, 0) for ch in prompt_chars]
    generated = list(prompt_chars)

    with torch.no_grad():
        for _ in range(max_length):
            # 前向传播
            x = torch.tensor([input_ids])
            logits = model(x)

            # 取最后一个位置
            last_logits = logits[-1] / temperature

            # Top-K采样
            top_k_logits, top_k_indices = torch.topk(last_logits, top_k)
            probs = F.softmax(top_k_logits, dim=-1)

            # 采样
            next_id = top_k_indices[torch.multinomial(probs, 1)].item()
            next_char = idx_to_char[next_id]

            if next_char == '[CLS]':
                break

            generated.append(next_char)
            input_ids.append(next_id)

    return ''.join(generated)


# 使用示例
def main():
    # 加载全量模型
    model = torch.load(r'model/full_model.pth', map_location='cpu', weights_only=False)
    model.eval()

    # 加载词表
    with open(r'data/chars.txt', 'r', encoding='utf-8') as f:
        vocab = [line.strip() for line in f.readlines()]

    char_to_idx = {ch: idx for idx, ch in enumerate(vocab)}
    idx_to_char = {idx: ch for idx, ch in enumerate(vocab)}

    # 生成文本
    result = generate_text(model, "昨天", char_to_idx, idx_to_char, max_length=50)
    print(result)


if __name__ == '__main__':
    main()