import math
import torch
from dataclasses import dataclass

def encode_text(texts, tokenizer, max_length):
    n_examples = len(texts)
    tokens = torch.zeros((n_examples, max_length), dtype=torch.int64)
    lengths = torch.zeros(n_examples, dtype=torch.int64)
    n = 0
    for i in range(n_examples):
        t = torch.tensor(tokenizer.encode(texts[i]), dtype=torch.int64)
        t_shape = t.shape[0]
        if t_shape > max_length:
            n += 1
        tokens[i, :min(max_length, t_shape)] = t[:min(max_length, t_shape)]
        lengths[i] = min(max_length, t_shape)
    print(f"{100 * n / n_examples: .2f}% of the examples exceed the max length {max_length}")
    return tokens, lengths


def encode_label(df, cls_size):
    n_examples = df.shape[0]
    labels = torch.zeros((n_examples, cls_size), dtype=torch.float32)
    n = 0
    for i, ys in enumerate(df):
        ys = ys.split(",")
        if len(ys) > 1:
            n += 1
        for each_y in ys:
            labels[i, int(each_y)] = 1
    print(f"{100 * n / n_examples: .2f}% of the examples have more than 1 emotion")
    return labels

def get_schedule(step, schelude_steps, init_value, final_value, warmup_steps):
    # 1) linear warmup for warmup_iters steps
    if step < warmup_steps:
        return init_value * step / warmup_steps
    # 2) if it > schelude_steps, return min value
    if step > schelude_steps:
        return final_value
    # 3) in between, use cosine decay down to min value
    ratio = (step - warmup_steps) / (schelude_steps - warmup_steps)
    assert 0 <= ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))  # coeff ranges 0..1
    return final_value + coeff * (init_value - final_value)
    
@ dataclass
class GoEmotionConfig:
    pretrained:str = 'gpt2-large'
    ckpt:str = "goemotion_1.pth"
    cls_size: int = 28
    is_causal: bool = True
    is_pad_mask: bool = False

    seed: int = 0
    loss_fn: str = 'focal'
    dropout: float = 0.
    batch_size: int = 16
    if pretrained == 'gpt2':
        epochs: int = 4
    else:
        epochs: int = 2
    init_lr: float = 1e-5
    final_lr: float = 0.
    init_wd: float = 0.
    final_wd: float = 1e-4
    max_length: int = 48
    alpha: float = 0.75
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    root: str = r"C:\Users\jeavan\datasets\GoEmotions"
    run_name: str = r'runs\gpt2_large_goemotion'
    # root = r"D:\datasets\GoEmotions"

    def update_from_dict(self, config_args: dict):
        for key, value in config_args.items():
            setattr(self, key, value)