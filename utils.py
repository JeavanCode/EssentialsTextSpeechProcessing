import math
from dataclasses import dataclass

import torch


def encode_text(texts, tokenizer, max_length):
    n_examples = texts.shape[0]
    tokens = torch.zeros((n_examples, max_length), dtype=torch.int64)
    n = 0
    for i in range(n_examples):
        t = torch.tensor(tokenizer.encode(texts[i]), dtype=torch.int64)
        t_shape = t.shape[0]
        if t_shape > max_length:
            n += 1
        tokens[i, :min(max_length, t_shape)] = t[:min(max_length, t_shape)]
    print(f"{100 * n / n_examples: .2f}% of the examples exceed the max length {max_length}")
    return tokens


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


def get_schedule(step, schelude_steps, init_value, final_value, warmup_steps=0):
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
class Config:
    block_size: int = 1024
    vocab_size: int = 50257 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    cls_size: int = 0

    drop_out: float = 0.1
    batch_size: int = 16
    epochs: int = 4
    cls_size: int = 28
    init_lr: float = 1E-4
    final_lr: float = 1E-5
    init_wd: float = 0.
    final_wd: float = 1e-3
    max_length: int = 32
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    root: str = r"C:\Users\jeavan\datasets\GoEmotions"
    # root = r"D:\datasets\GoEmotions"