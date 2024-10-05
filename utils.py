import math
import torch
import tiktoken
from dataclasses import dataclass

from gpt2 import GPTCls

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

class EmotionDetector():
    def __init__(self, config):
        self.config = config
        self.emotions = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral']
        self.model = GPTCls(config=self.config).to(self.config.device)
        state_dict = torch.load("goemotion_1.pth", map_location=self.config.device)
        self.model.backbone.load_state_dict(state_dict['backbone'])
        self.model.cls_head.load_state_dict(state_dict['cls'])
        self.tokenizer = tiktoken.get_encoding("gpt2")
    
    def inference(self, text, threshold=0.4):
        self.model.eval()
        encoded_text = torch.tensor(self.tokenizer.encode(text)).unsqueeze(0).to(self.config.device)
        print(f'input shape: {encoded_text.shape}')
        logits = self.model(encoded_text)
        probs = logits.sigmoid()
        major_emotion, major_prob = self.emotions[torch.argmax(probs, dim=1).item()], torch.max(probs).item()
        preds = probs.clone()
        preds[preds < threshold] = 0
        preds[preds >= threshold] = 1
        preds, probs = preds.squeeze(0), probs.squeeze(0)
        all_emotions = {}
        for i, each_prob in enumerate(preds):
            if each_prob == 1:
                all_emotions[self.emotions[i]] = round(probs[i].item(), 4)
        all_emotions = sorted(all_emotions.items(), key=lambda x:x[1], reverse=True)
        print(f"all emotions: {all_emotions}, major emotion: {major_emotion}, major prob: {major_prob:.4f}")
        return all_emotions, major_emotion, major_prob
    
@ dataclass
class GoEmotionConfig:
    pretrained:str = 'gpt2-large'
    ckpt:str = None
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
