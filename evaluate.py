import os

import pandas
import tiktoken
import torch
from torchmetrics.classification import MultilabelAccuracy, MultilabelPrecision, MultilabelRecall, MultilabelF1Score
from torchvision.ops import sigmoid_focal_loss

from gpt2 import Config, GPTCls
from utils import encode_label, encode_text


def evaluate(model, valid_loader, device, criteria, cls_size, threshold=0.5):
    model.eval()
    preds, labels, loss = [], [], 0
    metrics_acc = MultilabelAccuracy(num_labels=cls_size, average=None)
    metrics_pre = MultilabelPrecision(num_labels=cls_size, average=None)
    metrics_recall = MultilabelRecall(num_labels=cls_size, average=None)
    metrics_f1 = MultilabelF1Score(num_labels=cls_size, average=None)
    with torch.no_grad():
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss += criteria(logits, y, reduction='mean', alpha=0.75, gamma=2.0)
            
            probs = torch.sigmoid(logits)
            probs[probs < threshold] = 0
            probs[probs >= threshold] = 1
            preds.append(probs.to("cpu"))
            labels.append(y.to("cpu"))
    preds, labels = torch.cat(preds, dim=0), torch.cat(labels, dim=0)
    acc, pre, rec, f1 = metrics_acc(preds, labels).tolist(), metrics_pre(preds, labels).tolist(), metrics_recall(preds, labels).tolist(), metrics_f1(preds, labels).tolist()
    acc = [round(a, 4) for a in acc]
    pre = [round(a, 4) for a in pre]
    rec = [round(a, 4) for a in rec]
    f1 = [round(a, 4) for a in f1]
    loss /= len(valid_loader)
    print(f"accuracy: {acc}%\nprecision: {pre}\nrecall: {rec}\nf1: {f1}\nloss: {loss: .4f}")
    return loss, sum(acc)/len(acc), sum(pre)/len(pre), sum(rec)/len(rec), sum(f1)/len(f1)

if __name__ == '__main__':
    cls_size = 28
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 64
    max_length = 32
    root = r"C:\Users\jeavan\datasets\GoEmotions"

    config = Config(cls_size=cls_size)
    model = GPTCls(config=config).to(device)
    model.load_state_dict(torch.load("3.pth"))
    loss_fn = sigmoid_focal_loss

    valid = pandas.read_csv(os.path.join(root, r"data\dev.tsv"), encoding="utf-8", delimiter='\t', header=None)
    tokenizer = tiktoken.get_encoding("gpt2")
    va_labels= encode_label(valid.iloc[:, 1], cls_size=cls_size)
    va_texts = valid.iloc[:, 0]
    va_tokens = encode_text(va_texts, tokenizer, max_length)
    va_dataset = torch.utils.data.TensorDataset(va_tokens, va_labels)
    va_loader = torch.utils.data.DataLoader(va_dataset, batch_size=batch_size, shuffle=False)

    f1s = []
    for threshold in range(1, 20):
        threshold = threshold / 20
        valid_loss, valid_acc, valid_precision, valid_recall, valid_f1 = evaluate(model=model, valid_loader=va_loader, device=device, criteria=loss_fn, cls_size=cls_size, threshold=threshold)
        print(f"threshold: {threshold}, f1: {valid_f1: .4f}, loss: {valid_loss: .4f}, precision: {valid_precision: .4f}, recall: {valid_recall: .4f}, accuracy: {valid_acc: .4f}")
        f1s.append(valid_f1)
        if valid_f1 >= max(f1s):
            print(f"current best threshold: {threshold}")