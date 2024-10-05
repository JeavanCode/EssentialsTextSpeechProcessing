import os.path
import pandas
import torch
import tiktoken
import torch.utils
import torch.utils.data
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torchvision.ops import sigmoid_focal_loss

from utils import encode_label, encode_text, get_schedule, GoEmotionConfig
from gpt2 import GPTCls
from evaluate import evaluate

def cls(config):
    os.environ['PYTHONHASHSEED'] = str(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    g = torch.Generator()
    g.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

    train = pandas.read_csv(os.path.join(config.root, r"data\train.tsv"), encoding="utf-8", delimiter='\t', header=None)
    valid = pandas.read_csv(os.path.join(config.root, r"data\dev.tsv"), encoding="utf-8", delimiter='\t', header=None)
    test = pandas.read_csv(os.path.join(config.root, r"data\test.tsv"), encoding="utf-8", delimiter='\t', header=None)
    tr_examples, va_examples, te_examples = train.shape[0], valid.shape[0], test.shape[0]
    tokenizer = tiktoken.get_encoding("gpt2")
    tr_labels, va_labels, te_labels = encode_label(train.iloc[:, 1], cls_size=config.cls_size), encode_label(valid.iloc[:, 1], cls_size=config.cls_size), encode_label(test.iloc[:, 1], cls_size=config.cls_size)
    tr_texts, va_texts, te_texts = list(train.iloc[:, 0]), list(valid.iloc[:, 0]), list(test.iloc[:, 0])
    tr_tokens, tr_lengths = encode_text(tr_texts, tokenizer, config.max_length)
    va_tokens, va_lengths = encode_text(va_texts, tokenizer, config.max_length)
    te_tokens, te_lengths = encode_text(te_texts, tokenizer, config.max_length)
    print(f"n train examples: {tr_examples}, n valid examples: {va_examples}, n test examples: {te_examples}")
    tr_dataset, va_dataset, te_dataset, trva_dataset = torch.utils.data.TensorDataset(tr_tokens, tr_lengths, tr_labels), torch.utils.data.TensorDataset(va_tokens, va_lengths, va_labels), torch.utils.data.TensorDataset(te_tokens, te_lengths, te_labels), torch.utils.data.TensorDataset(torch.cat([tr_tokens, va_tokens], dim=0), torch.cat([tr_lengths, va_lengths], dim=0), torch.cat([tr_labels, va_labels], dim=0))
    tr_loader, va_loader, te_loader, trva_loader = torch.utils.data.DataLoader(tr_dataset, batch_size=config.batch_size, shuffle=True, generator=g), torch.utils.data.DataLoader(va_dataset, batch_size=config.batch_size, shuffle=False), torch.utils.data.DataLoader(te_dataset, batch_size=config.batch_size, shuffle=False), torch.utils.data.DataLoader(trva_dataset, batch_size=config.batch_size, shuffle=True, generator=g)

    model = GPTCls(config=config).to(config.device)
    loss_fn = sigmoid_focal_loss if config.loss_fn == "focal" else torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters())
    writer = SummaryWriter(log_dir=config.run_name)

    train_loader = tr_loader
    evaluate_loader = va_loader
    total_steps = len(train_loader) * config.epochs
    for epoch in range(config.epochs):
        for step, (x, l, y) in enumerate(train_loader):
            model.train()
            global_step = epoch * len(train_loader) + step
            c_lr = get_schedule(step=global_step, warmup_steps=100, schelude_steps=total_steps, init_value=config.init_lr, final_value=config.final_lr)
            c_wd = config.init_wd + (global_step / total_steps) * (config.final_wd - config.init_wd)
            for param_group in optimizer.param_groups:
                param_group['weight_decay'] = c_wd
            for param_group in optimizer.param_groups:
                param_group['lr'] = c_lr
            x, y = x.to(config.device), y.to(config.device)

            logits = model(x, l)
            if config.loss_fn == "focal":
                batch_loss = loss_fn(logits, y, alpha=config.alpha, gamma=2.0, reduction='mean')
            else:
                batch_loss = loss_fn(logits, y)

            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if global_step % 10 == 0:
                print(f"step {global_step}, batch loss: {batch_loss.item():.4f}, lr: {optimizer.param_groups[0]['lr']:.8f}, wd: {optimizer.param_groups[0]['weight_decay']:.8f}")
                writer.add_scalar("train_loss", batch_loss, global_step)
        valid_loss, valid_acc, valid_precision, valid_recall, valid_f1 = evaluate(config=config, model=model, valid_loader=evaluate_loader, criteria=loss_fn)
        writer.add_scalar("valid_loss", valid_loss, epoch)
        writer.add_scalar("valid_accuracy", valid_acc, epoch)
        writer.add_scalar("valid_precision", valid_precision, epoch)
        writer.add_scalar("valid_recall", valid_recall, epoch)
        writer.add_scalar("valid_f1", valid_f1, epoch)
        state_dict = {'backbone': model.backbone.state_dict(), 'cls': model.cls_head.state_dict()}
        torch.save(state_dict, f"goemotion_{epoch}.pth")
        if epoch >= 1:
            os.remove(rf"goemotion_{epoch-1}.pth")
        print(f"model saved at epoch {epoch}")
    return valid_f1

if __name__ == '__main__':
    search = False
    if search:
        f1s = []
        lrs = [1e-5, 5e-5, 1e-4]
        wds = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
        best_f1, best_lr, best_wd = None, None, None
        for i in range(len(lrs)):
            for j in range(len(wds)):
                config = GoEmotionConfig(init_lr=lrs[i], final_wd=wds[j], run_name=r'runs/gpt2_large_goemotion' + f'_lr{lrs[i]}' + f'_wd{wds[j]}')
                current_f1 = cls(config=config)
                f1s.append(current_f1)
                if current_f1 >= max(f1s):
                    best_f1, best_lr, best_wd = current_f1, lrs[i], wds[j]
                    print(f'best model is at lr {lrs[i]}, wd {wds[j]}, with f1 {best_f1}')
    else:
        config = GoEmotionConfig()
        cls(config=config)
    






