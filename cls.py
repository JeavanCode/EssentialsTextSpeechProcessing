import os.path
import pandas
import torch
import tiktoken
import torch.utils
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from torchvision.ops import sigmoid_focal_loss

from utils import encode_label, encode_text, get_schedule
from gpt2 import GPTCls, Config
from evaluate import evaluate

if __name__ == '__main__':
    config = Config()
    train = pandas.read_csv(os.path.join(config.root, r"data\train.tsv"), encoding="utf-8", delimiter='\t', header=None)
    valid = pandas.read_csv(os.path.join(config.root, r"data\dev.tsv"), encoding="utf-8", delimiter='\t', header=None)
    test = pandas.read_csv(os.path.join(config.root, r"data\test.tsv"), encoding="utf-8", delimiter='\t', header=None)
    tr_examples, va_examples, te_examples = train.shape[0], valid.shape[0], test.shape[0]
    tokenizer = tiktoken.get_encoding("gpt2")
    tr_labels, va_labels, te_labels = encode_label(train.iloc[:, 1], cls_size=config.cls_size), encode_label(valid.iloc[:, 1], cls_size=config.cls_size), encode_label(test.iloc[:, 1], cls_size=config.cls_size)
    tr_texts, va_texts, te_texts = train.iloc[:, 0], valid.iloc[:, 0], test.iloc[:, 0]
    tr_tokens, va_tokens, te_tokens = encode_text(tr_texts, tokenizer, config.max_length), encode_text(va_texts, tokenizer, config.max_length), encode_text(te_texts, tokenizer, config.max_length)
    print(f"n train examples: {tr_examples}, n valid examples: {va_examples}, n test examples: {te_examples}")

    tr_dataset, va_dataset, te_dataset = torch.utils.data.TensorDataset(tr_tokens, tr_labels), torch.utils.data.TensorDataset(va_tokens, va_labels), torch.utils.data.TensorDataset(te_tokens, te_labels)
    tr_loader, va_loader, te_loader = torch.utils.data.DataLoader(tr_dataset, batch_size=config.batch_size, shuffle=True), torch.utils.data.DataLoader(va_dataset, batch_size=config.batch_size, shuffle=False), torch.utils.data.DataLoader(te_dataset, batch_size=config.batch_size, shuffle=False)
    total_steps = len(tr_loader) * config.epochs

    model = GPTCls(config=config).to(config.device)
    loss_fn = sigmoid_focal_loss
    optimizer = torch.optim.AdamW(model.parameters())
    writer = SummaryWriter(log_dir=f"runs/gpt2_pre_dropout_3")

    for epoch in range(config.epochs):
        for step, (x, y) in enumerate(tr_loader):
            model.train()
            global_step = epoch * len(tr_loader) + step
            if global_step >= 10000:
                break
            c_lr = get_schedule(step=global_step, warmup_steps=total_steps // 100, schelude_steps=total_steps, init_value=config.init_lr, final_value=config.final_lr)
            c_wd = config.init_wd + (global_step / total_steps) * (config.final_wd - config.init_wd)
            for param_group in optimizer.param_groups:
                param_group['weight_decay'] = c_wd
            for param_group in optimizer.param_groups:
                param_group['lr'] = c_lr
            x, y = x.to(config.device), y.to(config.device)
            logits = model(x)
            batch_loss = loss_fn(logits, y, alpha=0.75, gamma=2.0)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            if global_step % 10 == 0:
                print(f"step {global_step}, batch loss: {batch_loss.item():.4f}, lr: {optimizer.param_groups[0]['lr']:.8f}, wd: {optimizer.param_groups[0]['weight_decay']:.8f}")
                writer.add_scalar("train_loss", batch_loss, global_step)
        valid_loss, valid_acc, valid_precision, valid_recall, valid_f1 = evaluate(model=model, valid_loader=va_loader, device=config.device, criteria=loss_fn, cls_size=config.cls_size)
        writer.add_scalar("valid_loss", valid_loss, epoch)
        writer.add_scalar("valid_accuracy", valid_acc, epoch)
        writer.add_scalar("valid_precision", valid_precision, epoch)
        writer.add_scalar("valid_recall", valid_recall, epoch)
        writer.add_scalar("valid_f1", valid_f1, epoch)
        torch.save(model.state_dict(), rf"{epoch}.pth")
        if epoch >= 1:
            os.remove(rf"{epoch-1}.pth")
        print(f"model saved at epoch {epoch}")






