import pandas as pd
import sentencepiece as spm
from torch.utils.data import Dataset, DataLoader
from model import Seq2SeqModel
import matplotlib.pyplot as plt
import torch
import os
import time

# load tokenizer
sp = spm.SentencePieceProcessor('tokenizer/ur_sp.model')

UNK_ID = sp.unk_id()
BOS_ID = sp.bos_id()
EOS_ID = sp.eos_id()
SP_PAD_ID = sp.pad_id()
VOCAB = sp.get_piece_size()


class UrduDataset(Dataset):
    def __init__(self, df, src_col, tgt_col):
        self.sources = df[src_col].tolist()
        self.targets = df[tgt_col].tolist()


    def __len__(self):
        return len(self.sources)

    def __getitem__(self, idx):
        return (self.sources[idx], self.targets[idx])


def make_collate(sampling):
    def _collate(batch):
        sources, targets = zip(*batch)
        src_ids = [sp.encode(s, out_type=int, enable_sampling=sampling, alpha=0.5, nbest_size=-1) for s in sources]
        tgt_ids = [[BOS_ID] + sp.encode(t, out_type=int, enable_sampling=sampling, alpha=0.5, nbest_size=-1) + [EOS_ID] for t in targets]
        src_lens = [len(s) for s in src_ids]
        tgt_lens = [len(t) for t in tgt_ids]
        src_padded = torch.nn.utils.rnn.pad_sequence([torch.tensor(s, dtype=torch.long) for s in src_ids], batch_first=True, padding_value=SP_PAD_ID)
        tgt_padded = torch.nn.utils.rnn.pad_sequence([torch.tensor(t, dtype=torch.long) for t in tgt_ids], batch_first=True, padding_value=SP_PAD_ID)
        return {'src': src_padded, 'src_lens': src_lens, 'tgt': tgt_padded, 'tgt_lens': tgt_lens}
    return _collate


train_data = UrduDataset(train_set, src_col="source", tgt_col="target")
val_data = UrduDataset(val_set, src_col="source", tgt_col="target")

train_loader = DataLoader(train_data, batch_size=64, shuffle=True, collate_fn=make_collate(sampling=True))
val_loader = DataLoader(val_data, batch_size=64, shuffle=False, collate_fn=make_collate(sampling=False))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Seq2SeqModel(in_dim=sp.get_piece_size(), V=VOCAB, H=512, E=256, dropout=0.3).to(device)

loss = torch.nn.CrossEntropyLoss(ignore_index=SP_PAD_ID)
optim = torch.optim.Adam(model.parameters(), lr=1e-3)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.5, patience=1)

# train loop
def train_run(model, loader, optim, loss, device):
    model.to(device)
    model.train()

    total_tokens = 0
    total_nll = 0.0
    for batch in loader:
        optim.zero_grad()

        src = batch['src'].to(device, non_blocking=True)
        src_lens = batch['src_lens']
        tgt = batch['tgt'].to(device, non_blocking=True)

        out, attn = model(src, src_lens, tgt)
        logits = out # logits: [B, T-1, V]
        labels = tgt[:, 1:] # targets: [B, T-1]

        step_loss = loss(logits.reshape(-1, VOCAB), labels.reshape(-1))
        step_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optim.step()

        mask = (labels != SP_PAD_ID)
        n_tokens = mask.sum().item()
        total_nll += step_loss.item() * n_tokens
        total_tokens += n_tokens

    nll = total_nll / total_tokens
    print(f"tr_loss = {nll:.4f} | ", end="")

    return nll


# validation loop
def val_run(model, loader, loss, device):
    model.to(device)
    model.eval()

    total_tokens = 0
    total_nll = 0.0
    with torch.no_grad():
        for batch in loader:
            src = batch['src'].to(device)
            src_lens = batch['src_lens']
            tgt = batch['tgt'].to(device)

            out, attn = model(src, src_lens, tgt, is_train=True)
            logits = out # logits: [B, T-1, V]
            labels = tgt[:, 1:] # targets: [B, T-1]

            step_loss = loss(logits.reshape(-1, VOCAB), labels.reshape(-1))

            mask = (labels != SP_PAD_ID)
            n_tokens = mask.sum().item()
            total_nll += step_loss.item() * n_tokens
            total_tokens += n_tokens

    nll = total_nll / total_tokens
    print(f"val_loss = {nll:.4f}")

    return nll


def train_model(model, train_loader, val_loader, optimizer, loss, device, epochs):
    train_losses, val_losses = [], []
    best_val_loss = float("inf")

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        tr_loss = train_run(model, train_loader, optimizer, loss, device)
        train_losses.append(tr_loss)

        val_loss = val_run(model, val_loader, loss, device)
        val_losses.append(val_loss)

        sched.step(val_loss)

        if val_loss < best_val_loss:
            torch.save(model.state_dict(), 'checkpoints/best_model.pt')
            best_val_loss = val_loss

    return {
        "train_loss": train_losses,
        "val_loss": val_losses,
    }


if __name__ == '__main__':
    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    train_set = pd.read_csv("data/train.tsv", delimiter='\t', header=None, names=['source', 'target'])
    val_set = pd.read_csv("data/valid.tsv", delimiter='\t', header=None, names=['source', 'target'])

    start = time.time()
    history = train_model(model, train_loader, val_loader, optim, loss, device='cuda' if torch.cuda.is_available() else 'cpu', epochs=15)
    wall_clock = time.time() - start

    pd.DataFrame(history).to_csv("results/training_history.csv", index=False)

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train")
    plt.plot(epochs, history["val_loss"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("results/loss_curve.png", dpi=300, bbox_inches="tight")
