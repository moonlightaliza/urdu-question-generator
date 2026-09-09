import pandas as pd 
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
from torch.utils.data import Dataset, DataLoader
from model import Seq2SeqModel

train_set = pd.read_csv("../data/train.tsv", delimiter='\t', header=None, names=['source', 'target'])
val_set = pd.read_csv("../data/valid.tsv", delimiter='\t', header=None, names=['source', 'target'])

# load tokenizer
sp = spm.SentencePieceProcessor(model_file='../tokenizer/ur_sp.model')

UNK_ID = sp.unk_id()
BOS_ID = sp.bos_id()
EOS_ID = sp.eos_id()
SP_PAD_ID = sp.pad_id()
VOCAB = 8000


class UrduDataset(Dataset):
    def __init__(self, df, src_col, tgt_col):
        self.sources = df[src_col].tolist()
        self.targets = df[tgt_col].tolist()

    
    def __len__(self):
        return len(self.sources)

    def __getitem__(self, idx):
        return (self.sources[idx], self.targets[idx])


def collate_function(batch):
    sources,targets = zip(*batch)

    src_ids = [sp.encode(s, out_type=int) for s in sources]
    tgt_ids = [sp.encode(t, out_type=int) for t in targets]

    src_lens = [len(s) for s in src_ids]
    tgt_lens = [len(t) for t in tgt_ids]

    src_padded = torch.nn.utils.rnn.pad_sequence([torch.tensor(src, dtype=torch.long) for src in src_ids], batch_first=True, padding_value=SP_PAD_ID)
    tgt_padded = torch.nn.utils.rnn.pad_sequence([torch.tensor(tgt, dtype=torch.long) for tgt in tgt_ids], batch_first=True, padding_value=SP_PAD_ID)

    return {
        'src': src_padded,
        'src_lens': src_lens,
        'tgt': tgt_padded,
        'tgt_lens': tgt_lens 
    }


train_data = UrduDataset(train_set, src_col="source", tgt_col="target")
val_data = UrduDataset(val_set, src_col="source", tgt_col="target")

train_loader = DataLoader(train_data, batch_size=64, shuffle=True, collate_fn=collate_function)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False, collate_fn=collate_function)

model = Seq2SeqModel(in_dim=sp.get_piece_size(), V=VOCAB, H=512, E=256, dropout=0.3)

loss = torch.nn.CrossEntropyLoss(ignore_index=SP_PAD_ID)
optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

# train loop
def train_run(model, loader, optim, loss, device):
    model.to(device)
    model.train()

    total_correct, total_tokens, total_loss = 0, 0, 0
    for batch in loader:
        optim.zero_grad()

        src = batch['src'].to(device)
        src_lens = batch['src_lens'].to(device)
        tgt = batch['tgt'].to(device)
        tgt_lens = batch['tgt_lens'].to(device)

        out, attn = model(src, src_lens, tgt)
        logits = out[:, :-1, :] # logits: [B, T-1, V]
        labels = tgt[:, 1:] # targets: [B, T-1]

        step_loss = loss(logits.reshape(-1, VOCAB), labels.reshape(-1))
        total_loss += step_loss.item()
        step_loss.backward()

        optim.step()

        pred = logits.detach().argmax(dim=-1)
        mask = (labels != SP_PAD_ID)
        correct = (pred == labels) & mask 
        total_correct += correct.sum().item()
        total_tokens += mask.sum().item()
    
    # training accuracy
    accuracy = total_correct / total_tokens
    print(f"tr_loss = {total_loss / len(loader):.4f} | tr_acc = {accuracy:.4f}")

    return total_loss, accuracy
    

# validation loop
def val_run(model, loader, loss, device):
    model.to(device)
    model.eval()

    total_correct, total_tokens, total_loss = 0, 0, 0
    with torch.no_grad():
        for batch in loader:
            src = batch['src'].to(device)
            src_lens = batch['src_lens'].to(device)
            tgt = batch['tgt'].to(device)
            tgt_lens = batch['tgt_lens'].to(device)

            out, attn = model(src, src_lens, tgt)
            logits = out[:, :-1, :] # logits: [B, T-1, V]
            labels = tgt[:, 1:] # targets: [B, T-1]

            step_loss = loss(logits.reshape(-1, VOCAB), labels.reshape(-1))
            total_loss += step_loss.item()
            
            pred = logits.detach().argmax(dim=-1)
            mask = (labels != SP_PAD_ID)
            correct = (pred == labels) & mask 
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()
    
    # validation accuracy
    accuracy = total_correct / total_tokens
    print(f"val_loss = {total_loss / len(loader):.4f} | val_acc = {accuracy:.4f}")

    return total_loss, accuracy  


def train_model(model, train_loader, val_loader, optimizer, loss, device, epochs):
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # train run
        tr_loss, tr_acc = train_run(model, train_loader, optimizer, loss, device)
        train_losses.append(tr_loss)
        
        # validation run
        val_loss, val_acc = val_run(model, val_loader, loss, device)
        val_losses.append(val_loss) 

