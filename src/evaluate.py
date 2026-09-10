import sacrebleu
from rouge_score import rouge_scorer
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
from model import Seq2SeqModel

sp = spm.SentencePieceProcessor('../tokenizer/ur_sp.model')

UNK_ID = sp.unk_id()
BOS_ID = sp.bos_id()
EOS_ID = sp.eos_id()
SP_PAD_ID = sp.pad_id()
VOCAB = sp.get_piece_size()

def score(hyps, refs):
    bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    rl = sum(scorer.score(r, h)["rougeL"].fmeasure  for h, r in zip( hyps , refs ) ) / len( refs )
    unk_rate = sum(h.count("\u2047") for h in hyps ) / max(1, sum(len( h . split ()) for h in hyps))

    return {'BLEU-4': bleu, 'ROUGE-L': rl, 'unk_rate': unk_rate}


def test_model(model, loader, loss, device):
    model.to(device)
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in loader:
            src = batch['src'].to(device)
            src_lens = batch['src_lens']
            tgt = batch['tgt'].to(device)

            out, attn = model(src, src_lens, tgt, is_train=True)
            logits = out # [B, T-1, V]
            labels = tgt[:, 1:] # [B, T-1]

            step_loss = loss(logits.reshape(-1, VOCAB), labels.reshape(-1))
            mask = (labels != SP_PAD_ID)
            n_tokens = mask.sum().item()
            total_loss += step_loss.item() * n_tokens

            pred = logits.detach().argmax(dim=-1)
            correct = (pred == labels) & mask
            total_correct += correct.sum().item()
            total_tokens += n_tokens

    accuracy = total_correct / total_tokens
    nll = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(nll)).item()

    print(
        f"test_loss = {nll:.4f} | "
        f"test_ppl = {perplexity:.4f} | "
        f"test_acc = {accuracy:.4f}"
    )

    return nll, perplexity, accuracy

# load saved model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Seq2SeqModel(in_dim=VOCAB, V=VOCAB, H=512, E=256, dropout=0.3)
state_dict = torch.load('../checkpoints/best_model.pt', map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()
loss = nn.CrossEntropyLoss(ignore_index=SP_PAD_ID)

test_scores, test_loss, test_acc = test_model(model, val_loader, loss, device)