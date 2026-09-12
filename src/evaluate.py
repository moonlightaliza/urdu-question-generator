import sacrebleu
from rouge_score import rouge_scorer
import torch
import torch.nn as nn
import sentencepiece as spm
from datasets import load_dataset
import unicodedata
from torch.utils.data import Dataset, DataLoader
from model import Seq2SeqModel
from data_prep import make_pair
import matplotlib.pyplot as plt

sp = spm.SentencePieceProcessor('tokenizer/ur_sp.model')

UNK_ID = sp.unk_id()
BOS_ID = sp.bos_id()
EOS_ID = sp.eos_id()
SP_PAD_ID = sp.pad_id()
VOCAB = sp.get_piece_size()

class TestDataset(Dataset):
    def __init__(self, pairs):
        self.ex = []

        for src, tgt in pairs:
            srce_ids = sp.encode(src, out_type=int)
            tgt_ids = ([BOS_ID] + sp.encode(tgt, out_type=int) + [EOS_ID])
            self.ex.append({
                'src_ids': srce_ids,
                'tgt_ids': tgt_ids,
                'src_text': src,
                'tgt_text': tgt
            })

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, idx):
        return self.ex[idx]


class UrduTokenizer:
    def tokenize(self, text):
        text = "".join(
            c if c.isalnum() or unicodedata.category(c).startswith("M") else " "
            for c in text.lower()
        )
        return text.split()



def test_collate(batch):
    src_ids = [torch.tensor(ex['src_ids']) for ex in batch]
    tgt_ids = [torch.tensor(ex['tgt_ids']) for ex in batch]

    src_lens = [len(s) for s in src_ids]
    src_pad = nn.utils.rnn.pad_sequence(src_ids, batch_first=True, padding_value=SP_PAD_ID)
    tgt_pad = nn.utils.rnn.pad_sequence(tgt_ids, batch_first=True, padding_value=SP_PAD_ID)

    return {
        "src": src_pad,
        "src_lens": src_lens,
        "tgt": tgt_pad,
        "src_text": [ex["src_text"] for ex in batch],
        "tgt_text": [ex["tgt_text"] for ex in batch],
    }


def score(hyps, refs):
    bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False, tokenizer=UrduTokenizer())
    rl = sum(scorer.score(r, h)["rougeL"].fmeasure  for h, r in zip( hyps , refs ) ) / len( refs )
    unk_rate = sum(h.count("\u2047") for h in hyps ) / max(1, sum(len( h . split ()) for h in hyps))

    return {'BLEU-4': bleu, 'ROUGE-L': rl, 'unk_rate': unk_rate}


def test_model(model, loader, loss, device):
    model.to(device)
    model.eval()
    total_loss = 0
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
            total_tokens += n_tokens

    nll = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(nll)).item()

    print(
        f"test_loss = {nll:.4f} | "
        f"test_ppl = {perplexity:.4f}"
    )

    return nll, perplexity


# BLEU score, ROUGEL, unknown rate
def decode_ids(ids):
    if torch.is_tensor(ids):
        ids = ids.tolist()
    if EOS_ID in ids:
        ids = ids[:ids.index(EOS_ID)]
    ids = [tkn for tkn in ids if tkn not in {BOS_ID, SP_PAD_ID, EOS_ID}]
    return sp.decode(ids)


def greedy_decode(model, loader, device):
    model.eval()
    hyps, refs, srcs = [], [], []

    with torch.no_grad():
        for batch in loader:
            src = batch['src'].to(device)
            src_lens = batch['src_lens']

            out, attn = model(src, src_lens, target=None, is_train=False)
            pred = out.argmax(dim=-1)
            for ids in pred:
                hyps.append(decode_ids(ids))

            refs.extend(batch['tgt_text'])
            srcs.extend(batch['src_text'])

    return srcs, hyps, refs


def beam_decode(model, loader, device, beam_size=3):
  model.eval()
  hyps = []

  with torch.no_grad():
    for batch in loader:
      for i in range(len(batch['src_lens'])):
          src = batch['src'][i:i+1].to(device)
          ids = model(src, [batch['src_lens'][i]], target=None, is_train=False, decoding='beam', beam_size=beam_size)
          hyps.append(decode_ids(ids))

  return hyps

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    uqa = load_dataset("uqa/UQA", split="validation")
    wuqa = load_dataset("uqa/Wiki-UQA", split="train")

    uqa_pairs = [pair for ex in uqa if (pair := make_pair(ex)) is not None]
    wuqa_pairs = [pair for ex in wuqa if (pair := make_pair(ex)) is not None]

    print(f"UQA validation pairs: {len(uqa_pairs)}")
    print(f"Wiki-UQA pairs: {len(wuqa_pairs)}")

    uqa_data_test = TestDataset(uqa_pairs)
    wuqa_data_test = TestDataset(wuqa_pairs)

    uqa_testloader = DataLoader(uqa_data_test, batch_size=64, shuffle=False, collate_fn=test_collate, pin_memory=True)
    wuqa_testloader = DataLoader(wuqa_data_test, batch_size=64, shuffle=False, collate_fn=test_collate, pin_memory=True)

    model = Seq2SeqModel(in_dim=VOCAB, V=VOCAB, H=512, E=256, dropout=0.3)
    state_dict = torch.load('checkpoints/best_model.pt', map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    model.eval()
    loss = nn.CrossEntropyLoss(ignore_index=SP_PAD_ID)

    print("Results on UQA evaluation:")
    uqa_loss, uqa_ppl = test_model(model, uqa_testloader, loss, device)
    print("\nResults on Wiki-UQA evaluation:")
    wqa_loss, wqa_ppl = test_model(model, wuqa_testloader, loss, device)

    uqa_srcs, uqa_greedy, uqa_refs = greedy_decode(model, uqa_testloader, device)
    wqa_srcs, wqa_greedy, wqa_refs = greedy_decode(model, wuqa_testloader, device)

    uqa_beam = beam_decode(model, uqa_testloader, device, beam_size=3)
    wqa_beam = beam_decode(model, wuqa_testloader, device, beam_size=3)
    print("Beam decode done.")

    uqa_scores_greedy = score(uqa_greedy, uqa_refs)
    uqa_scores_beam = score(uqa_beam, uqa_refs)
    print("UQA Decoding eval done..")

    wqa_scores_greedy = score(wqa_greedy, wqa_refs)
    wqa_scores_beam = score(wqa_beam, wqa_refs)
    print("WQA decoding eval done.")

    print("\nUQA validation (Greedy)")
    print("PPL:", uqa_ppl)
    print(uqa_scores_greedy)

    print("\nUQA validation (Beam)")
    print("PPL:", uqa_ppl)
    print(uqa_scores_beam)

    print("\nWiki-UQA validation (Greedy)")
    print("PPL:", wqa_ppl)
    print(wqa_scores_greedy)

    print("\nWiki-UQA validation (Beam)")
    print("PPL:", wqa_ppl)
    print(wqa_scores_beam)

    batch = next(iter(uqa_testloader))
    src = batch["src"].to(device)

    with torch.no_grad():
        out, attn = model(src, batch["src_lens"], target=None, is_train=False)

    i = 0
    pred = out[i].argmax(-1)
    src_ids = src[i, :batch["src_lens"][i]].cpu().tolist()
    pred_ids = pred.cpu().tolist()

    if EOS_ID in pred_ids:
        pred_ids = pred_ids[:pred_ids.index(EOS_ID) + 1]

    A = attn[i, :len(pred_ids), :len(src_ids)].cpu().numpy()

    plt.figure(figsize=(10, 6))
    plt.imshow(A, aspect="auto")
    plt.xticks(range(len(src_ids)), [sp.id_to_piece(x) for x in src_ids], rotation=90)
    plt.yticks(range(len(pred_ids)), [sp.id_to_piece(x) for x in pred_ids])
    plt.xlabel("Source tokens")
    plt.ylabel("Generated tokens")
    plt.tight_layout()
    plt.savefig("results/attention_heatmap.png", dpi=300, bbox_inches="tight")
