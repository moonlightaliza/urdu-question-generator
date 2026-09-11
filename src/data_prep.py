from datasets import load_dataset
from pathlib import Path
import matplotlib.pyplot as plt
import csv

BASE_DIR = Path(__file__).resolve().parent.parent  
FIGURES_DIR = BASE_DIR / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

ANS_OPEN, ANS_CLOSE = "<ans>", "</ans>"
SENT_DELIMS = "\u06D4\u061F!"

def split_sentences(text):
    start = 0
    for i, ch in enumerate(text):
        if ch in SENT_DELIMS:
            yield start, i + 1, text[start:i + 1]
            start = i + 1
    if start < len(text):
        yield start, len(text), text[start:]

def make_pair(example, max_src=60, max_tgt=25):
    a_text = example["answer"]
    if not a_text:
        return None
    context = example["context"]
    a_start = example["answer_start"]
    if a_start == -1:
        return None
    for s, e, sent in split_sentences(context):
        if s <= a_start < e:
            rel = a_start - s
            if sent[rel:rel + len(a_text)] != a_text:
                return None
            src = (sent[:rel] + " " + ANS_OPEN + " " + a_text + " "
                   + ANS_CLOSE + " " + sent[rel + len(a_text):]).strip()
            src = " ".join(src.split())
            tgt = " ".join(example["question"].split())
            if len(src.split()) > max_src or len(tgt.split()) > max_tgt:
                return None
            return src, tgt
    return None


#%%
def build_split(split, out_path):
    pairs = [p for p in map(make_pair, split) if p is not None]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
        w.writerows(pairs)
    print(f"{out_path}: {len(pairs)} pairs")
    return pairs

def plot_length_histogram(pairs, title, out_path):
    src_lens = [len(src.split()) for src, tgt in pairs]
    tgt_lens = [len(tgt.split()) for src, tgt in pairs]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(src_lens, bins=20)
    axes[0].set_title(f"{title} — source length")
    axes[0].set_xlabel("tokens")

    axes[1].hist(tgt_lens, bins=20)
    axes[1].set_title(f"{title} — target length")
    axes[1].set_xlabel("tokens")

    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved histogram to {out_path}")


if __name__ == "__main__":
    ds = load_dataset("uqa/UQA")
    print(ds)

    ex = ds["train"][0]
    print(ex.keys())
    print(ex["question"])
    print(ex["answer"])

    n_total = len(ds["train"])
    n_ans = sum(not ex["is_impossible"] for ex in ds["train"])
    print(f"train rows: {n_total}, answerable: {n_ans}")
    train_pairs = build_split(ds["train"], "train.tsv")
    valid_pairs = build_split(ds["validation"], "valid.tsv")
    

    plot_length_histogram(train_pairs, "Train", FIGURES_DIR / "train_length_hist.png")
    plot_length_histogram(valid_pairs, "Valid", FIGURES_DIR / "valid_length_hist.png")

# %%
