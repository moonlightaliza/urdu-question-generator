import streamlit as st
import pandas as pd
import sentencepiece as spm
import sys
import torch
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR.parent / "src"))
from model import Seq2SeqModel
sp = spm.SentencePieceProcessor(model_file=str(BASE_DIR.parent / "src" / "ur_sp.model"))

MODEL_CONFIG = {
    "checkpoint_path": BASE_DIR.parent / "src" / "best_model.pt",
    "in_dim": 8000,
    "V": 8000,
    "H": 512,
    "E": 256,
}

model = Seq2SeqModel(
    in_dim=MODEL_CONFIG["in_dim"],
    V=MODEL_CONFIG["V"],
    H=MODEL_CONFIG["H"],
    E=MODEL_CONFIG["E"],
)
model.load_state_dict(torch.load(str(MODEL_CONFIG["checkpoint_path"]), map_location="cpu"))
model.eval()


st.title("URDU QUESTION GENERATOR")
urdu_sentence = st.text_input("Enter an Urdu sentence with answer marked: ")
answer_text = st.text_input("Enter the answer text: ")
a_start = urdu_sentence.find(answer_text)

if(a_start == -1):
    st.write("Answer not found in sentence")
else:
    before = urdu_sentence[:a_start]
    after = urdu_sentence[a_start + len(answer_text):]
    marked_sentence = before + "<ans>" + answer_text + "</ans>" + after
    st.write("Marked sentence: ", marked_sentence)
    token_ids = sp.encode(marked_sentence)
    token_tensors = torch.tensor([token_ids], dtype = torch.long)
    tokens_len = [len(token_ids)]

    with torch.no_grad():
        greedy_out, greedy_attn = model(token_tensors, tokens_len, target=None, is_train=False, decoding='greedy')
        beam_ids = model(token_tensors, tokens_len, target=None, is_train=False, decoding='beam', beam_size=3)

    greedy_predicted_ids = greedy_out.argmax(dim=-1)[0].tolist()
    greedy_question = sp.decode(greedy_predicted_ids)

    beam_question = sp.decode(beam_ids)

    st.write("Greedy output:", greedy_question)
    st.write("Beam search output:", beam_question)







