import streamlit as st
import pandas as pd

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