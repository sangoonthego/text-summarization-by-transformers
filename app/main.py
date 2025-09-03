import streamlit as st
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.summarizer import TextSummarizer
from utils import text_preprocess, file_reader
from constraints import max_len, min_len, temperature, top_k, top_p, chunk_size

summarizer = TextSummarizer()

st.set_page_config(page_title="Text Summarizer", layout="centered")
st.title("Text Summariztion by Using Transformer Application")

uploaded_file = st.file_uploader("Upload file (txt or pdf)", type=["txt", "pdf"])
text_input = st.text_area("Write Down here", height=200)

max_len = st.slider("Max Length", 50, 300, max_len, step=10)
min_len = st.slider("Min Length", 10, 100, min_len, step=5)

if st.button("Summarize"):
    raw_text = ""
    if uploaded_file is not None:
        raw_text = file_reader.read_file(uploaded_file)
    elif text_input.strip():
        raw_text = text_preprocess.clean_text(text_input)

    if raw_text:
        with st.spinner("Summarizing..."):
            # summary = summarizer.summarize(raw_text, max_len, min_len, temperature, top_k, top_p)
            summary = summarizer.chunk_text(raw_text, chunk_size=chunk_size)
        st.success("Summariztion Result: ")
        st.write(summary)
    else:
        st.warning("Upload File or Input Text befor Summarizing!!!")
            