import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
@st.cache_data
def split_text_into_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    return splitter.split_text(text)