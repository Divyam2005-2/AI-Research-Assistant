import streamlit as st
from sentence_transformers import SentenceTransformer
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")
model  = load_embedding_model() 

# embedds sentences into numerical value for more relatable meanings

def create_embedding(chunks):
    embedding = model.encode(chunks)
    return embedding