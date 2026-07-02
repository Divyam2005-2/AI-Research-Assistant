import streamlit as st
import chromadb
@st.cache_resource
def get_Client():
    return chromadb.PersistentClient()
client = get_Client()
path = "vectorstore"
def create_collection(collection_name= "Research_Papers"):
    collection = client.get_or_create_collection(name = collection_name)
    return collection

def add_chunks_to_db(collection, chunks, embeddings):

    ids= [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(
        ids= ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=[
        {
            "chunk": i
        }
        for i in range(len(chunks))
    ]
    )
    return collection

def search_chunks(collection, query_embedding, n_results=6):
    results= collection.query(
        query_embeddings=[query_embedding],
        n_results= n_results
    )
    return results