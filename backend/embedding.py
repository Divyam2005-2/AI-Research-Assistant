from sentence_transformers import SentenceTransformer

model  = SentenceTransformer("all-MiniLM-L6-v2")

# embedds sentences into numerical value for more relatable meanings

def create_embedding(chunks):
    embedding = model.encode(chunks)
    return embedding