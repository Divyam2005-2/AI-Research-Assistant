import chromadb
client = chromadb.Client()
def create_collection(collection_name= "Research_Papers"):
    collection = client.get_or_create_collection(name = collection_name)
    return collection

def add_chunks_to_db(collection, chunks, embeddings):

    ids= [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(
        ids= ids,
        documents=chunks,
        embeddings=embeddings
    )
    return collection

def search_chunks(collection, query_embedding, n_results=3):
    results= collection.query(
        query_embeddings=[query_embedding],
        n_results= n_results
    )
    return results