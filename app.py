import streamlit as st
from backend.pdf_loader import extract_text_from_pdf
from backend.chunking import split_text_into_chunks
from backend.embedding import create_embedding
from backend.vector_db import create_collection, add_chunks_to_db, search_chunks
from backend.llm import generate_answer
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="📚",
    layout="wide"
)

# Sidebar
with st.sidebar:
    st.title("AI research Assistant")
    st.markdown("---")

    uploaded_file = st.file_uploader('Upload a research paper', type=["pdf"])

# Main page

st.title("AI Research Assistant")

st.header("Welcome")

st.write("The application will hlep you analyze Research paper with the help of AL")

st.markdown("---")

if uploaded_file is not None:
    st.success(f"File uploaded: {uploaded_file.name}")

    pdf_text = extract_text_from_pdf(uploaded_file)

    chunks = split_text_into_chunks(pdf_text)

    # st.success(f"Total chunks: {len(chunks)}")

    # for i,chunk in enumerate(chunks[:10]):
        # st.subheader(f"Chunk {i+1}: {len(chunk)} characters")
        # st.write(chunk)

    embeddings = create_embedding(chunks)
    # st.success(f"Created {len(embeddings)} embeddings")
    # st.write("Embedding shape:", embeddings.shape)

    collection= create_collection()

    st.success("Collection created successfully")

    add_chunks_to_db(collection, chunks,embeddings)

    st.success("Chunks add successfully to db")

    st.markdown("---")

    user_question=st.text_input("Ask a question about the research paper!")

    if user_question:
        query_embedding= create_embedding(user_question)
        result_chunks= search_chunks(collection, query_embedding, n_results=3)
        # st.write(result_chunks)
        with st.spinner("🤖 Gemini is analyzing the research paper..."):
            answer= generate_answer(
                user_question,
                result_chunks["documents"][0]
        )
        st.subheader("🤖 AI Answer")
        st.write(answer)
        
        st.subheader("Top most relevent chunks")
        for chunk in result_chunks["documents"][0]:
            st.write(chunk)
            st.markdown("---")
        
        
    # else:
    #     st.warning("Enter a question!")


    # st.subheader("Extracted text")

    # st.text_area(
    #     "PDF content", pdf_text[:1000], height=300 
    # )
else:
    st.warning("unable to upload file")

