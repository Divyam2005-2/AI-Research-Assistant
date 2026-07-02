import streamlit as st
from pathlib import Path
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

def load_css():
    with open("assets/styles.css") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )
load_css()
# Sidebar
with st.sidebar:
    st.title("📄 Upload Research Paper")
    st.caption("Upload a PDF and ask questions about its content.")
    st.divider()
    uploaded_file = st.file_uploader('Upload a research paper', type=["pdf"])
    st.divider()
    st.subheader("⚙️ AI Stack")
    st.markdown("""
- 🤖 Gemini 2.5 Flash
- 🧠 Sentence Transformers
- 📚 ChromaDB
- ⚡ Streamlit
""")

# Main page
st.markdown("""
<div class="hero">
    <h1>📚 AI Research Assistant</h1>
    <p>
        Analyze research papers with Retrieval-Augmented Generation (RAG),
        semantic search, and Gemini AI. Upload a PDF, ask questions,
        and receive context-aware answers instantly.
    </p>
</div>
""", unsafe_allow_html=True)


if uploaded_file is not None:
    pdf_text = extract_text_from_pdf(uploaded_file)
    chunks = split_text_into_chunks(pdf_text)
    embeddings = create_embedding(chunks)
    collection= create_collection()
    add_chunks_to_db(collection, chunks,embeddings)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📄 File", uploaded_file.name)
    with col2:
        st.metric("🧩 Chunks", len(chunks))
    with col3:
        st.metric("🤖 Model", "Gemini 2.5 Flash")
    st.divider()

# User question
    st.subheader("💬 Ask Your Question")
    user_question=st.text_input("",placeholder="Example: What is the main contribution of this paper?")

    if user_question:
        query_embedding= create_embedding(user_question)
        result_chunks= search_chunks(collection, query_embedding, n_results=3)
        with st.spinner("🤖 Gemini is analyzing the research paper..."):
            answer= generate_answer(
                user_question,
                result_chunks["documents"][0]
        )
        with st.container():
            st.subheader("🤖 AI Answer")
            st.markdown(
                f"""<div class="answer-box">
                    {answer}
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.divider()
        
        st.subheader("📖 Retrieved Sources")
        for i, chunk in enumerate(result_chunks["documents"][0]):
            with st.expander(f"📄 Source {i+1}"):
                st.write(chunk)
        st.markdown("---")
else:
    st.info("👈 Upload a research paper from the sidebar to get started.")
    st.divider()

st.caption("Built with ❤️ using Streamlit • Gemini • ChromaDB • Sentence Transformers")
st.divider()
