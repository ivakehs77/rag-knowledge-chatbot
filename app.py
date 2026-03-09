import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from chatbot import RAGChatbot
from ingest import run_ingestion

load_dotenv()

DATA_DIR = Path("data/uploads")
INDEX_DIR = Path("vector_store")


st.set_page_config(page_title="RAG Knowledge Chatbot", page_icon="📚", layout="wide")
st.title("📚 RAG Knowledge Chatbot")
st.write("Upload your docs, build index, then ask grounded questions.")

if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY not found. Add it to your .env file.")

with st.sidebar:
    st.header("1) Upload Documents")
    uploaded_files = st.file_uploader(
        "Supported: PDF, TXT, MD",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )

    if st.button("Save Uploaded Files"):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        for f in uploaded_files or []:
            (DATA_DIR / f.name).write_bytes(f.getbuffer())
        st.success(f"Saved {len(uploaded_files or [])} file(s) to {DATA_DIR}.")

    st.header("2) Build / Refresh Index")
    if st.button("Run Ingestion"):
        try:
            with st.spinner("Embedding and indexing documents..."):
                count = run_ingestion(
                    data_dir=DATA_DIR,
                    out_dir=INDEX_DIR,
                    embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
                )
        except RuntimeError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Ingestion failed: {e}")
        else:
            if count == 0:
                st.warning("No supported files found in data/uploads.")
            else:
                st.success(f"Index ready with {count} chunks.")

st.header("3) Ask Questions")
question = st.text_input("Ask something from your uploaded documents")
top_k = st.slider("Top-K retrieved chunks", min_value=2, max_value=8, value=4)
min_similarity = st.slider("Minimum similarity threshold", min_value=0.0, max_value=1.0, value=0.35, step=0.05)

if st.button("Ask") and question.strip():
    try:
        bot = RAGChatbot(index_dir=str(INDEX_DIR))
    except Exception as e:
        st.error(f"Index not available yet. Run ingestion first. Error: {e}")
    else:
        try:
            with st.spinner("Retrieving context and generating answer..."):
                result = bot.answer(question, top_k=top_k, min_similarity=min_similarity)
        except Exception as e:
            st.error(f"Question failed: {e}")
        else:
            st.subheader("Answer")
            st.write(result["answer"])

            st.subheader("Citations")
            if result["citations"]:
                for c in result["citations"]:
                    st.write(f"- {c['source']} (page {c['page']}, similarity {c['score']})")
            else:
                st.write("No citations available.")

            if result.get("chunks"):
                st.subheader("Retrieved Chunk Previews")
                for i, chunk in enumerate(result["chunks"], start=1):
                    title = f"{i}. {chunk['source']} (page {chunk['page']}, similarity {chunk['score']})"
                    with st.expander(title):
                        st.write(chunk["text"])
