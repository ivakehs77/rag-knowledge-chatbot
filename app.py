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
st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(255, 184, 108, 0.18), transparent 28%),
                radial-gradient(circle at top right, rgba(86, 204, 242, 0.16), transparent 24%),
                linear-gradient(180deg, #07111f 0%, #0d1728 100%);
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(7, 17, 31, 0.95) 0%, rgba(13, 23, 40, 0.92) 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }
        [data-testid="stSidebar"] > div:first-child {
            padding-top: 1.8rem;
        }
        .block-container {
            max-width: 1080px;
            padding-top: 2.2rem;
            padding-bottom: 2rem;
        }
        .hero-card {
            text-align: center;
            padding: 2.2rem 1.5rem 1.6rem;
            margin-bottom: 1.4rem;
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 24px;
            background: rgba(9, 18, 32, 0.72);
            box-shadow: 0 24px 60px rgba(0, 0, 0, 0.35);
            backdrop-filter: blur(12px);
        }
        .hero-title {
            margin: 0;
            font-size: 3rem;
            font-weight: 800;
            letter-spacing: -0.04em;
            color: #f8fafc;
        }
        .hero-subtitle {
            margin: 0.75rem 0 0;
            font-size: 1.05rem;
            color: #cbd5e1;
        }
        .hero-credit {
            margin-top: 1rem;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.18em;
            color: #f6ad55;
        }
        .content-card {
            padding: 1.4rem 1.2rem 1rem;
            border-radius: 22px;
            background: rgba(10, 19, 34, 0.72);
            border: 1px solid rgba(255, 255, 255, 0.07);
            box-shadow: 0 20px 44px rgba(0, 0, 0, 0.28);
        }
        .stButton > button {
            width: 100%;
            border-radius: 14px;
            border: 1px solid rgba(246, 173, 85, 0.5);
            background: linear-gradient(135deg, #f6ad55 0%, #dd6b20 100%);
            color: #08111f;
            font-weight: 700;
            box-shadow: 0 12px 28px rgba(221, 107, 32, 0.26);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 16px 34px rgba(221, 107, 32, 0.34);
            color: #08111f;
            border-color: rgba(246, 173, 85, 0.72);
        }
        .stTextInput > div > div > input {
            border-radius: 14px;
            background: rgba(15, 23, 42, 0.92);
            border: 1px solid rgba(148, 163, 184, 0.2);
        }
        .stSlider [data-baseweb="slider"] {
            padding-top: 0.4rem;
            padding-bottom: 0.4rem;
        }
        .stExpander {
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-radius: 16px;
            background: rgba(15, 23, 42, 0.45);
        }
        .stAlert {
            border-radius: 16px;
        }
    </style>
    <div class="hero-card">
        <h1 class="hero-title">RAG Knowledge Chatbot</h1>
        <p class="hero-subtitle">Upload your docs, build the index, and ask grounded questions with visible evidence.</p>
        <div class="hero-credit">Made by IVAKEsh</div>
    </div>
    <div class="content-card">
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("OpenAI API Key")
    api_key = st.text_input(
        "Your OpenAI API key",
        type="password",
        placeholder="sk-...",
        help="Get yours at platform.openai.com — never stored or logged.",
    )
    if not api_key:
        st.warning("Enter your API key to use the app.")

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
        if not api_key:
            st.error("Enter your OpenAI API key first.")
        else:
            try:
                with st.spinner("Embedding and indexing documents..."):
                    count = run_ingestion(
                        data_dir=DATA_DIR,
                        out_dir=INDEX_DIR,
                        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
                        api_key=api_key,
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
    if not api_key:
        st.error("Enter your OpenAI API key in the sidebar first.")
    else:
        try:
            bot = RAGChatbot(index_dir=str(INDEX_DIR), api_key=api_key)
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

st.markdown("</div>", unsafe_allow_html=True)
