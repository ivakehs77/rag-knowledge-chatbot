# RAG Knowledge Chatbot (Python + OpenAI + FAISS)

This project builds a document-grounded chatbot using Retrieval-Augmented Generation (RAG).

## What it does

- Ingests your documents (`.pdf`, `.txt`, `.md`)
- Splits text into chunks
- Converts chunks into embeddings (`text-embedding-3-small`)
- Stores vectors in FAISS for semantic search
- Retrieves relevant chunks for each user question
- Uses an OpenAI chat model to answer with document context
- Shows citations (`source + page`)
- Includes similarity threshold gating for safe "I don't know" responses
- Displays retrieved chunk previews for answer auditability

## Project Structure

```text
rag-knowledge-chatbot
├── data
│   └── uploads
├── vector_store
├── ingest.py
├── chatbot.py
├── app.py
├── requirements.txt
└── README.md
```

## Setup

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure environment:

```bash
cp .env.example .env
# then edit .env and set OPENAI_API_KEY
```

## Run (Web App)

```bash
streamlit run app.py
```

Then:
1. Upload files from sidebar
2. Click **Save Uploaded Files**
3. Click **Run Ingestion**
4. Ask questions in the main panel

## Cost Note

- This project uses paid OpenAI API calls (embeddings + chat completions).
- Keep uploaded docs small during development.
- Set billing limits/alerts in your OpenAI dashboard before heavy testing.

## Example Questions

- "What is the difference between computer science and software engineering in this document?"
- "Summarize chapter 1 in 5 bullet points."
- "What topics are covered on page 7?"
- "Is there any mention of binary search? If yes, where?"

## Run (CLI flow)

1. Put files in `data/` (or change path)
2. Build index:

```bash
python ingest.py --data-dir data --out-dir vector_store
```

3. Chat in terminal:

```bash
python chatbot.py
```

## Architecture (MVP)

```text
User Question
    -> Embed question
    -> Search FAISS for top-k chunks
    -> Send chunks + question to LLM
    -> Return grounded answer + citations
```

## Resume Value

This project demonstrates:
- LLM app design (RAG)
- Embeddings and semantic retrieval
- Vector DB usage (FAISS)
- Prompt grounding and source citation
- End-to-end Python AI engineering

## Next upgrades

- Add chunk reranking before generation
- Add metadata filters (by file/topic/date)
- Add conversation memory
- Add evaluation metrics (faithfulness/retrieval precision)
- Deploy with Docker + cloud hosting
