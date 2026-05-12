import os
import pickle
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


@dataclass
class RetrievalResult:
    text: str
    source: str
    page: int
    score: float


class RAGChatbot:
    def __init__(self, index_dir: str = "vector_store", api_key: str | None = None):
        self.index_dir = Path(index_dir)
        self.index = faiss.read_index(str(self.index_dir / "docs.index"))
        with (self.index_dir / "metadata.pkl").open("rb") as f:
            self.metadata = pickle.load(f)

        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.chat_model = os.getenv("CHAT_MODEL", "gpt-4o-mini")

    def retrieve(self, question: str, top_k: int = 4) -> list[RetrievalResult]:
        query_vec = self.client.embeddings.create(model=self.embedding_model, input=question)
        q = np.array([query_vec.data[0].embedding], dtype="float32")
        faiss.normalize_L2(q)

        scores, indices = self.index.search(q, top_k)
        results: list[RetrievalResult] = []

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            item = self.metadata[idx]
            results.append(
                RetrievalResult(
                    text=item["text"],
                    source=item["source"],
                    page=item["page"],
                    score=float(score),
                )
            )

        return results

    def answer(self, question: str, top_k: int = 4, min_similarity: float = 0.35) -> dict:
        hits = self.retrieve(question, top_k=top_k)
        if not hits:
            return {
                "answer": "I couldn't find relevant content in your documents.",
                "citations": [],
                "chunks": [],
            }

        best_score = max(h.score for h in hits)
        if best_score < min_similarity:
            return {
                "answer": (
                    "I don't know based on the uploaded documents. "
                    "Try rephrasing your question or upload more relevant sources."
                ),
                "citations": [
                    {"source": h.source, "page": h.page, "score": round(h.score, 3)}
                    for h in hits
                ],
                "chunks": [
                    {
                        "source": h.source,
                        "page": h.page,
                        "score": round(h.score, 3),
                        "text": h.text,
                    }
                    for h in hits
                ],
            }

        context = "\n\n".join(
            [
                f"[Source: {h.source}, Page: {h.page}]\n{h.text}"
                for h in hits
            ]
        )

        system_prompt = (
            "You are a helpful assistant that answers ONLY from the provided context. "
            "If the answer is not in context, say you don't know based on the documents."
        )

        user_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer with concise explanation and cite sources."

        response = self.client.chat.completions.create(
            model=self.chat_model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        answer_text = response.choices[0].message.content
        citations = [
            {"source": h.source, "page": h.page, "score": round(h.score, 3)}
            for h in hits
        ]
        chunks = [
            {"source": h.source, "page": h.page, "score": round(h.score, 3), "text": h.text}
            for h in hits
        ]

        return {"answer": answer_text, "citations": citations, "chunks": chunks}


if __name__ == "__main__":
    bot = RAGChatbot(index_dir="vector_store")
    print("RAG chatbot ready. Type 'exit' to quit.\n")
    while True:
        q = input("You: ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        res = bot.answer(q)
        print(f"\nBot: {res['answer']}\n")
        if res["citations"]:
            print("Sources:")
            for c in res["citations"]:
                print(f"- {c['source']} (page {c['page']}, score={c['score']})")
        print()
