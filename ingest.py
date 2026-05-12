import argparse
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np
from openai import AuthenticationError, OpenAI, RateLimitError
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Chunk:
    text: str
    source: str
    page: int


def read_pdf(path: Path) -> list[Chunk]:
    reader = PdfReader(str(path))
    chunks: list[Chunk] = []
    for page_idx, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            chunks.append(Chunk(text=text, source=path.name, page=page_idx))
    return chunks


def read_text_file(path: Path) -> list[Chunk]:
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return []
    return [Chunk(text=text, source=path.name, page=1)]


def split_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    text = " ".join(text.split())
    if len(text) <= chunk_size:
        return [text]

    result: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        result.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return result


def load_documents(data_dir: Path) -> list[Chunk]:
    doc_chunks: list[Chunk] = []
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            raw_chunks = read_pdf(path)
        elif suffix in {".txt", ".md"}:
            raw_chunks = read_text_file(path)
        else:
            continue

        for raw in raw_chunks:
            for piece in split_text(raw.text):
                doc_chunks.append(Chunk(text=piece, source=raw.source, page=raw.page))

    return doc_chunks


def batched(items: list[str], batch_size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def embed_texts(texts: list[str], model: str, api_key: str | None = None) -> np.ndarray:
    client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
    vectors: list[list[float]] = []

    for batch in batched(texts, batch_size=100):
        try:
            response = client.embeddings.create(model=model, input=batch)
        except RateLimitError as e:
            raise RuntimeError(
                "OpenAI quota exceeded (insufficient_quota). Add billing/credits, then retry."
            ) from e
        except AuthenticationError as e:
            raise RuntimeError(
                "OpenAI authentication failed. Check OPENAI_API_KEY in .env and restart the app."
            ) from e
        vectors.extend([d.embedding for d in response.data])

    arr = np.array(vectors, dtype="float32")
    faiss.normalize_L2(arr)
    return arr


def save_index(vectors: np.ndarray, metadata: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(out_dir / "docs.index"))

    with (out_dir / "metadata.pkl").open("wb") as f:
        pickle.dump(metadata, f)


def run_ingestion(data_dir: Path, out_dir: Path, embedding_model: str, api_key: str | None = None) -> int:
    chunks = load_documents(data_dir)
    if not chunks:
        return 0

    texts = [c.text for c in chunks]
    vectors = embed_texts(texts, model=embedding_model, api_key=api_key)

    metadata = [
        {"text": c.text, "source": c.source, "page": c.page}
        for c in chunks
    ]
    save_index(vectors, metadata, out_dir)
    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into FAISS")
    parser.add_argument("--data-dir", default="data", help="Path containing documents")
    parser.add_argument("--out-dir", default="vector_store", help="Path to save FAISS index")
    parser.add_argument(
        "--embedding-model", default="text-embedding-3-small", help="OpenAI embedding model"
    )
    args = parser.parse_args()

    count = run_ingestion(Path(args.data_dir), Path(args.out_dir), args.embedding_model)
    if count == 0:
        print("No supported documents found (.pdf, .txt, .md).")
        return

    print(f"Ingestion complete. Indexed {count} chunks.")


if __name__ == "__main__":
    main()
