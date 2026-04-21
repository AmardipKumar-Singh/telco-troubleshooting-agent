"""
One-time FAISS index builder for the Telco RAG retriever.
Sources indexed:
  - Track A/README.md
  - Track B/README.md
  - Track A/examples/traces.json  (reasoning examples)
  - Track A/data/Phase_1/train.json  (Q&A pairs)
Run: python rag/build_index.py
"""
import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_PATH  = "rag/index.faiss"
CHUNKS_PATH = "rag/chunks.json"
MODEL_NAME  = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE  = 400   # characters per chunk


def chunk_text(text: str, size: int = CHUNK_SIZE) -> list:
    words = text.split()
    chunks, current = [], []
    length = 0
    for w in words:
        current.append(w)
        length += len(w) + 1
        if length >= size:
            chunks.append(" ".join(current))
            current, length = [], 0
    if current:
        chunks.append(" ".join(current))
    return chunks


def load_sources() -> list:
    docs = []

    for readme in ["Track A/README.md", "Track B/README.md"]:
        if os.path.exists(readme):
            docs += chunk_text(open(readme).read())

    traces_path = "Track A/examples/traces.json"
    if os.path.exists(traces_path):
        traces = json.load(open(traces_path))
        for t in traces:
            q = t.get("question", "")
            a = t.get("answer", "")
            steps = " ".join(
                str(s.get("obs", "")) for s in t.get("steps", [])
            )
            docs.append(f"Q: {q}\nSteps: {steps}\nAnswer: {a}")

    train_path = "Track A/data/Phase_1/train.json"
    if os.path.exists(train_path):
        train = json.load(open(train_path))
        for item in train:
            q = item.get("question", "")
            a = item.get("answer", "")
            docs.append(f"Q: {q}\nA: {a}")

    return docs


def build():
    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)
    docs = load_sources()
    print(f"Indexing {len(docs)} chunks...")
    embeddings = model.encode(docs, show_progress_bar=True).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, INDEX_PATH)
    json.dump(docs, open(CHUNKS_PATH, "w"))
    print(f"Index saved to {INDEX_PATH} ({len(docs)} chunks).")


if __name__ == "__main__":
    build()
