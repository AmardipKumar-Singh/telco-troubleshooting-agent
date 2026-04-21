"""
Query-time RAG retriever backed by a pre-built FAISS index.
"""
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

INDEX_PATH  = "rag/index.faiss"
CHUNKS_PATH = "rag/chunks.json"
MODEL_NAME  = "sentence-transformers/all-MiniLM-L6-v2"


class TelcoRetriever:
    def __init__(self, index_path=INDEX_PATH, chunks_path=CHUNKS_PATH, model_name=MODEL_NAME):
        self.model  = SentenceTransformer(model_name)
        self.index  = faiss.read_index(index_path)
        self.chunks = json.load(open(chunks_path))

    def query(self, question: str, top_k: int = 3) -> str:
        vec = self.model.encode([question]).astype("float32")
        _, indices = self.index.search(vec, top_k)
        results = [self.chunks[i] for i in indices[0] if 0 <= i < len(self.chunks)]
        return "\n---\n".join(results)
