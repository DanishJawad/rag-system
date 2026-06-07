from typing import List, Sequence, Optional
import numpy as np
from langchain_ollama import OllamaEmbeddings
import os


class EmbeddingGenerator:

    def __init__(self, model_name: str = "nomic-embed-text"):
        self.model_name = model_name
        self.model = OllamaEmbeddings(model=model_name)

    def embed_texts(
        self,
        texts: Sequence[str],
        batch_size: int = 64,
    ) -> np.ndarray:

        all_embs = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            emb = self.model.embed_documents(list(batch))

            all_embs.extend(emb)

        return np.array(all_embs, dtype=np.float32)

    def embed_chunks(
        self,
        chunks: Sequence[dict],
        batch_size: int = 64,
    ) -> np.ndarray:

        texts = [c["text"] for c in chunks]

        return self.embed_texts(texts, batch_size)

    def save_embeddings(self, path: str, embeddings: np.ndarray):

        os.makedirs(os.path.dirname(path), exist_ok=True)

        np.save(path, embeddings)

    def load_embeddings(self, path: str):

        if not os.path.exists(path):
            return None

        return np.load(path)