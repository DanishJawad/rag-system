from typing import List, Optional
import logging
import os

import chromadb
import numpy as np

logger = logging.getLogger(__name__)


class ChromaRetrieval:
    """
    Chroma-backed retrieval using cosine similarity.
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        chunks: List[dict],
        persist_directory: Optional[str] = None,
        collection_name: str = "rag_papers",
    ):
        self.chunks = chunks
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        if self.persist_directory:
            os.makedirs(self.persist_directory, exist_ok=True)

        self.client = (
            chromadb.PersistentClient(path=self.persist_directory)
            if self.persist_directory
            else chromadb.EphemeralClient()
        )

        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        self._add_chunks(embeddings)

    def _add_chunks(self, embeddings: np.ndarray) -> None:
        if not self.chunks:
            return

        embeddings = embeddings.astype("float32")

        ids = [chunk["id"] for chunk in self.chunks]

        documents = [
            chunk["text"]
            for chunk in self.chunks
        ]

        metadatas = [
            {
                "document": chunk["document"],
                "chunk_index": chunk["chunk_index"],
            }
            for chunk in self.chunks
        ]

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
        )

        logger.info("Indexed %d chunks", len(documents))

    def retrieve(
        self,
        query_embedding: np.ndarray,
        k: int = 3,
    ) -> List[dict]:

        if query_embedding.ndim == 2:
            query_vector = query_embedding[0].astype("float32").tolist()
        else:
            query_vector = query_embedding.astype("float32").tolist()

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for doc, meta, dist in zip(
            documents,
            metadatas,
            distances,
        ):
            chunks.append(
                {
                    "document": meta.get("document"),
                    "chunk_index": meta.get("chunk_index"),
                    "text": doc,
                    "distance": float(dist),
                }
            )

        chunks.sort(key=lambda x: x["distance"])

        return chunks