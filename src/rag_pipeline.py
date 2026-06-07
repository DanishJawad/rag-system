from pathlib import Path
import json
import logging
from typing import List, Dict, Optional

from dotenv import load_dotenv

from .ingestion import DocumentIngestion
from .embeddings import EmbeddingGenerator
from .retrieval import ChromaRetrieval
from .generation import AnswerGenerator

load_dotenv()

logger = logging.getLogger(__name__)


class RAGPipeline:
    """End-to-end RAG orchestration.

    Responsibilities:
    - Ingest PDFs and chunk
    - Generate embeddings (with batching)
    - Build/Load Chroma collection
    - Answer queries by retrieving + generating
    """

    def __init__(
        self,
        data_dir: str = "data/documents",
        index_dir: str = "data/chroma",
        k: int = 3,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ):
        self.data_dir = Path(data_dir)
        self.index_dir = Path(index_dir)
        self.k = k
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.embeddings_path = self.index_dir / "embeddings.npy"
        self.chroma_path = str(self.index_dir)

        # Components (initialized later)
        self.chunks: List[Dict] = []
        self.embedding_gen: Optional[EmbeddingGenerator] = None
        self.retrieval: Optional[ChromaRetrieval] = None
        self.generator: Optional[AnswerGenerator] = None

        # Build or load pipeline
        self._prepare()

    def _prepare(self) -> None:
        # 1) Ingest / chunk
        ingestion = DocumentIngestion(str(self.data_dir), self.chunk_size, self.chunk_overlap)
        documents = ingestion.load_documents()
        self.chunks = ingestion.chunk_documents(documents)
        logger.info("Loaded %d documents -> %d chunks", len(documents), len(self.chunks))

        # 2) Embeddings
        self.embedding_gen = EmbeddingGenerator()

        # Try to load saved embeddings & index; otherwise compute
        self.index_dir.mkdir(parents=True, exist_ok=True)

        embeddings = None
        try:
            if (self.embeddings_path).exists():
                embeddings = self.embedding_gen.load_embeddings(str(self.embeddings_path))
        except Exception:
            logger.debug("No saved embeddings found, will compute fresh ones")

        if embeddings is None:
            texts = [c["text"] for c in self.chunks]
            embeddings = self.embedding_gen.embed_texts(texts)
            # persist
            try:
                self.embedding_gen.save_embeddings(str(self.embeddings_path), embeddings)
            except Exception:
                logger.warning("Failed to save embeddings to disk")

        # 3) Index / persistent vector store
        try:
            self.retrieval = ChromaRetrieval(
                embeddings,
                self.chunks,
                persist_directory=self.chroma_path,
                collection_name="rag_papers",
            )
        except Exception as exc:
            logger.exception("Failed to build Chroma collection: %s", exc)
            raise

        # 4) Generator
        self.generator = AnswerGenerator()

        logger.info("RAG pipeline ready (chunks=%d)", len(self.chunks))

    def query(self, question: str) -> Dict:
        """Run a full query: embed question, retrieve chunks, generate answer, add citations."""
        if not self.embedding_gen or not self.retrieval or not self.generator:
            raise RuntimeError("Pipeline not initialized")

        # Embed question
        q_emb = self.embedding_gen.embed_texts([question])

        # Retrieve
        retrieved = self.retrieval.retrieve(q_emb, k=self.k)

        # Generate
        result = self.generator.generate_answer(question, retrieved)

        guard_response = (
            "I don't have enough information in the provided documents to answer this question."
        )

        if result["answer"].strip() == guard_response:
            result["sources"] = []
            return result

        seen = set()
        sources = []

        for c in retrieved:
            doc = c.get("document")

            if doc in seen:
                continue

            seen.add(doc)

            sources.append(
                {
                    "document": doc,
                    "preview": c.get("text", "")[:200],
                }
            )

        result["sources"] = sources

        return result

    def stats(self) -> Dict:
        return {
            "total_chunks": len(self.chunks),
            "embedding_model": (
                self.embedding_gen.model_name
                if self.embedding_gen
                else None
            ),
            "retrieval_k": self.k,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "vector_store": "Chroma",
        }
