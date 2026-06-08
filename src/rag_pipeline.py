from pathlib import Path
import logging
import os
from typing import List, Dict, Optional

from dotenv import load_dotenv
from langchain_ollama import ChatOllama

from .ingestion import DocumentIngestion
from .embeddings import EmbeddingGenerator
from .retrieval import ChromaRetrieval
from .generation import AnswerGenerator
from .agent import RAGAgent
from .memory import ConversationMemory

load_dotenv()

logger = logging.getLogger(__name__)


class RAGPipeline:
    """End-to-end RAG orchestration with agentic routing and memory.

    Responsibilities:
    - Ingest PDFs and chunk
    - Generate embeddings (with caching)
    - Build/load Chroma vector store
    - Route queries through agent (retrieve vs direct)
    - Maintain conversation memory for follow-ups
    - Answer queries with citations
    """

    def __init__(
        self,
        data_dir: str = None,
        index_dir: str = None,
        k: int = None,
        chunk_size: int = None,
        chunk_overlap: int = None,
        use_agent: bool = True,
        use_memory: bool = True,
    ):
        # Load configuration from environment variables with sensible defaults
        self.data_dir = Path(data_dir or os.getenv("DATA_DIR", "data/documents"))
        self.index_dir = Path(index_dir or os.getenv("INDEX_DIR", "data/chroma"))
        self.k = k or int(os.getenv("RETRIEVAL_K", "3"))
        self.chunk_size = chunk_size or int(os.getenv("CHUNK_SIZE", "500"))
        self.chunk_overlap = chunk_overlap or int(os.getenv("CHUNK_OVERLAP", "100"))
        self.use_agent = use_agent
        self.use_memory = use_memory

        self.embeddings_path = self.index_dir / "embeddings.npy"
        self.chroma_path = str(self.index_dir)

        # Components initialised in _prepare()
        self.chunks: List[Dict] = []
        self.embedding_gen: Optional[EmbeddingGenerator] = None
        self.retrieval: Optional[ChromaRetrieval] = None
        self.generator: Optional[AnswerGenerator] = None
        self.agent: Optional[RAGAgent] = None
        self.memory: Optional[ConversationMemory] = None

        self._prepare()

    def _prepare(self) -> None:
        # 1) Ingest and chunk documents
        ingestion = DocumentIngestion(
            str(self.data_dir),
            self.chunk_size,
            self.chunk_overlap,
        )
        documents = ingestion.load_documents()
        self.chunks = ingestion.chunk_documents(documents)
        logger.info("Loaded %d documents -> %d chunks", len(documents), len(self.chunks))

        # 2) Embeddings (load from cache if available)
        self.embedding_gen = EmbeddingGenerator()
        self.index_dir.mkdir(parents=True, exist_ok=True)

        embeddings = None
        try:
            if self.embeddings_path.exists():
                embeddings = self.embedding_gen.load_embeddings(str(self.embeddings_path))
        except Exception:
            logger.debug("No cached embeddings found, computing fresh ones")

        if embeddings is None:
            texts = [c["text"] for c in self.chunks]
            embeddings = self.embedding_gen.embed_texts(texts)
            try:
                self.embedding_gen.save_embeddings(str(self.embeddings_path), embeddings)
            except Exception:
                logger.warning("Could not save embeddings to disk")

        # 3) Vector store
        self.retrieval = ChromaRetrieval(
            embeddings,
            self.chunks,
            persist_directory=self.chroma_path,
            collection_name="rag_papers",
        )

        # 4) Generator
        self.generator = AnswerGenerator()

        # 5) Agent (LLM-based routing)
        if self.use_agent:
            try:
                router_llm = ChatOllama(
                    model=os.getenv("OLLAMA_MODEL", "qwen3:1.7b"),
                    temperature=0.0,
                )
                self.agent = RAGAgent(
                    retrieval=self.retrieval,
                    embedding_gen=self.embedding_gen,
                    generator=self.generator,
                    llm=router_llm,
                    k=self.k,
                )
                logger.info("Agent enabled (chunks=%d)", len(self.chunks))
            except Exception as exc:
                logger.warning("Agent init failed, falling back to direct mode: %s", exc)
                self.use_agent = False

        # 6) Conversation memory
        if self.use_memory:
            self.memory = ConversationMemory(max_turns=10)
            logger.info("Conversation memory enabled")

    def _query_direct(self, question: str) -> Dict:
        """Fallback RAG query without the agent (used when use_agent=False)."""
        q_embedding = self.embedding_gen.embed_texts([question])
        chunks = self.retrieval.retrieve(q_embedding, k=self.k)
        result = self.generator.generate_answer(question, chunks)

        guard_phrase = "I don't have enough information"
        if guard_phrase.lower() in result["answer"].lower():
            return {
                "answer": result["answer"],
                "sources": [],
                "strategy": "RETRIEVE",
                "agent_thoughts": ["Hallucination guard triggered"],
            }

        sources = []
        seen: set = set()
        for chunk in chunks:
            doc = chunk.get("document")
            if doc and doc not in seen:
                seen.add(doc)
                sources.append({"document": doc, "preview": chunk.get("text", "")[:200]})

        return {
            "answer": result["answer"],
            "sources": sources,
            "strategy": "RETRIEVE",
            "agent_thoughts": [f"Retrieved {len(chunks)} chunks"],
        }

    def query(self, question: str) -> Dict:
        """Run a full query through the pipeline.

        Routes through the agent when available, otherwise uses direct RAG.
        Stores the turn in memory if memory is enabled.
        """
        conversation_context = ""

        if self.memory:
            conversation_context = self.memory.get_context(
                num_turns=3
            )

        if self.use_agent and self.agent:
            result = self.agent.query(
                question,
                conversation_context=conversation_context,
            )
        else:
            result = self._query_direct(question)

        if self.memory:
            self.memory.add_turn(
                question=question,
                answer=result["answer"],
            )

        return result

    def query_with_context(self, question: str) -> Dict:
        """Query that also surfaces follow-up detection metadata."""
        result = self.query(question)

        if self.memory:
            result["is_followup"] = self.memory.is_followup(question)
            result["conversation_context"] = self.memory.get_context(num_turns=2)
        else:
            result["is_followup"] = False
            result["conversation_context"] = ""

        return result

    def clear_memory(self) -> None:
        """Clear conversation history."""
        if self.memory:
            self.memory.clear()
            logger.info("Conversation memory cleared")

    def get_memory(self) -> Optional[List[Dict]]:
        """Return conversation history as a list of dicts."""
        if self.memory:
            return self.memory.get_history()
        return None

    def stats(self) -> Dict:
        return {
            "total_chunks": len(self.chunks),
            "embedding_model": self.embedding_gen.model_name if self.embedding_gen else None,
            "retrieval_k": self.k,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "vector_store": "Chroma",
            "agent_enabled": self.use_agent,
            "memory_enabled": self.use_memory,
            "conversation_turns": self.memory.size() if self.memory else 0,
        }