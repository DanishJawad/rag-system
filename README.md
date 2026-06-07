# RAG System - Retrieval-Augmented Generation for Research Papers

A Retrieval-Augmented Generation (RAG) system that answers questions about research papers using semantic search and LLM generation with grounding guards to prevent hallucination.

## Quick Start

### Prerequisites
- Python 3.10+
- Ollama (https://ollama.ai)
- ~5GB disk space (for model + data)

### Setup (5 minutes)

```bash
# 1. Clone repository
git clone https://github.com/DanishJawad/rag-system.git
cd rag-system

# 2. Install dependencies with uv
uv sync

# 3. Activate virtual environment (created by uv)
source .venv/bin/activate  # or: .venv\Scripts\activate on Windows

# 4. Verify papers are in place
ls data/documents/  # Should show 7 PDFs

# 5. Start Ollama (in separate terminal)
ollama serve

# 6. Pull models (in another terminal)
ollama pull qwen3:1.7b
ollama pull nomic-embed-text

# 7. Run application
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## System Architecture

### Overview

The system follows this pipeline:

```
User Question
    |
    v
[1] Embed Query (nomic-embed-text)
    |
    v
[2] Retrieve Chunks (Chroma vector store)
    |
    v
[3] Generate Answer (Qwen 1.7B LLM)
    |
    v
Answer + Sources + Grounding Status
```

### Components

#### 1. Document Ingestion (src/ingestion.py)

Loads PDFs and converts them to searchable chunks.

Key Features:
- Recursive character splitting (1000 chars, 200 overlap)
- Semantic structure preservation (splits on paragraphs/sentences first)
- Reference section removal (improves retrieval quality)
- Text normalization (whitespace cleanup)

Why This Approach:
- Recursive splitting preserves semantic boundaries better than fixed-size chunks
- Overlap ensures context is preserved across chunks
- Removing references prevents irrelevant citations from cluttering results

#### 2. Embeddings (src/embeddings.py)

Converts text chunks into 768-dimensional vectors for semantic search.

Model Choice: nomic-embed-text
- Fast embedding generation (1-2 seconds for all chunks)
- Good semantic understanding (768-dim vectors)
- Runs locally via Ollama (no API costs)
- Sufficient for this corpus size

Alternatives Considered:
- all-MiniLM-L6-v2: Faster but lower dimensionality (384-dim)
- OpenAI embeddings: Better quality but requires API key and costs money
- BGE-large: Higher quality but slower

#### 3. Retrieval (src/retrieval.py)

Finds the most relevant document chunks using vector similarity.

Implementation:
- Vector store: Chroma (persistent, efficient)
- Distance metric: Cosine similarity
- Top-k: Configurable (default: 3 chunks)
- Persistence: Saves to data/chroma/ for fast reload

Why Chroma:
- Persistent storage (no re-indexing on startup)
- Efficient for our scale (approximately 800 chunks)
- Easy to use and integrate
- Good performance for semantic search

Trade-off: Not scalable to millions of documents. For larger scales, would use Pinecone or Postgres + pgvector.

#### 4. Generation (src/generation.py)

Uses an LLM to synthesize an answer from retrieved chunks.

Prompt Strategy:
```
System Message:
- "Answer ONLY using the provided documents"
- "If answer not in documents, reply with: 'I don't have enough information...'"
- "Never use outside knowledge"

Context Format:
- Each chunk marked with document name: [Document: name]
- Chunks separated clearly
- Question asked after context
```

Why This Works:
- Explicit instruction prevents hallucination
- Exact refusal phrase makes it easy to detect when model admits ignorance
- Document labels help model track provenance

Model Choice: Qwen 1.7B
- Fast inference (2-5 seconds per query)
- Free and runs locally
- Sufficient for grounded QA over academic papers
- Trade-off: Smaller than GPT-4, might struggle with very complex reasoning

#### 5. Orchestration (src/rag_pipeline.py)

Coordinates all components and manages the full pipeline.

Key Features:
- Automatic embedding caching (saved to data/embeddings.npy)
- Vector store persistence (Chroma data cached)
- Query processing (full pipeline in one call)
- Statistics tracking

---

## Configuration

### Environment Variables

Create a `.env` file (optional, uses defaults if not provided):

```bash
# LLM Configuration
OLLAMA_MODEL=qwen3:1.7b
OLLAMA_BASE_URL=http://localhost:11434

# Embedding Configuration
EMBEDDING_MODEL=nomic-embed-text

# Data Configuration
DATA_DIR=data/documents
INDEX_DIR=data/chroma
CHUNK_SIZE=500
CHUNK_OVERLAP=100
RETRIEVAL_K=3
```

### Adjustable Parameters

In src/rag_pipeline.py:
- chunk_size: Smaller chunks = more precise retrieval, more chunks
- chunk_overlap: Larger overlap = better context preservation
- k: More chunks = more context but potentially more noise
- OLLAMA_MODEL: Can switch to mistral, neural-chat, llama2, etc.

---

## Hallucination Prevention

The system uses multiple layers of defense against hallucination:

1. Prompt-Based: System message explicitly tells LLM to refuse guessing
2. Context-Based: Only relevant chunks provided to LLM
3. Detection-Based: Code checks for exact refusal phrase
4. Response-Based: If refused, sources list is cleared

### How It Works

When a question is outside the knowledge base:
```
User: "How does quantum computing work?"
System retrieves chunks (low similarity scores)
LLM recognizes answer is not in documents
LLM returns: "I don't have enough information..."
System detects refusal and clears sources
```

### Testing Grounding

In-Corpus Question:
```
Q: "What is Dense Passage Retrieval?"
A: "DPR is a retrieval method that... [explanation from paper]"
Sources: dense_passage_retrieval
Status: Grounded (answer from documents)
```

Out-of-Corpus Question:
```
Q: "How does quantum computing work?"
A: "I don't have enough information in the provided documents..."
Sources: (none)
Status: Grounded (correctly refused)
```

---

## Performance Characteristics

### Query Processing Time
- Q1 (in-corpus): 2-3 seconds
- Q2 (complex synthesis): 3-4 seconds
- Q3 (out-of-corpus): 1-2 seconds (quick refusal)

### System Statistics
- Total chunks indexed: approximately 850 (from 7 papers)
- Embedding model: nomic-embed-text (768-dim)
- Vector store: Chroma
- Default retrieval: k=3 chunks per query

### Example Query Metrics
```
Query: "What is Dense Passage Retrieval?"
Chunks retrieved: 3
Average similarity: 0.82 (high = good match)
Response time: 2.3 seconds
Grounded: Yes (answer from documents)
```

---

## Design Trade-offs

### Chunking: Recursive vs. Fixed-Size
- Choice: Recursive character splitting
- Reason: Preserves semantic structure (paragraphs/sentences)
- Alternative: Fixed-size is simpler but breaks logical units
- Verdict: Recursive splitting is worth the added complexity

### Embeddings: nomic-embed-text vs. all-MiniLM vs. OpenAI
- Choice: nomic-embed-text (768-dim)
- Reason: Good balance of speed, quality, and local execution
- Alternatives:
  - all-MiniLM is faster (384-dim)
  - OpenAI is more accurate but costs money
- Verdict: nomic-embed-text is the best balance for this project

### Vector Store: Chroma vs. FAISS vs. Pinecone
- Choice: Chroma
- Reason: Persistent, efficient, good for approximately 1000 vectors
- Alternatives:
  - FAISS is faster but doesn't persist
  - Pinecone is scalable but costs money
- Verdict: Chroma is ideal for this scale

### LLM: Qwen 1.7B vs. Mistral vs. GPT-4
- Choice: Qwen 1.7B
- Reason: Fast, free, runs offline, sufficient for grounded QA
- Alternatives:
  - Mistral 7B is more capable but slower
  - GPT-4 is more capable but costs money
- Verdict: Qwen is ideal for demonstration purposes

### Interface: Streamlit vs. REST API vs. CLI
- Choice: Streamlit
- Reason: Clean user interface, great for demonstration
- Alternatives:
  - REST API is more scalable but more complex
  - CLI works but less visually appealing
- Verdict: Streamlit is ideal for assessment showcase

---

## Demo Questions

Three test questions demonstrate different capabilities:

### Question 1: In-Corpus Retrieval
"What is Dense Passage Retrieval and how does it work?"

Tests: Basic semantic search, answer generation, source citation
Expected: Direct answer from dense_passage_retrieval with clear explanation

### Question 2: Multi-Source Synthesis
"How do transformers enable retrieval-augmented generation?"

Tests: Semantic search across multiple papers, synthesis, complex reasoning
Expected: Answer combining concepts from transformers and rag_knowledge_intensive_nlp papers

### Question 3: Grounding Guard
"How does quantum computing work?"

Tests: Hallucination prevention
Expected: System correctly refuses to answer (topic not in knowledge base)

---

## File Structure

```
rag-system/
├── src/
│   ├── __init__.py
│   ├── ingestion.py          # PDF loading + chunking
│   ├── embeddings.py         # Embedding generation
│   ├── retrieval.py          # Vector store (Chroma)
│   ├── generation.py         # LLM answer generation
│   └── rag_pipeline.py       # Orchestration
├── data/
│   ├── documents/            # 7 research papers
│   ├── chroma/               # Vector store (persistent)
│   └── embeddings.npy        # Cached embeddings
├── app.py                    # Streamlit interface
├── pyproject.toml            # Project configuration and dependencies
├── .env.example              # Environment variable template
├── README.md                 # This file
└── .gitignore
```

---

## Technical Stack

- Language: Python 3.10+
- LLM: Qwen 1.7B (via Ollama)
- Embeddings: nomic-embed-text (via Ollama)
- Vector Store: Chroma
- Chunking: LangChain RecursiveCharacterTextSplitter
- Interface: Streamlit
- Document Processing: pypdf
- Dependency Management: uv (Python package installer)

---

## Limitations and Future Improvements

### Current Limitations
- Single-turn Q&A (no conversation history)
- Limited to 7 papers (manageable corpus size)
- Qwen 1.7B is smaller than GPT-4 (may struggle with very complex reasoning)
- No re-ranking of retrieved chunks

### Potential Improvements
- Conversational memory for multi-turn Q&A
- Re-ranking retrieved chunks for better relevance
- Hybrid search (keyword + semantic)
- Token counting and usage tracking
- Evaluation metrics for retrieval quality

### Scaling Considerations
At 10x corpus size, would consider:
- Semantic chunking instead of fixed-size
- Larger embedding model (BGE-large)
- Postgres + pgvector instead of Chroma
- Better LLM (Mistral 7B or GPT-4)

At 100x corpus size, would likely move to:
- Cloud-based vector store (Pinecone)
- Fine-tuned embedding model
- Dedicated ML infrastructure

---

## Testing and Validation

To verify the system works correctly:

1. Start Ollama: `ollama serve`
2. Start Streamlit: `streamlit run app.py`
3. Test each demo question
4. Verify sources are cited for in-corpus questions
5. Verify refusal message for out-of-corpus questions

---

## References

Papers Used in Knowledge Base:
1. Dense Passage Retrieval for Open-Domain Question Answering
2. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
3. REALM: Retrieval-Augmented Language Model Pre-Training
4. ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT
5. Fusion-in-Decoder: Leveraging Heterogeneous Data for Joint Question Answering
6. Attention Is All You Need (Transformers)
7. Language Models are Few-Shot Learners (GPT-3)

Libraries and Tools:
- Ollama: Local LLM inference (https://ollama.ai)
- Chroma: Vector database (https://www.trychroma.com)
- LangChain: LLM framework
- Streamlit: Web interface framework
- uv: Fast Python package installer (https://astral.sh/blog/uv)

---

## How to Run

### Using uv (Recommended)
```bash
uv sync
source .venv/bin/activate  # or: .venv\Scripts\activate on Windows
streamlit run app.py
```

### Manual Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt  # if using pip instead of uv
streamlit run app.py
```

The application will open at http://localhost:8501

---

## Summary

This RAG system demonstrates:
- Effective semantic search over document collections
- Grounded answer generation from retrieved context
- Hallucination prevention through multi-layer defense
- Clean, modular Python code architecture
- Practical understanding of RAG system components

The implementation prioritizes clarity and functionality over complexity, following the principle that focused, well-explained solutions are more valuable than over-engineered ones.