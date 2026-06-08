# RAG System — Information Retrieval & Language Models Q&A

A Retrieval-Augmented Generation (RAG) system that answers questions grounded strictly in a corpus of research papers on **Information Retrieval, Retrieval-Augmented Generation, and Transformer-based Language Models**. Built with LangGraph, Chroma, Ollama, and Streamlit.

---

## Stretch Goals Implemented ✅

This submission includes **two stretch goals** beyond core requirements:

1. **Agentic Layer (LangGraph):** An LLM-based router decides whether to RETRIEVE (search documents) or DIRECT (answer from model knowledge). Implemented as a LangGraph StateGraph with explicit node functions and conditional edges. Defaults to RETRIEVE when uncertain—safer than silently hallucinating.

2. **Conversation Memory:** Multi-turn Q&A support with keyword-based follow-up detection. Stores up to 10 turns. Follow-ups detected via heuristics (pronouns, referential words, short questions). Memory enables UI history display and metadata tracking (not injected into LLM prompt for simplicity).

Both features are working, tested, and demonstrated in the Loom video.

---

## Quick Start

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.ai) installed and running
- ~5 GB disk space

### Setup

```bash
# 1. Clone
git clone https://github.com/DanishJawad/rag-system.git
cd rag-system

# 2. Install dependencies
uv sync

# 3. Activate virtualenv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows

# 4. Pull models (Ollama must be running)
ollama pull qwen3:1.7b
ollama pull nomic-embed-text

# 5. Start the UI
streamlit run streamlit_app.py
```

Open **http://localhost:8501**

---

## Environment Variables

Copy `.env.example` to `.env` and adjust if needed. All values have sensible defaults.

```bash
OLLAMA_MODEL=qwen3:1.7b
OLLAMA_BASE_URL=http://localhost:11434
EMBEDDING_MODEL=nomic-embed-text
DATA_DIR=data/documents
INDEX_DIR=data/chroma
CHUNK_SIZE=500
CHUNK_OVERLAP=100
RETRIEVAL_K=3
```

---

## Architecture

```
User Question
      │
      ▼
┌─────────────────────────────────────────────────┐
│  LangGraph Agent  (src/agent.py)                │
│                                                 │
│  router_node  (decides: retrieve or direct?)    │
│    │                                            │
│    ├─ YES  ──► retrieve_node                    │
│    │           ├─ embed query                   │
│    │           ├─ search Chroma (k=3)           │
│    │           ├─ generate_node                 │
│    │           │   ├─ call LLM with chunks      │
│    │           │   ├─ check refusal phrase      │
│    │           │   └─ extract sources           │
│    │           └─ END                           │
│    │                                            │
│    └─ NO  ──► direct_node                       │
│               ├─ answer from model knowledge    │
│               ├─ no retrieval                   │
│               └─ END                            │
└─────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────┐
│  Conversation Memory  (src/memory.py)           │
│  ├─ Store Q&A turn                              │
│  ├─ Detect follow-ups                           │
│  └─ UI history display                          │
└─────────────────────────────────────────────────┘
      │
      ▼
Return: Answer + Sources (if grounded) + Strategy + Reasoning
```

### Components

| File | Responsibility |
|------|---------------|
| `src/ingestion.py` | Load PDFs, split into chunks (RecursiveCharacterTextSplitter), remove references |
| `src/embeddings.py` | Generate embeddings (nomic-embed-text), cache to disk |
| `src/retrieval.py` | Chroma vector store, cosine similarity retrieval |
| `src/generation.py` | Prompt construction, LLM call, hallucination guard |
| `src/agent.py` | LangGraph routing: RETRIEVE vs DIRECT paths |
| `src/memory.py` | Conversation history, follow-up detection |
| `src/rag_pipeline.py` | Orchestrates all components, dependency injection |
| `streamlit_app.py` | Streamlit web interface |

---

## Chunking Strategy

**Method:** `RecursiveCharacterTextSplitter`  
**Chunk size:** 500 characters  
**Overlap:** 100 characters  
**Separators (priority order):** `\n\n`, `\n`, `. `, ` `, ``

**Why recursive splitting?**  
Fixed-size splitting breaks mid-sentence and mid-paragraph, losing semantic coherence. Recursive splitting tries splitting at natural boundaries (paragraph → sentence → word → character), only going smaller when necessary. This preserves meaning within each chunk, improving both retrieval accuracy and generation quality.

**Why 500 chars with 100 overlap?**  
- 500 chars ≈ 3–5 sentences: enough context for a meaningful answer without padding the LLM with noise
- 100-char overlap ensures sentences spanning a chunk boundary are captured by at least one chunk
- Trade-off: Smaller chunks = more precise retrieval; larger chunks = more context per chunk. 500 is a reasonable middle ground for academic papers

**Reference removal:**  
The ingestion pipeline strips the References section from each PDF. References are dense citation lists that match keyword queries but contain no substantive answers—they're noise for semantic retrieval.

---

## Embedding Model

**Model:** `nomic-embed-text` (via Ollama)  
**Dimensions:** 768  

**Why this model?**
- Runs locally — no API cost, no data leaving your machine
- 768-dim vectors provide good semantic resolution for academic text
- Faster than BGE-large, more semantically aware than all-MiniLM-L6-v2
- Specifically trained on semantic similarity tasks

**Caching:** Embeddings are saved to `data/embeddings.npy` on first run. Subsequent startups skip re-embedding (saves ~30–60 seconds on 7 papers).

---

## Vector Store

**Store:** Chroma  
**Distance metric:** Cosine similarity  
**Persistence:** `data/chroma/`

**Why Chroma?**  
Persistent (saves to disk), easy to integrate, well-suited to corpora of this size (~850 chunks). Cosine similarity is the standard metric for semantic search—it measures directional alignment between embedding vectors, not raw magnitude.

**Trade-off vs. alternatives:**
- **FAISS:** Faster at scale (millions of vectors), but in-memory only (no persistence without extra work). Overkill for ~850 chunks.
- **Pinecone:** Fully managed and scalable, but requires an API key and costs money.
- **pgvector:** Good for production systems at scale, but heavier setup.

For ~850 chunks and local inference, Chroma is the right choice.

---

## Hallucination Prevention

Three layers of defense:

1. **Prompt-level:** System message explicitly instructs the LLM to refuse if the answer is not in the provided documents, and never use outside knowledge.

2. **Exact refusal phrase:** The system message specifies the exact string the LLM should return when it doesn't know: *"I don't have enough information in the provided documents to answer this question."* This makes detection deterministic—no regex, no fuzzy matching.

3. **Code-level enforcement:** Both `agent.py` (generate_node) and `rag_pipeline.py` (_query_direct) check for this phrase after generation. If detected, `sources` is set to `[]` so the UI never shows documents as citations for a refused answer.

---

## Agentic Layer (Stretch Goal)

**Framework:** LangGraph  
**Graph structure:**

```
START → router_node → [conditional_edges]
                        ├─ "retrieve" → retrieve_node → generate_node → END
                        └─ "direct"   → direct_node → END
```

**Router logic:**  
The router is an LLM call (same model, temperature=0) that reads the question and decides which path to take. Its system prompt lists the exact topics in the corpus. If uncertain, it **defaults to `retrieve`**—this is intentional.

**Why default to retrieve?**
- False positive on retrieval: costs one embedding + Chroma lookup (~200ms) + LLM call with context (negligible latency)
- False negative (skipping retrieval when needed): silently answers from training data without corpus grounding, violating RAG principles

The safer trade-off is to over-retrieve.

**Why LangGraph over custom router?**  
LangGraph's StateGraph provides a declarative graph with explicit state passing between nodes. Each node is a pure function operating on `AgentState`. Benefits:
- Each step is independently testable
- The graph is easy to visualize and extend
- New nodes (e.g., re-ranking) can be added without modifying existing logic

**Direct path:**  
For questions clearly outside the corpus (e.g., "What is the capital of France?"), the agent answers directly from the LLM's training knowledge. The system prompt for the direct node explicitly says *"Do not claim to have used documents. Do not fabricate citations."*

---

## Conversation Memory (Stretch Goal)

**Class:** `ConversationMemory` (src/memory.py)  
**Storage:** In-memory list of dicts, max 10 turns  

**Capabilities:**
- Stores each Q&A turn as `{"question": ..., "answer": ...}`
- Displays conversation history in UI (last 5 turns visible)
- Detects follow-up questions using keyword heuristics
- Provides metadata for follow-up detection

**Follow-up Detection:**
Uses keyword matching (pronouns, referential words like "it", "they", "tell me more") and length heuristics (questions with ≤3 words are usually follow-ups). Fast, lightweight, no LLM call needed.

**Honest scope & limitations:**  
Memory is **NOT** injected into the LLM prompt. Each query is processed independently by the generation model. This is a deliberate trade-off:

| Aspect | Benefit | Cost |
|--------|---------|------|
| **No injection** | Reduces token usage; avoids noise from unrelated follow-ups | True multi-turn contextual reasoning limited to UI display |
| **Conditional injection** | Would enable smarter follow-up handling | Adds complexity, more tokens |

In a production system, we'd inject context conditionally:
```python
if memory.is_followup(question):
    context = memory.get_context(num_turns=2)
    # Include context in LLM prompt
```

This implementation prioritizes simplicity and token efficiency.

---

## Demo Questions & Expected Behavior

| # | Question | Expected Path | Behavior |
|---|----------|---|----------|
| 1 | *What is Dense Passage Retrieval and how does it work?* | RETRIEVE | Grounded answer from DPR paper, cited |
| 2 | *How do transformers enable retrieval-augmented generation?* | RETRIEVE | Synthesis across 2+ papers, cited |
| 3 | *What is the capital of France?* | DIRECT | Answered from model knowledge, no sources |
| 4 | *How does quantum computing work?* | RETRIEVE | Hallucination guard fires, refuses answer, no sources |

---

## File Structure

```
rag-system/
├── src/
│   ├── __init__.py
│   ├── ingestion.py         # PDF loading, chunking
│   ├── embeddings.py        # nomic-embed-text generation + caching
│   ├── retrieval.py         # Chroma vector store
│   ├── generation.py        # LLM prompt + answer generation
│   ├── agent.py             # LangGraph agentic router
│   ├── memory.py            # Conversation history + follow-up detection
│   └── rag_pipeline.py      # Orchestration + dependency injection
├── data/
│   ├── documents/           # 7 research papers (PDF)
│   ├── chroma/              # Persistent vector store
│   └── embeddings.npy       # Cached embeddings
├── streamlit_app.py         # Streamlit web interface
├── pyproject.toml           # Dependencies (uv)
├── .env.example             # Environment variable template
└── README.md                # This file
```

---

## Technical Stack

| Component | Choice | Reason |
|-----------|--------|--------|
| **LLM** | Qwen 1.7B via Ollama | Fast on CPU, free, local, sufficient for RAG |
| **Embeddings** | nomic-embed-text via Ollama | Local, 768-dim, good quality, no API needed |
| **Vector Store** | Chroma | Persistent, easy integration, right scale (~850 chunks) |
| **Chunking** | LangChain RecursiveCharacterTextSplitter | Semantic boundary preservation, industry standard |
| **Agent framework** | LangGraph | Declarative graph, testable nodes, extensible architecture |
| **Interface** | Streamlit | Rapid UI development, live reloading, caching support |
| **PDF parsing** | pypdf | Lightweight, no external service dependency |
| **Package manager** | uv | Fast, reproducible installs |

---

## Corpus

This system is trained on **7 research papers** covering Information Retrieval, Retrieval-Augmented Generation, and Transformer Language Models:

| # | Paper | Year | Domain | Relevance |
|----|-------|------|--------|-----------|
| 1 | Dense Passage Retrieval for Open-Domain QA | 2020 | IR/RAG | Core retrieval technique; dense embeddings |
| 2 | Retrieval-Augmented Generation for Knowledge-Intensive NLP | 2020 | RAG | Foundational RAG paper; combines retrieval + seq2seq |
| 3 | REALM: Retrieval-Augmented Language Model Pre-Training | 2020 | Pre-training | Scalable RAG at pre-training time |
| 4 | ColBERT: Efficient and Effective Passage Search | 2020 | IR | Efficient dense retrieval via late interaction |
| 5 | Fusion-in-Decoder | 2021 | QA | Multi-document fusion for open-domain QA |
| 6 | Attention Is All You Need | 2017 | Transformers | Transformer architecture; enables retrieval-aware models |
| 7 | Language Models are Few-Shot Learners (GPT-3) | 2020 | LLMs | In-context learning; demonstrates scaling laws |

**Why this corpus?**  
These papers form a coherent narrative: from foundational IR techniques (DPR, ColBERT) through RAG systems (RAG, REALM, Fusion-in-Decoder) to the Transformer architecture enabling modern retrieval-aware language models (Attention Is All You Need) and large-scale LLMs (GPT-3). The corpus is large enough (~850 chunks) that naive keyword search would fail—semantic retrieval is meaningfully tested.

---

## Limitations & Trade-offs

- **Conversation context not in generation:** Memory is for UI display + metadata only. Each query is independent. Could inject conditionally (if `is_followup()` returns True) for smarter multi-turn reasoning.
- **Corpus is static:** Adding/removing papers requires reingest + re-embed. A production system would support differential updates.
- **Qwen 1.7B may struggle with complex reasoning:** Multi-hop reasoning, nuance, and edge cases may be beyond the model's capability. Larger models (7B+) or fine-tuning would help.
- **No chunk re-ranking:** Retrieved chunks are used as-is. A cross-encoder could re-rank for higher quality.
- **Memory does not persist across restarts:** History is lost when the app restarts. SQLite or Redis would enable persistence.
- **Hallucination guard uses exact phrase matching:** If the LLM paraphrases the refusal, it won't be detected. Fuzzy matching or a learned classifier would be more robust.

## Potential Improvements

- Inject conversation context into generation **conditionally** (if `is_followup()` returns True)
- Add re-ranking (cross-encoder) of retrieved chunks
- Implement hybrid search (BM25 keyword + semantic embeddings)
- Evaluation harness (RAGAS metrics or custom Q&A pairs)
- Streaming responses
- Persistent memory (SQLite or Redis)
- Fuzzy matching for hallucination detection
- Support for dynamic corpus updates (add/remove papers without reingest)

---

## Running the Demo

### Start the system
```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Start Streamlit
source .venv/bin/activate
streamlit run streamlit_app.py
```

### Ask questions
Visit http://localhost:8501 and try:
1. "What is Dense Passage Retrieval?"
2. "How do transformers enable RAG?"
3. "What's the capital of France?"
4. "How does quantum computing work?"

Watch the agent routing, retrieval, and generation happen in real-time.

---

## Development Notes

### Logging
Set `LOGGING_LEVEL` environment variable (default: `INFO`):
```bash
LOGGING_LEVEL=DEBUG streamlit run streamlit_app.py
```

### Disabling Agent or Memory
```python
pipeline = RAGPipeline(use_agent=False, use_memory=False)
```

### Profile Queries
Check how long each step takes:
```python
import time
start = time.time()
result = pipeline.query("your question")
print(f"Total: {time.time() - start:.2f}s")
```

---
