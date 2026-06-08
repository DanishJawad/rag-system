# RAG System — Information Retrieval & Language Models Q&A

A Retrieval-Augmented Generation (RAG) system that answers questions grounded strictly in a corpus of research papers on **Information Retrieval, Retrieval-Augmented Generation, and Transformer-based Language Models**. Built with LangGraph, Chroma, Ollama, and Streamlit.

---

## Stretch Goals Implemented 

This submission includes **two stretch goals** beyond core requirements, both fully integrated:

1. **Agentic Layer (LangGraph):** An LLM-based router decides whether to RETRIEVE (search documents) or DIRECT (answer from model knowledge). Implemented as a LangGraph StateGraph with explicit node functions and conditional edges. Defaults to RETRIEVE when uncertain—safer than silently hallucinating.

2. **Conversation Memory with Context Injection:** Multi-turn Q&A support with conversation context **injected into the LLM prompt** for true contextual reasoning. Enables anaphora resolution ("it", "that"), multi-turn synthesis, and follow-up understanding. Memory is stored in code (not just UI), with configurable context windows.

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

> **CLI alternative:** `python app.py`

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
User Question + Conversation Context
        ↓
┌─────────────────────────────────────────────┐
│  RAGPipeline.query()                        │
│  ├─ Get context from memory (last 3 turns)  │
│  └─ Pass to agent with context              │
└─────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────┐
│  Agent.query(question, conversation_ctx)    │
│  ├─ Router decides RETRIEVE or DIRECT       │
│  ├─ Retrieve path: vector search + RAG      │
│  └─ Direct path: answer from LLM knowledge  │
└─────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────┐
│  AnswerGenerator (with context in prompt)   │
│  ├─ Conversation History: {context}         │
│  ├─ Documents: {chunks}                     │
│  └─ Question: {question}                    │
└─────────────────────────────────────────────┘
        ↓
LLM (Qwen 1.7B) generates grounded answer
        ↓
┌─────────────────────────────────────────────┐
│  RAGPipeline.query()                        │
│  ├─ Add turn to memory                      │
│  └─ Return result + sources                 │
└─────────────────────────────────────────────┘
        ↓
Result: Answer + Sources + Strategy + Agent Thoughts
```

### Components

| File | Responsibility |
|------|---------------|
| `src/ingestion.py` | Load PDFs, split into chunks (RecursiveCharacterTextSplitter), remove references |
| `src/embeddings.py` | Generate embeddings (nomic-embed-text), cache to disk |
| `src/retrieval.py` | Chroma vector store, cosine similarity retrieval |
| `src/generation.py` | Prompt construction (with context), LLM call, hallucination guard |
| `src/agent.py` | LangGraph routing: RETRIEVE vs DIRECT paths (with context in both) |
| `src/memory.py` | Conversation history, follow-up detection, context formatting |
| `src/rag_pipeline.py` | Orchestrates all components, memory injection, dependency injection |
| `streamlit_app.py` | Streamlit web interface |
| `app.py` | CLI interface |

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

The safer choice is to over-retrieve.

**Why LangGraph over custom router?**  
LangGraph's StateGraph provides a declarative graph with explicit state passing between nodes. Each node is a pure function operating on `AgentState`. Benefits:
- Each step is independently testable
- The graph is easy to visualize and extend
- New nodes (e.g., re-ranking) can be added without modifying existing logic

**Direct path:**  
For questions clearly outside the corpus (e.g., "What is the capital of France?"), the agent answers directly from the LLM's training knowledge. The system prompt for the direct node explicitly says *"Do not claim to have used documents. Do not fabricate citations."*

---

## Conversation Memory with Context Injection (Stretch Goal)

**Class:** `ConversationMemory` (src/memory.py)  
**Storage:** In-memory list of dicts, max 10 turns  

**Capabilities:**
- Stores each Q&A turn as `{"question": ..., "answer": ...}`
- Displays conversation history in UI (last 5 turns visible)
- Detects follow-up questions using keyword heuristics
- **Generates formatted context** for LLM injection

**Follow-up Detection:**
Uses keyword matching (pronouns, referential words like "it", "they", "tell me more") and length heuristics (questions with ≤3 words are usually follow-ups). Fast, lightweight, no LLM call needed.

### Context Injection (The Key Innovation)

Memory context is now **injected into the LLM prompt** in three places:

1. **RAG Generation** (`generation.py`):
```python
human_text = f"""
Conversation History:
{conversation_context}  # ← Context here!

Documents:
{context}

Question:
{query}

Answer:
"""
```

2. **Direct Answers** (`agent.py`, direct_node):
```python
HumanMessage(content=f"""
Conversation History:
{state.get("conversation_context", "")}  # ← Context here!

Current Question:
{state["question"]}
""")
```

3. **Orchestration** (`rag_pipeline.py`):
```python
conversation_context = ""
if self.memory:
    # Get last 3 turns formatted
    conversation_context = self.memory.get_context(num_turns=3)

# Pass to agent (both RETRIEVE and DIRECT paths receive it)
result = self.agent.query(question, conversation_context=context)
```

### Why Context Injection Matters

**Without context injection (old approach):**
```
Q1: "What is DPR?"
A1: "Dense Passage Retrieval is..."

Q2: "Tell me more about it"
LLM sees: "Tell me more about it" (no context!)
Answer: Confused—doesn't know what "it" refers to
```

**With context injection (new approach):**
```
Q1: "What is DPR?"
A1: "Dense Passage Retrieval is..."

Q2: "Tell me more about it"
LLM sees: 
  Conversation History:
  Q: What is DPR?
  A: Dense Passage Retrieval is...
  
  Current Question: Tell me more about it
Answer: Understands "it" = DPR 
```

### Performance Trade-off

| Metric | Cost |
|--------|------|
| **Token usage increase** | ~25% (3 turns × 100 chars ≈ 100 tokens per query) |
| **Latency increase** | <5ms (negligible) |
| **Quality improvement** | Enormous (multi-turn reasoning now possible) |

**Trade-off assessment:** Worth it. The cost is small, the benefit is huge.

### Memory API

```python
pipeline = RAGPipeline(use_memory=True)

# Normal query (includes context)
result = pipeline.query("Tell me more")

# Query with metadata
result = pipeline.query_with_context("What about...")
# result["is_followup"] = True
# result["conversation_context"] = "Q: ...\nA: ..."

# Access history
history = pipeline.get_memory()  # List[{"question": ..., "answer": ...}]

# Clear history
pipeline.clear_memory()

# Check stats
stats = pipeline.stats()
# {..., "conversation_turns": 5, ...}
```

---

## Demo Questions & Expected Behavior

| # | Question | Expected Path | Behavior |
|---|----------|---|----------|
| 1 | *What is Dense Passage Retrieval and how does it work?* | RETRIEVE | Grounded answer from DPR paper, cited |
| 2 | *How do transformers enable retrieval-augmented generation?* | RETRIEVE | Synthesis across 2+ papers, cited |
| 3 | *What is the capital of France?* | DIRECT | Answered from model knowledge, no sources |
| 4 | *How does quantum computing work?* | RETRIEVE | Hallucination guard fires, refuses answer, no sources |

### Multi-turn Example

```
Q1: "What is REALM?"
A1: "REALM is Retrieval-Augmented Language Model Pre-Training..."

Q2: "How does it relate to RAG?"  
A2: "REALM and RAG are related. REALM applies RAG at pre-training time,
     while RAG applies it at inference time..."
     (understands "it" = REALM from context!)
```

---

## File Structure

```
rag-system/
├── src/
│   ├── __init__.py
│   ├── ingestion.py         # PDF loading, chunking
│   ├── embeddings.py        # nomic-embed-text generation + caching
│   ├── retrieval.py         # Chroma vector store
│   ├── generation.py        # LLM prompt + answer generation (with context)
│   ├── agent.py             # LangGraph agentic router (with context)
│   ├── memory.py            # Conversation history + follow-up detection
│   └── rag_pipeline.py      # Orchestration + memory injection
├── data/
│   ├── documents/           # 7 research papers (PDF)
│   ├── chroma/              # Persistent vector store
│   └── embeddings.npy       # Cached embeddings
├── streamlit_app.py         # Streamlit web interface
├── app.py                   # CLI interface
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
| **Memory** | In-memory + formatted context | Lightweight, fast, context-aware |
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

- **Token usage:** Context injection increases tokens by ~25% (small cost for major quality improvement)
- **Corpus is static:** Adding/removing papers requires reingest + re-embed. A production system would support differential updates.
- **Qwen 1.7B may struggle with complex reasoning:** Multi-hop reasoning, nuance, and edge cases may be beyond the model's capability. Larger models (7B+) or fine-tuning would help.
- **No chunk re-ranking:** Retrieved chunks are used as-is. A cross-encoder could re-rank for higher quality.
- **Memory does not persist across restarts:** History is lost when the app restarts. SQLite or Redis would enable persistence.
- **Hallucination guard uses exact phrase matching:** If the LLM paraphrases the refusal, it won't be detected. Fuzzy matching or a learned classifier would be more robust.

---

## Potential Production Improvements

- Persistent memory (SQLite or Redis) across restarts
- Add re-ranking (cross-encoder) of retrieved chunks
- Implement hybrid search (BM25 keyword + semantic embeddings)
- Evaluation harness (RAGAS metrics or custom Q&A pairs)
- Streaming responses
- Fuzzy matching for hallucination detection
- Support for dynamic corpus updates (add/remove papers without reingest)
- Multiple conversation branches (for A/B testing different responses)

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
5. (Follow-up) "Tell me more about that"

Watch the agent routing, retrieval, context injection, and generation happen in real-time.

