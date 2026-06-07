#!/usr/bin/env python3
"""
Minimal Streamlit UI for RAG Pipeline.
Simple, fast, and focused on core functionality.
"""

import streamlit as st
from src.rag_pipeline import RAGPipeline

# Configure page
st.set_page_config(page_title="RAG Q&A", page_icon="🔍", layout="centered")

# Title
st.title("🔍 RAG Research Papers Q&A")
st.markdown("Ask questions about the research papers in the knowledge base.")

# Cache pipeline (loads once)
@st.cache_resource
def load_pipeline():
    """Load RAG pipeline once and cache it."""
    return RAGPipeline()

# Load pipeline
try:
    pipeline = load_pipeline()
except Exception as e:
    st.error(f"❌ Failed to load pipeline: {e}")
    st.info("Make sure Ollama is running: `ollama serve`")
    st.stop()

# Show stats
with st.sidebar:
    st.header("📊 Stats")
    stats = pipeline.stats()
    st.metric("Total Chunks", stats["total_chunks"])
    st.metric("Vector Store", stats["vector_store"])
    st.metric("Retrieval K", stats["retrieval_k"])

# Question input
st.markdown("---")
question = st.text_input("❓ Your question:", placeholder="e.g., What is Dense Passage Retrieval?")

# Submit button
if st.button("Submit", type="primary", use_container_width=True):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("🔍 Searching and generating answer..."):
            try:
                result = pipeline.query(question)
                
                # Display answer
                st.subheader("📝 Answer")
                answer = result.get("answer", "").strip()
                
                # Check if hallucination guard was triggered
                guard_phrase = "I don't have enough information"
                if guard_phrase in answer:
                    st.info(f"**⚠️ Grounding Guard:** {answer}")
                else:
                    st.success(answer)
                
                # Display sources
                sources = result.get("sources", [])
                if sources:
                    st.subheader("📚 Sources")
                    for src in sources:
                        st.write(f"- **{src['document']}**")
                
            except Exception as e:
                st.error(f"Error: {e}")

# Footer
st.markdown("---")
st.caption("Built with Streamlit + Ollama + Chroma for RAG assessment")