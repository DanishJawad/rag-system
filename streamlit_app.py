#!/usr/bin/env python3
import streamlit as st
from src.rag_pipeline import RAGPipeline

st.set_page_config(page_title="RAG Q&A", page_icon="🔍")
st.title("RAG Research Paper Q&A")

@st.cache_resource
def load_pipeline():
    return RAGPipeline(use_agent=True, use_memory=True)

pipeline = load_pipeline()
stats = pipeline.stats()

# Sidebar
with st.sidebar:
    st.subheader("System Status")
    st.write(f"Chunks indexed: **{stats['total_chunks']}**")
    st.write(f"Agent: {'enabled' if stats['agent_enabled'] else 'disabled'}")
    st.write(f"Memory turns: **{stats['conversation_turns']}**")
    if st.button("Clear Memory"):
        pipeline.clear_memory()
        st.rerun()

# Question input
st.markdown("---")
question = st.text_input("Ask a question:", placeholder="What is Dense Passage Retrieval?")

if question:
    with st.spinner("Thinking..."):
        result = pipeline.query_with_context(question)

    strategy = result.get("strategy", "")
    if strategy == "RETRIEVE":
        st.caption(f"Strategy: RETRIEVE — answered from documents")
    else:
        st.caption(f"Strategy: DIRECT — answered from model knowledge")

    st.write(result["answer"])

    if result.get("sources"):
        st.subheader("Sources")
        for src in result["sources"]:
            st.write(f"- {src['document']}")

    if result.get("agent_thoughts"):
        with st.expander("Agent Reasoning"):
            for thought in result["agent_thoughts"]:
                st.write(thought)

# Conversation history
history = pipeline.get_memory() or []
if history:
    st.markdown("---")
    st.subheader("Conversation History")
    for turn in reversed(history[-5:]):
        with st.expander(f"Q: {turn['question'][:70]}"):
            st.write(f"**Q:** {turn['question']}")
            st.write(f"**A:** {turn['answer'][:400]}")