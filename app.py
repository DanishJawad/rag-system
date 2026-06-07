#!/usr/bin/env python3
"""
Simple CLI interface for RAG pipeline.

Usage:
    python app.py
"""

import sys
from src.rag_pipeline import RAGPipeline


def main():
    """Interactive Q&A loop."""
    print("\n" + "="*60)
    print("🚀 RAG SYSTEM - RESEARCH PAPER Q&A")
    print("="*60)
    print("\nInitializing pipeline...")
    
    try:
        pipeline = RAGPipeline(
            data_dir="data/documents",
            index_dir="data/chroma",
            k=3,
            chunk_size=500,
            chunk_overlap=100,
        )
    except Exception as e:
        print(f"\n❌ Failed to initialize pipeline: {e}")
        print("\nMake sure:")
        print("  1. PDFs are in data/documents/")
        print("  2. Ollama is running: ollama serve")
        print("  3. Mistral model is available: ollama pull mistral")
        sys.exit(1)
    
    print("\n" + "-"*60)
    print(f"✅ Pipeline ready!")
    print(f"   📊 {pipeline.stats()['total_chunks']} chunks indexed")
    print(f"   📚 Vector store: {pipeline.stats()['vector_store']}")
    print("-"*60)
    
    print("\n📝 Ask questions about the research papers (type 'exit' to quit):\n")
    
    while True:
        try:
            question = input("❓ Question: ").strip()
            
            if question.lower() in ["exit", "quit", "q"]:
                print("\n👋 Goodbye!")
                break
            
            if not question:
                print("⚠️  Please enter a question.\n")
                continue
            
            print("\n🔍 Processing...\n")
            
            # Query pipeline
            result = pipeline.query(question)
            
            # Display answer
            print("📝 ANSWER:")
            print("-" * 60)
            print(result.get("answer", "No answer generated"))
            print("-" * 60)
            
            # Display sources
            sources = result.get("sources", [])
            if sources:
                print("\n📚 SOURCES:")
                for src in sources:
                    print(f"   • {src['document']}")
            
            print("\n" + "="*60 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    main()