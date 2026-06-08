from typing import TypedDict, Literal, List, Dict, Any

from langgraph.graph import StateGraph, END

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
)

Route = Literal["retrieve", "direct"]


class AgentState(TypedDict):
    question: str
    route: Route
    answer: str
    retrieved_chunks: List[Dict[str, Any]]
    sources: List[Dict[str, Any]]
    thoughts: List[str]


class RAGAgent:
    """Agentic RAG system using LangGraph.

    The router decides which path to take:
    - retrieve: use vector DB + RAG pipeline
    - direct: answer from model's own knowledge

    If the router is uncertain, it defaults to retrieve.
    """

    def __init__(
        self,
        retrieval,
        embedding_gen,
        generator,
        llm,
        k: int = 3,
    ):
        self.retrieval = retrieval
        self.embedding_gen = embedding_gen
        self.generator = generator
        self.llm = llm
        self.k = k

        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(AgentState)

        graph.add_node("router", self.router_node)
        graph.add_node("retrieve", self.retrieve_node)
        graph.add_node("generate", self.generate_node)
        graph.add_node("direct", self.direct_node)

        graph.set_entry_point("router")

        graph.add_conditional_edges(
            "router",
            self.route_decision,
            {
                "retrieve": "retrieve",
                "direct": "direct",
            },
        )

        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", END)
        graph.add_edge("direct", END)

        return graph.compile()

    def router_node(self, state: AgentState):
        """Decide whether retrieval is needed.

        Defaults to retrieve when uncertain — safer than answering
        from model knowledge when the topic may be in the documents.
        """
        system_prompt = """
You are a routing agent for a Retrieval-Augmented Generation (RAG) system.

The document corpus contains research papers about:
- Dense Passage Retrieval (DPR)
- Retrieval-Augmented Generation (RAG)
- REALM
- Transformers
- Embeddings
- Semantic Search
- Question Answering Systems

Task:
Determine whether the user question requires information
from the document corpus.

Rules:
1. If the question is related to the topics above,
   answer: retrieve

2. If the question is clearly unrelated and can be answered
   using general knowledge,
   answer: direct

3. If uncertain, ALWAYS choose retrieve.

Respond with exactly one word:

retrieve

or

direct
"""

        response = self.llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=state["question"]),
            ]
        )

        raw_decision = response.content.strip().lower()

        # Default to "retrieve" when uncertain — aligns with rule 3 above
        decision: Route = (
            "direct"
            if raw_decision.startswith("direct")
            else "retrieve"
        )

        state["route"] = decision

        state["thoughts"].append(
            f"Router selected: {decision.upper()}"
        )

        return state

    def route_decision(self, state: AgentState):
        return state["route"]

    def retrieve_node(self, state: AgentState):
        """Retrieve relevant chunks from the vector store."""
        q_embedding = self.embedding_gen.embed_texts(
            [state["question"]]
        )

        chunks = self.retrieval.retrieve(
            q_embedding,
            k=self.k,
        )

        state["retrieved_chunks"] = chunks

        state["thoughts"].append(
            f"Retrieved {len(chunks)} chunk(s)"
        )

        return state

    def generate_node(self, state: AgentState):
        """Generate a grounded answer from retrieved chunks."""
        result = self.generator.generate_answer(
            state["question"],
            state["retrieved_chunks"],
        )

        answer = result["answer"]
        state["answer"] = answer

        refusal_phrase = "I don't have enough information"

        if refusal_phrase.lower() in answer.lower():
            state["sources"] = []
            state["thoughts"].append("Hallucination guard triggered")
            return state

        seen_docs: set = set()
        sources = []

        for chunk in state["retrieved_chunks"]:
            doc = chunk.get("document")
            if not doc or doc in seen_docs:
                continue
            seen_docs.add(doc)
            sources.append(
                {
                    "document": doc,
                    "preview": chunk.get("text", "")[:200],
                }
            )

        state["sources"] = sources

        state["thoughts"].append(
            f"Generated grounded answer using {len(sources)} source(s)"
        )

        return state

    def direct_node(self, state: AgentState):
        """Answer directly from model knowledge (no retrieval)."""
        response = self.llm.invoke(
            [
                SystemMessage(
                    content=(
                        "Answer the user's question directly. "
                        "Do not claim to have used documents. "
                        "Do not fabricate citations."
                    )
                ),
                HumanMessage(content=state["question"]),
            ]
        )

        state["answer"] = response.content
        state["sources"] = []
        state["thoughts"].append("Answered directly from model knowledge")

        return state

    def query(self, question: str) -> Dict:
        """Execute the agent workflow for a question."""
        initial_state: AgentState = {
            "question": question,
            "route": "retrieve",
            "answer": "",
            "retrieved_chunks": [],
            "sources": [],
            "thoughts": [],
        }

        result = self.graph.invoke(initial_state)

        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "strategy": result["route"].upper(),
            "agent_thoughts": result["thoughts"],
        }