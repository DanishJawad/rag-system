from typing import List, Dict, Optional
import logging

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)


class AnswerGenerator:
    """Wraps an LLM used to generate grounded answers from retrieved chunks.

    The class tries to use LangChain's ChatOllama chat interface but will
    gracefully surface useful errors if the model call fails.
    """

    def __init__(self, model_name: str = "qwen3:1.7b", temperature: float = 0.0):
        self.model_name = model_name
        self.temperature = temperature
        self.llm = ChatOllama(model=model_name, temperature=temperature)

    def _build_prompt(self, query: str, retrieved_chunks: List[dict]) -> List:
        context = "\n\n".join([f"[{c['document']}]\n{c['text']}" for c in retrieved_chunks])

        system_prompt = (
            "Answer ONLY using the provided documents.\n\n"
            "If the answer is NOT in the documents, reply exactly:"
            "\n\n'I don't have enough information in the provided documents to answer this question.'\n\n"
            "Do NOT guess, infer, or use outside knowledge."
        )

        human_text = f"Documents:\n{context}\n\nQuestion: {query}\n\nAnswer:"

        return [SystemMessage(content=system_prompt), HumanMessage(content=human_text)]

    def _call_llm(self, messages: List) -> str:
        # Prefer the high-level chat call; LangChain chat models support __call__
        try:
            resp = self.llm.invoke(messages)
            # Response may be a ChatResponse or similar; try to extract plain text
            if hasattr(resp, "content"):
                return resp.content
            # If the response is a LLMResult / ChatResult, extract generatively
            if hasattr(resp, "generations"):
                gens = resp.generations
                if gens and len(gens) > 0 and len(gens[0]) > 0:
                    return gens[0][0].text
            # Fallback to string conversion
            return str(resp)
        except TypeError:
            # Older/langchain versions expose .generate or .predict
            try:
                return self.llm.generate(messages).generations[0][0].text
            except Exception as exc:
                logger.exception("LLM generate failed: %s", exc)
                raise

    def generate_answer(self, query: str, retrieved_chunks: List[dict]) -> Dict[str, Optional[object]]:
        """Generate a grounded answer from retrieved chunks.

        Returns a dict with keys: `answer` (str) and `retrieved_chunks` (list).
        """
        messages = self._build_prompt(query, retrieved_chunks)

        try:
            answer_text = self._call_llm(messages).strip()
        except Exception as exc:
            logger.exception("LLM call failed: %s", exc)
            answer_text = ""

        return {"answer": answer_text, "retrieved_chunks": retrieved_chunks}