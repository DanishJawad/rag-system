from typing import List, Dict


class ConversationMemory:
    """Lightweight conversation memory for multi-turn Q&A.

    Stores recent question/answer turns and provides:
    - History retrieval for UI display
    - Simple follow-up detection using keyword heuristics
    - Context injection for prompt enrichment
    """

    # Words that suggest the question refers back to prior context
    FOLLOWUP_KEYWORDS = {
        "it", "its", "they", "them", "their",
        "that", "this", "these", "those",
        "more", "also", "additionally",
        "and how", "what about", "how about",
        "explain more", "tell me more", "elaborate",
        "expand", "compare", "versus", "vs",
        "difference", "similar",
    }

    def __init__(self, max_turns: int = 10):
        """
        Args:
            max_turns: Maximum number of turns to keep in history.
        """
        self.max_turns = max_turns
        self.history: List[Dict] = []

    def add_turn(self, question: str, answer: str) -> None:
        """Store one question/answer pair."""
        self.history.append({"question": question, "answer": answer})
        if len(self.history) > self.max_turns:
            self.history = self.history[-self.max_turns:]

    def get_history(self) -> List[Dict]:
        """Return all stored turns."""
        return list(self.history)

    def get_context(self, num_turns: int = 2) -> str:
        """Format the most recent turns into a plain-text context string."""
        recent = self.history[-num_turns:] if self.history else []
        if not recent:
            return ""
        parts = []
        for turn in recent:
            parts.append(f"User: {turn['question']}")
            parts.append(f"Assistant: {turn['answer']}")
        return "\n".join(parts)

    def is_followup(self, question: str) -> bool:
        """Return True if the question likely refers to a previous turn.

        Uses keyword heuristics — fast, no LLM call needed.
        """
        if not self.history:
            return False

        q_lower = question.lower().strip()

        for keyword in self.FOLLOWUP_KEYWORDS:
            if q_lower.startswith(keyword + " ") or f" {keyword} " in q_lower:
                return True

        # Very short questions are almost always follow-ups
        if len(q_lower.split()) <= 3:
            return True

        return False

    def clear(self) -> None:
        """Clear all history."""
        self.history.clear()

    def size(self) -> int:
        """Number of turns stored."""
        return len(self.history)