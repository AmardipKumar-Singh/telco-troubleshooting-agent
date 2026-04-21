"""
Diagnostic memory module.
Tracks per-question conversation state and cross-question tool call cache.
"""
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DiagnosticStep:
    step: int
    thought: str
    tool: str
    args: Any
    obs: str


@dataclass
class QuestionMemory:
    question_id: str
    question: str
    track: str
    steps: list = field(default_factory=list)
    final_answer: str = None
    resolved: bool = False

    def add_step(self, thought: str, tool: str, args: Any, obs: str):
        self.steps.append(DiagnosticStep(
            step=len(self.steps),
            thought=thought,
            tool=tool,
            args=args,
            obs=obs
        ))

    def to_dict_list(self) -> list:
        return [
            {"step": s.step, "thought": s.thought,
             "tool": s.tool, "args": s.args, "obs": s.obs}
            for s in self.steps
        ]

    def to_trace(self) -> dict:
        """Export in Phase 2 trace upload format."""
        return {
            "id": self.question_id,
            "track": self.track,
            "question": self.question,
            "steps": self.to_dict_list(),
            "answer": self.final_answer
        }


class ToolCallCache:
    """Memoize identical (tool_name, args) calls within a session.
    Saves API quota under the 1,000 calls/day Phase 1 limit."""

    def __init__(self):
        self._cache = {}

    def get(self, tool: str, args: dict):
        key = (tool, str(sorted(args.items()) if isinstance(args, dict) else args))
        return self._cache.get(key)

    def set(self, tool: str, args: dict, result):
        key = (tool, str(sorted(args.items()) if isinstance(args, dict) else args))
        self._cache[key] = result

    def clear(self):
        self._cache.clear()
