"""
Core ReAct Agent for the Telco Troubleshooting Agentic Challenge.
Works for both Track A (wireless, IoU scoring) and Track B (IP, exact match).
"""
import re
import json
import time
import asyncio
from typing import Callable
from .prompt_builder import build_prompt
from .memory import QuestionMemory, ToolCallCache

# ── Output parser ─────────────────────────────────────────────────────────────

_THOUGHT_RE  = re.compile(r"Thought:\s*(.+?)(?=\nAction:)", re.DOTALL)
_ACTION_RE   = re.compile(r"Action:\s*(.+?)(?=\nAction Input:|\Z)", re.DOTALL)
_INPUT_RE    = re.compile(r"Action Input:\s*(.+)", re.DOTALL)

def parse_react_output(text: str) -> tuple[str, str, dict | str]:
    thought = (_THOUGHT_RE.search(text) or type("", (), {"group": lambda s, i: ""})()).group(1).strip()
    action  = (_ACTION_RE.search(text) or type("", (), {"group": lambda s, i: ""})()).group(1).strip()
    raw_inp = (_INPUT_RE.search(text) or type("", (), {"group": lambda s, i: "{}"})()).group(1).strip()

    if action == "FINAL_ANSWER":
        return thought, "FINAL_ANSWER", raw_inp

    try:
        args = json.loads(raw_inp)
    except json.JSONDecodeError:
        args = {"raw": raw_inp}

    return thought, action, args


# ── Answer formatter ──────────────────────────────────────────────────────────

def format_track_a_answer(raw: str, confidence_threshold: float = 0.0) -> str:
    """
    Normalise Track A answers to 'C3|C7|C11' ascending-order format.
    Raw may be: 'C3, C7, C11' / 'C3|C7' / 'C3 C7' / already correct.
    """
    tokens = re.findall(r"C\d+", raw.upper())
    unique_sorted = sorted(set(tokens), key=lambda x: int(x[1:]))
    return "|".join(unique_sorted) if unique_sorted else raw.strip()


# ── Main agent ────────────────────────────────────────────────────────────────

class TelcoReActAgent:
    def __init__(
        self,
        llm_fn: Callable[[str, str], str],   # fn(system_prompt, user_prompt) -> str
        tools: dict[str, Callable],           # {"tool_name": callable(**kwargs) -> str}
        retriever=None,                        # optional RAG retriever
        track: str = "A",
        max_steps: int = 5,
        step_timeout: float = 60.0,           # seconds per LLM call
    ):
        self.llm      = llm_fn
        self.tools    = tools
        self.retriever = retriever
        self.track    = track
        self.max_steps = max_steps
        self.step_timeout = step_timeout
        self.cache    = ToolCallCache()
        self.tool_descriptions = self._build_tool_descriptions()

    def _build_tool_descriptions(self) -> str:
        lines = []
        for name, fn in self.tools.items():
            doc = (fn.__doc__ or "No description.").strip().split("\n")[0]
            lines.append(f"- {name}: {doc}")
        return "\n".join(lines)

    def _call_tool(self, tool_name: str, args: dict) -> str:
        # Check cache first
        cached = self.cache.get(tool_name, args)
        if cached is not None:
            return f"[CACHED] {cached}"

        if tool_name not in self.tools:
            return f"ERROR: Unknown tool '{tool_name}'. Available: {list(self.tools)}"
        try:
            result = self.tools[tool_name](**args)
            self.cache.set(tool_name, args, result)
            return str(result)
        except Exception as e:
            return f"ERROR calling {tool_name}: {e}"

    def run(self, question: str, question_id: str = "q0") -> tuple[str, QuestionMemory]:
        mem = QuestionMemory(question_id=question_id, question=question, track=self.track)
        rag_ctx = self.retriever.query(question) if self.retriever else ""
        start = time.time()

        for step in range(self.max_steps):
            elapsed = time.time() - start
            if elapsed > 270:  # 4.5 min safety cutoff — leave buffer for Phase 3 <5min discount
                mem.final_answer = "TIMEOUT"
                mem.resolved = False
                return "TIMEOUT", mem

            system_prompt, user_prompt = build_prompt(
                track=self.track,
                question=question,
                memory=mem.to_dict_list(),
                tool_descriptions=self.tool_descriptions,
                rag_context=rag_ctx
            )

            raw_output = self.llm(system_prompt, user_prompt)
            thought, action, action_input = parse_react_output(raw_output)

            if action == "FINAL_ANSWER":
                answer = (
                    format_track_a_answer(action_input)
                    if self.track == "A"
                    else str(action_input).strip()
                )
                mem.final_answer = answer
                mem.resolved = True
                return answer, mem

            obs = self._call_tool(action, action_input if isinstance(action_input, dict) else {})
            mem.add_step(thought=thought, tool=action, args=action_input, obs=obs)

        # Fallback: agent exceeded max steps
        mem.final_answer = "ESCALATE"
        return "ESCALATE", mem
