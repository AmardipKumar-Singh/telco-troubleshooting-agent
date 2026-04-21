"""
Entry point for the ITU Telco Troubleshooting Agentic Challenge.

Steps:
  1. Load Track A (500 Qs) and Track B (50 Qs) test files.
  2. Run the ReAct agent on each question.
  3. Write result.csv (ID, Track A, Track B) and traces.json.

Usage:
  python main.py [--track-a-only] [--track-b-only] [--limit N]
"""
import argparse
import csv
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from agent.react_agent import TelcoReActAgent
from tools.track_a_client import TRACK_A_TOOLS
from tools.track_b_client import TRACK_B_TOOLS
from rag.retriever import TelcoRetriever

# ── Model ──────────────────────────────────────────────────────────────────────

MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen3.5-35B-A3B")

print(f"Loading model: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()


def llm_fn(system_prompt: str, user_prompt: str) -> str:
    """Call Qwen3.5-35B-A3B with system + user messages. Returns generated text."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            inputs,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0][inputs.shape[1]:], skip_special_tokens=True)


# ── RAG ────────────────────────────────────────────────────────────────────────

retriever = None
if os.path.exists("rag/index.faiss"):
    print("Loading RAG retriever...")
    retriever = TelcoRetriever()
else:
    print("WARNING: RAG index not found. Run `python rag/build_index.py` first.")


# ── Agents ─────────────────────────────────────────────────────────────────────

agent_a = TelcoReActAgent(
    llm_fn=llm_fn,
    tools=TRACK_A_TOOLS,
    retriever=retriever,
    track="A",
    max_steps=4,
    cutoff_sec=270.0,
)

agent_b = TelcoReActAgent(
    llm_fn=llm_fn,
    tools=TRACK_B_TOOLS,
    retriever=retriever,
    track="B",
    max_steps=6,
    cutoff_sec=270.0,
)


# ── Data loading ───────────────────────────────────────────────────────────────

def load_json(path: str) -> list:
    if not os.path.exists(path):
        print(f"WARNING: {path} not found — returning empty list.")
        return []
    return json.load(open(path))


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args):
    track_a_data = load_json("Track A/data/Phase_1/test.json")   # 500 Qs
    track_b_data = load_json("Track B/data/Phase_1/train.json")  # 50 Qs

    if args.limit:
        track_a_data = track_a_data[:args.limit]
        track_b_data = track_b_data[:args.limit]

    n = max(len(track_a_data), len(track_b_data))
    rows, traces = [], []

    for i in range(n):
        qa = track_a_data[i] if i < len(track_a_data) else None
        qb = track_b_data[i] if i < len(track_b_data) else None

        ans_a, ans_b = "", ""

        if qa and not args.track_b_only:
            t0 = time.time()
            ans_a, mem_a = agent_a.run(qa["question"], question_id=str(qa.get("id", i)))
            traces.append(mem_a.to_trace())
            print(f"[A {i+1}/{len(track_a_data)}] ({time.time()-t0:.1f}s) {ans_a[:80]}")

        if qb and not args.track_a_only:
            t0 = time.time()
            ans_b, mem_b = agent_b.run(qb["question"], question_id=str(qb.get("id", i)))
            traces.append(mem_b.to_trace())
            print(f"[B {i+1}/{len(track_b_data)}] ({time.time()-t0:.1f}s) {ans_b[:80]}")

        rows.append({"ID": i + 1, "Track A": ans_a, "Track B": ans_b})

    # Write result.csv
    with open("result.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "Track A", "Track B"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n✓ result.csv written ({len(rows)} rows)")

    # Write traces.json (required for Phase 2 upload)
    with open("traces.json", "w") as f:
        json.dump(traces, f, indent=2)
    print("✓ traces.json written")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--track-a-only", action="store_true")
    parser.add_argument("--track-b-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N questions per track (for testing)")
    main(parser.parse_args())
