# Telco Troubleshooting Agent

Autonomous ReAct agent for the [ITU Telco Troubleshooting Agentic Challenge](https://zindi.africa/competitions/telco-troubleshooting-agentic-challenge) — solving wireless (Track A) and IP (Track B) network fault diagnosis tasks using **Qwen3.5-35B-A3B**.

---

## Architecture

```
telco_agent/
├── main.py                  # Entry point: runs both tracks, writes result.csv + traces.json
├── agent/
│   ├── react_agent.py       # Core ReAct loop (Thought → Action → Observation)
│   ├── prompt_builder.py    # CoT system/user prompt templates per track
│   └── memory.py            # Per-question state manager + tool call cache
├── tools/
│   ├── track_a_client.py    # HTTP wrappers for Track A wireless server.py tools
│   └── track_b_client.py    # HTTP wrappers for Track B IP server.py tools (~45 tools)
├── rag/
│   ├── build_index.py       # One-time FAISS index builder
│   └── retriever.py         # Query-time top-k retrieval
└── requirements.txt
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set environment variables

```bash
export TRACK_A_SERVER_URL=http://localhost:8000
export TRACK_A_AUTH_TOKEN=your_token_here
export TRACK_B_SERVER_URL=http://localhost:8001
export TRACK_B_AUTH_TOKEN=your_token_here
```

### 3. Build the RAG index (run once)

```bash
python rag/build_index.py
```

### 4. Run the agent

```bash
python main.py
```

Outputs:
- `result.csv` — submission file (`ID, Track A, Track B`)
- `traces.json` — execution trace for Phase 2 upload

---

## Scoring

| Track | Metric | Notes |
|-------|--------|-------|
| Track A | Intersection over Union (IoU) | Multi-answers: `C3\|C7\|C11` ascending |
| Track B | Exact match | Open-ended, one precise answer |

Phase 3 efficiency discount:

| Answering time | Discount |
|----------------|----------|
| < 5 minutes | 100% |
| 5–10 minutes | 80% |
| 10–15 minutes | 60% |
| > 15 minutes | 0% |

---

## Model

Base model: **Qwen3.5-35B-A3B** (mandatory per challenge rules).
- 35B total params, ~3B active (Sparse MoE)
- Served via `transformers` (BF16) or `vllm` (FP8/AWQ for speed)
- Fine-tuning with LoRA/QLoRA is allowed and recommended

---

## License

MIT
