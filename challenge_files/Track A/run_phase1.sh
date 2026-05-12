#!/usr/bin/env bash
# =============================================================================
# run_phase1.sh — Launch the Track A agent for Phase 1 submission
# =============================================================================
# Prerequisites:
#   1. Activate venv:  source .venv/bin/activate
#   2. Set your OpenRouter API key:
#        export AGENT_API_KEY="sk-or-v1-..."
#   3. Run: bash run_phase1.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# 1. Resolve Python — use venv python if available, else system python3
# ---------------------------------------------------------------------------
if [ -f "$SCRIPT_DIR/.venv/bin/python" ]; then
    PYTHON="$SCRIPT_DIR/.venv/bin/python"
else
    PYTHON="python3"
fi
echo "[INFO] Using Python: $PYTHON"

# ---------------------------------------------------------------------------
# 2. Validate AGENT_API_KEY is set
# ---------------------------------------------------------------------------
if [ -z "$AGENT_API_KEY" ]; then
    echo "ERROR: AGENT_API_KEY is not set."
    echo "  Get a free OpenRouter key at: https://openrouter.ai/keys"
    echo "  Then run: export AGENT_API_KEY=\"sk-or-v1-...\""
    exit 1
fi

# ---------------------------------------------------------------------------
# 3. Start the Tool Server in the background (if not already running)
# ---------------------------------------------------------------------------
SERVER_URL="http://localhost:7860"

if curl -s --max-time 2 "$SERVER_URL/health" > /dev/null 2>&1; then
    echo "[INFO] Tool Server already running at $SERVER_URL"
else
    echo "[INFO] Starting Tool Server at $SERVER_URL ..."
    DATA_SOURCE=data/Phase_1 DATA_SPLIT=test \
        "$PYTHON" -m uvicorn server:app --host 0.0.0.0 --port 7860 &
    SERVER_PID=$!
    echo "[INFO] Tool Server PID: $SERVER_PID"

    echo "[INFO] Waiting for Tool Server to be ready..."
    for i in $(seq 1 20); do
        if curl -s --max-time 2 "$SERVER_URL/health" > /dev/null 2>&1; then
            echo "[INFO] Tool Server is ready!"
            break
        fi
        echo "  ... retry $i/20"
        sleep 2
    done

    if ! curl -s --max-time 2 "$SERVER_URL/health" > /dev/null 2>&1; then
        echo "ERROR: Tool Server failed to start. Check logs above."
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# 4. Run the agent benchmark
# Primary model: qwen/qwen3.6-plus:free (1M context, free, Qwen flagship May 2026)
# Fallback rotation: Gemma4 -> Llama4 Maverick -> Mistral Small -> DeepSeek R1
# ---------------------------------------------------------------------------
echo "[INFO] Starting agent benchmark with Qwen3.6-Plus (free)..."
echo "[INFO] Results will be saved to: $SCRIPT_DIR/results/result.csv"

"$PYTHON" main.py \
    --server_url "$SERVER_URL" \
    --model_url "https://openrouter.ai/api/v1" \
    --model_name "qwen/qwen3.6-plus:free" \
    --max_iterations 10 \
    --max_tokens 16000 \
    --rate_limit_per_minute 18 \
    --save_dir ./results \
    --save_freq 10 \
    --log_file ./log.log \
    --verbose

echo ""
echo "[DONE] Benchmark complete."
echo "  Submit this file to Zindi: $SCRIPT_DIR/results/result.csv"
