#!/usr/bin/env bash
# =============================================================================
# run_phase1.sh — Launch the Track A agent for Phase 1 submission
# Primary model: qwen/qwen3-coder:free (strongest free Qwen, 262K ctx)
# Rotation: 7 active free models with tool-calling support (May 2026)
# Free tier: 20 req/min AND 200 req/day per model
# 7 models x 200 = 1400 req/day budget for 500 scenarios
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f "$SCRIPT_DIR/.venv/bin/python" ]; then
    PYTHON="$SCRIPT_DIR/.venv/bin/python"
else
    PYTHON="python3"
fi
echo "[INFO] Using Python: $PYTHON"

if [ -z "$AGENT_API_KEY" ]; then
    echo "ERROR: AGENT_API_KEY is not set."
    echo "  export AGENT_API_KEY=\"sk-or-v1-...\""
    exit 1
fi

SERVER_URL="http://localhost:7860"

if curl -s --max-time 2 "$SERVER_URL/health" > /dev/null 2>&1; then
    echo "[INFO] Tool Server already running at $SERVER_URL"
else
    echo "[INFO] Starting Tool Server..."
    DATA_SOURCE=data/Phase_1 DATA_SPLIT=test \
        "$PYTHON" -m uvicorn server:app --host 0.0.0.0 --port 7860 &
    SERVER_PID=$!
    echo "[INFO] Tool Server PID: $SERVER_PID"
    for i in $(seq 1 20); do
        if curl -s --max-time 2 "$SERVER_URL/health" > /dev/null 2>&1; then
            echo "[INFO] Tool Server is ready!"
            break
        fi
        echo "  ... retry $i/20"
        sleep 2
    done
    if ! curl -s --max-time 2 "$SERVER_URL/health" > /dev/null 2>&1; then
        echo "ERROR: Tool Server failed to start."
        exit 1
    fi
fi

echo "[INFO] Starting agent benchmark (primary: qwen/qwen3-coder:free)..."
echo "[INFO] Results -> $SCRIPT_DIR/results/result.csv"

"$PYTHON" main.py \
    --server_url "$SERVER_URL" \
    --model_url "https://openrouter.ai/api/v1" \
    --model_name "qwen/qwen3-coder:free" \
    --max_iterations 10 \
    --max_tokens 16000 \
    --rate_limit_per_minute 15 \
    --save_dir ./results \
    --save_freq 5 \
    --log_file ./log.log \
    --verbose

echo ""
echo "[DONE] Submit: $SCRIPT_DIR/results/result.csv"
