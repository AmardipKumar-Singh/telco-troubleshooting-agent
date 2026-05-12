#!/usr/bin/env bash
# =============================================================================
# run_phase1.sh — Launch the Track A agent for Phase 1 submission
# =============================================================================
# Prerequisites:
#   1. Set your OpenRouter API key:
#        export AGENT_API_KEY="sk-or-v1-..."
#   2. Install dependencies:
#        pip install -r requirements.txt
#        pip install uvicorn fastapi
#   3. Run this script from the repo root OR from challenge_files/Track A/
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# 1. Validate AGENT_API_KEY is set
# ---------------------------------------------------------------------------
if [ -z "$AGENT_API_KEY" ]; then
    echo "ERROR: AGENT_API_KEY is not set."
    echo "  Get a free OpenRouter key at: https://openrouter.ai/keys"
    echo "  Then run: export AGENT_API_KEY=\"sk-or-v1-...\""
    exit 1
fi

# ---------------------------------------------------------------------------
# 2. Start the Tool Server in the background (if not already running)
# ---------------------------------------------------------------------------
SERVER_URL="http://localhost:7860"

if curl -s --max-time 2 "$SERVER_URL/health" > /dev/null 2>&1; then
    echo "[INFO] Tool Server already running at $SERVER_URL"
else
    echo "[INFO] Starting Tool Server at $SERVER_URL ..."
    # DATA_SOURCE points to Phase 1 test data
    DATA_SOURCE=data/Phase_1 DATA_SPLIT=test \
        uvicorn server:app --host 0.0.0.0 --port 7860 &
    SERVER_PID=$!
    echo "[INFO] Tool Server PID: $SERVER_PID"

    # Wait for server to be ready
    echo "[INFO] Waiting for Tool Server to be ready..."
    for i in $(seq 1 20); do
        if curl -s --max-time 2 "$SERVER_URL/health" > /dev/null 2>&1; then
            echo "[INFO] Tool Server is ready!"
            break
        fi
        echo "  ... retry $i/20"
        sleep 2
    done

    # Final check
    if ! curl -s --max-time 2 "$SERVER_URL/health" > /dev/null 2>&1; then
        echo "ERROR: Tool Server failed to start. Check logs above."
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# 3. Run the agent benchmark (Phase 1 — all 500 scenarios)
# ---------------------------------------------------------------------------
echo "[INFO] Starting agent benchmark with Qwen3-235B-A22B..."
echo "[INFO] Results will be saved to: $SCRIPT_DIR/results/result.csv"

python main.py \
    --server_url "$SERVER_URL" \
    --model_url "https://openrouter.ai/api/v1" \
    --model_name "qwen/qwen3-235b-a22b:free" \
    --max_iterations 10 \
    --max_tokens 16000 \
    --rate_limit_per_minute 9 \
    --save_dir ./results \
    --save_freq 10 \
    --log_file ./log.log \
    --verbose

echo ""
echo "[DONE] Benchmark complete."
echo "  Submit this file to Zindi: $SCRIPT_DIR/results/result.csv"
