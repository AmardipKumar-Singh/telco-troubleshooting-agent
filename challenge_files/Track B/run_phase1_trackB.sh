#!/usr/bin/env bash
# =============================================================================
# run_phase1_trackB.sh — Launch Track B Phase 1 agent
# Uses local tool server (server.py) + OpenClaw agent
# 50 questions, accuracy-only scoring, max 2 concurrent, 1000 API calls/day
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Python ──────────────────────────────────────────────────────────────────
if [ -f "$SCRIPT_DIR/.venv/bin/python" ]; then
    PYTHON="$SCRIPT_DIR/.venv/bin/python"
else
    PYTHON="python3"
fi
echo "[INFO] Using Python: $PYTHON"

# ── Copy openclaw config & skills into OpenClaw install dir ─────────────────
OPENCLAW_DIR="/opt/homebrew/lib/node_modules/openclaw"

echo "[INFO] Syncing openclaw_config into OpenClaw..."
cp -f agent/openclaw_config/IDENTITY.md  "$OPENCLAW_DIR/"
cp -f agent/openclaw_config/SOUL.md      "$OPENCLAW_DIR/"
cp -f agent/openclaw_config/USER.md      "$OPENCLAW_DIR/"
cp -f agent/openclaw_config/AGENTS.md    "$OPENCLAW_DIR/"
cp -f agent/openclaw_config/TOOLS.md     "$OPENCLAW_DIR/"

echo "[INFO] Syncing skills into OpenClaw..."
mkdir -p "$OPENCLAW_DIR/skills"
cp -rf agent/skills/infra_maintenance "$OPENCLAW_DIR/skills/"
cp -rf agent/skills/l2_link           "$OPENCLAW_DIR/skills/"
cp -rf agent/skills/l3_route          "$OPENCLAW_DIR/skills/"
cp -rf agent/skills/adv_tunnel        "$OPENCLAW_DIR/skills/"

# ── Local Tool Server ────────────────────────────────────────────────────────
SERVER_URL="http://localhost:7860"

if curl -s --max-time 2 "$SERVER_URL/health" > /dev/null 2>&1; then
    echo "[INFO] Tool Server already running at $SERVER_URL"
else
    echo "[INFO] Starting local Tool Server (server.py)..."
    DATA_SOURCE=data/Phase_1 \
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
        echo "ERROR: Tool Server failed to start. Check server.py logs."
        exit 1
    fi
fi

# ── Install Python deps for agent ────────────────────────────────────────────
pip install -q -r agent/requirements.txt

# ── Run evaluation ───────────────────────────────────────────────────────────
echo "[INFO] Starting Track B Phase 1 evaluation..."
echo "[INFO] Input:   $SCRIPT_DIR/data/Phase_1/test.json"
echo "[INFO] Results: $SCRIPT_DIR/agent/eval_results/result.csv"

"$PYTHON" agent/evaluate_openclaw.py \
    -i data/Phase_1/test.json \
    --concurrency 2

echo ""
echo "[DONE] Submit: $SCRIPT_DIR/agent/eval_results/result.csv"
