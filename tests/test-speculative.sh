#!/bin/bash
# test-speculative.sh — Speculative decoding E2E verification
#
# Uses a small draft model (Qwen3.5-0.8B) to accelerate a larger main model
# (Qwen3.5-4B) via speculative decoding. Verifies:
#   1. Dual-model loading (draft + main)
#   2. Speculative decoding path activation
#   3. Correct token generation
#   4. Server stability under dual-model memory pressure
#
# Usage:
#   ./tests/test-speculative.sh [binary_path] [port]
#
# Requirements:
#   - ~4 GB RAM (0.8B draft ~1 GB + 4B main ~3 GB)
#   - macos-15 (7 GB) on GitHub Actions is sufficient
#   - curl, jq

set -euo pipefail

BINARY="${1:-.build/release/SwiftLM}"
PORT="${2:-15414}"
HOST="127.0.0.1"
MAIN_MODEL="${MAIN_MODEL:-mlx-community/Qwen3.5-2B-4bit}"
DRAFT_MODEL="${DRAFT_MODEL:-mlx-community/Qwen3.5-0.8B-MLX-4bit}"
NUM_DRAFT_TOKENS=4
URL="http://${HOST}:${PORT}"
PASS=0
FAIL=0
TOTAL=0
LOG_FILE="/tmp/SwiftLM-test-speculative.log"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${YELLOW}[spec-test]${NC} $*"; }
pass() { PASS=$((PASS + 1)); TOTAL=$((TOTAL + 1)); echo -e "  ${GREEN}✅ PASS${NC}: $*"; }
fail() { FAIL=$((FAIL + 1)); TOTAL=$((TOTAL + 1)); echo -e "  ${RED}❌ FAIL${NC}: $*"; }

cleanup() {
    if [ -n "${SERVER_PID:-}" ]; then
        log "Stopping server (PID $SERVER_PID)"
        kill -9 "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# ── Check prerequisites ─────────────────────────────────────────────
if [ ! -f "$BINARY" ]; then
    echo "Error: Binary not found at $BINARY"
    echo "Run 'swift build -c release' first."
    exit 1
fi

if ! command -v jq &>/dev/null; then
    echo "Error: jq is required. Install with: brew install jq"
    exit 1
fi

# ── Memory check ────────────────────────────────────────────────────
TOTAL_RAM_GB=$(sysctl -n hw.memsize 2>/dev/null | awk '{printf "%.0f", $1 / 1073741824}')
log "System RAM: ${TOTAL_RAM_GB} GB"

if [ "$TOTAL_RAM_GB" -lt 8 ] 2>/dev/null; then
    log "⚠️  WARNING: ${TOTAL_RAM_GB} GB RAM detected. Dual-model test requires ~6 GB."
    log "   Consider running on a machine with ≥8 GB RAM."
fi

# ══════════════════════════════════════════════════════════════════════
echo -e "\n${CYAN}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║  SwiftLM Speculative Decoding E2E Test                  ║${NC}"
echo -e "${CYAN}║  Draft: Qwen3.5-0.8B (4-bit) → Main: Qwen3.5-2B (4-bit) ║${NC}"
echo -e "${CYAN}║  Draft tokens per round: ${NUM_DRAFT_TOKENS}                              ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════╝${NC}\n"

# ── Start server with dual models ───────────────────────────────────
log "Starting server with speculative decoding..."
log "  Main model:  $MAIN_MODEL"
log "  Draft model: $DRAFT_MODEL"
log "  Draft tokens per round: $NUM_DRAFT_TOKENS"

"$BINARY" --model "$MAIN_MODEL" --port "$PORT" --host "$HOST" \
    --draft-model "$DRAFT_MODEL" \
    --num-draft-tokens "$NUM_DRAFT_TOKENS" \
    > "$LOG_FILE" 2>&1 &
SERVER_PID=$!

# Wait for server to be ready (both models need to download + load)
log "Waiting for server to load both models (this may take a while on first run)..."
MAX_WAIT=900  # 15 minutes for two model downloads
for i in $(seq 1 "$MAX_WAIT"); do
    if curl -sf "$URL/health" >/dev/null 2>&1; then
        log "Server ready after ${i}s"
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Error: Server process died. Server Log:"
        cat "$LOG_FILE"
        exit 1
    fi
    # Print progress every 30 seconds
    if [ $((i % 30)) -eq 0 ]; then
        log "  Still waiting... (${i}s elapsed)"
    fi
    sleep 1
done

if ! curl -sf "$URL/health" >/dev/null 2>&1; then
    echo "Error: Server did not become ready in ${MAX_WAIT}s"
    echo "Server Log:"
    cat "$LOG_FILE"
    exit 1
fi

# ── Test 1: Verify server loaded both models ────────────────────────
log "Test 1: Verify dual-model loading"

# Check server log for draft model loading confirmation
if grep -q "Draft model loaded successfully" "$LOG_FILE"; then
    pass "Draft model loaded successfully"
else
    fail "Draft model loading not confirmed in server logs"
fi

if grep -q "speculative decoding" "$LOG_FILE"; then
    pass "Speculative decoding mode detected in server logs"
else
    fail "Speculative decoding not mentioned in server logs"
fi

# ── Test 2: Health endpoint works with dual models ──────────────────
log "Test 2: Health endpoint"

HEALTH=$(curl -sf "$URL/health")
if echo "$HEALTH" | jq -e '.status == "ok"' >/dev/null 2>&1; then
    pass "Health endpoint returns status=ok"
else
    fail "Health endpoint: $HEALTH"
fi

# ── Test 3: Streaming speculative generation ────────────────────────
log "Test 3: Streaming speculative generation"

STREAM_OUTPUT=$(curl -sf -N --max-time 120 -X POST "$URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MAIN_MODEL\",\"stream\":true,\"max_tokens\":30,\"messages\":[{\"role\":\"user\",\"content\":\"Name three fruits.\"}]}" \
    2>/dev/null || true)

if echo "$STREAM_OUTPUT" | grep -q "data: \[DONE\]"; then
    pass "Streaming speculative: received [DONE] sentinel"
else
    fail "Streaming speculative: missing [DONE] sentinel"
fi

CHUNK_COUNT=$(echo "$STREAM_OUTPUT" | grep -c "^data: {" || true)
if [ "$CHUNK_COUNT" -gt 0 ]; then
    pass "Streaming speculative: received $CHUNK_COUNT data chunks"
else
    fail "Streaming speculative: no data chunks received"
fi

# Check server log for speculative decoding activation
if grep -q "Using speculative decoding" "$LOG_FILE"; then
    pass "Speculative decoding path activated during generation"
else
    fail "Speculative decoding path not activated (missing log line)"
fi

# ── Test 5: Multiple sequential requests (stability) ────────────────
log "Test 5: Sequential request stability (3 requests)"

SEQ_PASS=true
for i in 1 2 3; do
    SEQ_RESP=$(curl -sf --max-time 120 -X POST "$URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$MAIN_MODEL\",\"max_tokens\":10,\"messages\":[{\"role\":\"user\",\"content\":\"Say the number $i.\"}]}" 2>/dev/null || echo "")

    SEQ_CONTENT=$(echo "$SEQ_RESP" | jq -r '.choices[0].message.content // empty' 2>/dev/null || echo "")

    if [ -z "$SEQ_CONTENT" ]; then
        SEQ_PASS=false
        fail "Sequential request $i: empty response"
        break
    fi
done

if [ "$SEQ_PASS" = true ]; then
    pass "Sequential stability: 3/3 speculative requests completed successfully"
fi

# ── Test 6: Memory stability check ─────────────────────────────────
log "Test 6: Memory stability"

HEALTH_FINAL=$(curl -sf "$URL/health")
MEM_ACTIVE=$(echo "$HEALTH_FINAL" | jq -r '.memory.active_mb // 0')
MEM_PEAK=$(echo "$HEALTH_FINAL" | jq -r '.memory.peak_mb // 0')

if [ "$MEM_ACTIVE" -gt 0 ] 2>/dev/null; then
    pass "Memory: active=${MEM_ACTIVE} MB, peak=${MEM_PEAK} MB"
else
    fail "Memory: could not read memory stats"
fi

# Verify server is still responsive after all tests
if curl -sf "$URL/health" >/dev/null 2>&1; then
    pass "Server still responsive after all speculative decoding tests"
else
    fail "Server became unresponsive"
fi

# ── Test 7: same checkpoint, no draft -> auto-detected qwen3_5 VLM ──
# Every other Qwen3.5 launch in CI passes --draft-model/--dflash, which forces the LLM
# factory and bypasses the hybrid prompt-cache split. b769 shipped with every text-only
# chat to an auto-detected Qwen3.5 VLM failing with HTTP 500
# (unsupportedBatchContinuation), and CI never ran that path.
log "Test 7: $MAIN_MODEL without a draft (auto-detected VLM, text-only chat)"
cleanup
"$BINARY" --model "$MAIN_MODEL" --port "$PORT" --host "$HOST" >> "$LOG_FILE" 2>&1 &
SERVER_PID=$!
for _ in $(seq 1 300); do
    kill -0 "$SERVER_PID" 2>/dev/null || break
    curl -sf "$URL/health" >/dev/null 2>&1 && break
    sleep 1
done
T1='{"role":"user","content":"Say hi."}'
T2="$T1"',{"role":"assistant","content":"Hi."},{"role":"user","content":"Say bye."}'
for MSGS in "$T1" "$T2"; do
    N=$(echo "[$MSGS]" | jq length)
    # No -f: keep the error body of a failed request, since the server does not log it.
    R=$(curl -s --max-time 120 -w '\n%{http_code}' -X POST "$URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$MAIN_MODEL\",\"max_tokens\":8,\"temperature\":0,\"messages\":[$MSGS]}" 2>/dev/null || true)
    CODE="${R##*$'\n'}"
    BODY="${R%$'\n'*}"
    if [ "$CODE" = "200" ] \
        && echo "$BODY" | jq -e '.choices[0].finish_reason and .usage.prompt_tokens > 0' >/dev/null 2>&1; then
        pass "No-draft VLM text chat, $N message(s): completed"
    else
        fail "No-draft VLM text chat, $N message(s): no completion (HTTP ${CODE:-none}: ${BODY:0:300})"
    fi
done
# Stop with SIGTERM before reading the log, not cleanup()'s SIGKILL. stdout is a fully
# buffered file here and a failing request never flushes it, so a "Prompt cache HIT" line
# printed just before the error only reaches the log when the server exits normally.
# If the server already died (a crash skips that flush too), the cache check below
# cannot pass.
SERVER_ALIVE=1
kill -0 "$SERVER_PID" 2>/dev/null || SERVER_ALIVE=0
kill -TERM "$SERVER_PID" 2>/dev/null || true
for _ in $(seq 1 30); do
    kill -0 "$SERVER_PID" 2>/dev/null || break
    sleep 1
done
cleanup
SERVER_PID=""
# Guards against this test silently covering the LLM path (auto-detect off, or the
# VLM->LLM fallback, which still prints the auto-detect line).
if grep -q "Loading VLM (vision-language model)" "$LOG_FILE" \
    && ! grep -q "Auto-detected VLM failed to load" "$LOG_FILE"; then
    pass "No-draft: $MAIN_MODEL was loaded as a VLM"
else
    fail "No-draft: $MAIN_MODEL was not loaded as a VLM, so Test 7 did not cover the VLM path"
fi
# The earlier --draft-model runs never restore from the prompt cache, so any hit here
# comes from the VLM server, which must bypass the cache.
if grep -q "Prompt cache HIT" "$LOG_FILE"; then
    fail "No-draft: the VLM server restored from the prompt cache"
elif [ "$SERVER_ALIVE" = 0 ]; then
    fail "No-draft: the server exited during Test 7, so its log may be missing a prompt cache hit"
else
    pass "No-draft: the VLM server bypassed the prompt cache"
fi

# ── Results ──────────────────────────────────────────────────────────
echo ""
log "═══════════════════════════════════════"
log "Speculative Decoding Test Results"
log "  Draft:  $DRAFT_MODEL"
log "  Main:   $MAIN_MODEL"
log "  Tokens/round: $NUM_DRAFT_TOKENS"
log "  Results: ${PASS} passed, ${FAIL} failed, ${TOTAL} total"
log "═══════════════════════════════════════"

if [ "$FAIL" -gt 0 ]; then
    echo ""
    log "Server completely failed. Full Log:"
    cat "$LOG_FILE"
    exit 1
fi

echo ""
log "Server log tail (last 50 lines):"
tail -50 "$LOG_FILE"
exit 0
