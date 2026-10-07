#!/bin/bash
# T3 LIP ON arm — v3 (post-loop demotion, composed md5 d9850d99).
# GUARDED: refuses whenever any SLOT_LOCK_* is live (T4 runbook rule);
# verify request gate before the 3x1024; mem-wait restarts; restores default env.
set -euo pipefail
export PATH="/opt/homebrew/bin:$PATH"
OMLX_BASE="$HOME/.omlx"
PORT=8000
LOG="$OMLX_BASE/logs/moefit-serve.log"
PIDFILE="$OMLX_BASE/moefit-serve.pid"
BENCH=/tmp/moefit-bench
MODEL=Qwen3.8-Flash-Next-oQ4e-mtp
PATCHED=/opt/homebrew/Cellar/omlx/0.7.0/libexec/lib/python3.11/site-packages/omlx/patches/moe_expert_offload.py

locks=$(ls "$BENCH"/SLOT_LOCK_* 2>/dev/null | grep -v '\.cleared' | grep -v FROZEN || true)
if [ -n "$locks" ]; then
  echo "REFUSED: live slot lock(s): $locks"; exit 2
fi

stop_omlx() {
  if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    kill "$(cat "$PIDFILE")" || true
    for i in $(seq 1 60); do kill -0 "$(cat "$PIDFILE")" 2>/dev/null || break; sleep 1; done
    kill -9 "$(cat "$PIDFILE")" 2>/dev/null || true
    rm -f "$PIDFILE"
  fi
  pids=$(lsof -tiTCP:$PORT -sTCP:LISTEN 2>/dev/null || true)
  if [ -n "${pids:-}" ]; then kill $pids 2>/dev/null || true; sleep 2; fi
}

wait_mem() {
  for i in $(seq 1 120); do
    f=$(memory_pressure -Q 2>/dev/null | grep -o 'free percentage: [0-9]*' | grep -o '[0-9]*')
    if [ -n "${f:-}" ] && [ "$f" -ge 70 ]; then echo "mem ok (${f}%)"; return 0; fi
    sleep 2
  done
  echo "mem wait timeout (${f:-?}%)"
}

start_omlx() {
  nohup env "$@" omlx serve --model-dir "$OMLX_BASE/models" --port "$PORT" \
    --max-concurrent-requests 1 --memory-guard aggressive >"$LOG" 2>&1 &
  echo $! >"$PIDFILE"
  echo "started pid=$(cat "$PIDFILE") env=$*"
  for i in $(seq 1 300); do
    curl -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { echo ready; return 0; }
    kill -0 "$(cat "$PIDFILE")" 2>/dev/null || { echo died; tail -5 "$LOG"; return 1; }
    sleep 1
  done
  echo timeout; return 1
}

echo "=== claim slot ==="
echo "T3 LIP v3 arm $(date)" > "$BENCH/SLOT_LOCK_LIP"
trap 'rm -f "$BENCH/SLOT_LOCK_LIP"' EXIT

echo "=== install v3 composed file (backup .d420b305.bak already on disk) ==="
md5 -q "$PATCHED" || true
cp /tmp/omlx_t2_sidecar_plus_lip_v3.py "$PATCHED"
md5 -q "$PATCHED"   # must be d9850d99ab0be7ee723fee72e360adbc

echo "=== stop, wait mem, start LIP ON (OMLX_ADMISSION=1) ==="
stop_omlx
wait_mem
rm -f /tmp/omlx_moe_stats.json
start_omlx OMLX_ADMISSION=1

echo "=== VERIFY request (64 tokens; must finish=length) ==="
sleep 5
curl -s --max-time 300 http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a long, detailed essay on the history of unified memory architectures in personal computers, from early shared-memory designs to modern unified memory fabrics.\"}],\"max_tokens\":64,\"stream\":true}" \
  > "$BENCH/lip5_verify.stream"
fin=$(grep -o '"finish_reason":"[a-z]*"' "$BENCH/lip5_verify.stream" | tail -1 || true)
echo "verify finish: $fin"
if [ -z "${fin:-}" ] || ! echo "$fin" | grep -q length; then
  echo "VERIFY FAILED - aborting (server left up for diagnosis)"
  exit 1
fi

echo "=== collect 3x n=1024 ==="
cd "$BENCH"
/usr/bin/python3 t9_collect.py --url http://127.0.0.1:8000/v1/chat/completions \
  --model "$MODEL" --max-tokens 1024 --runs 3 \
  --label "T3 LIP ON v3 (post-loop demotion, OMLX_ADMISSION=1) n1024" \
  --host m4max-36gb --ram-gib 36 --out t3_lip_on5.json

echo "=== snapshot ON counters (new process: full window from 0) ==="
sleep 3
cp /tmp/omlx_moe_stats.json "$BENCH/lip5_onS.json" || true
cat "$BENCH/lip5_onS.json" || true

echo "=== restore default env (v3 file stays; flags-off = stock behavior) ==="
stop_omlx
wait_mem
start_omlx
echo DONE_LIP_ARM5
