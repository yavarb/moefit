#!/bin/bash
# ONE LIP ON arm then restore stock. Does not touch :8317.
set -euo pipefail
export PATH="/opt/homebrew/bin:$PATH"
OMLX_BASE="${OMLX_BASE:-$HOME/.omlx}"
PORT=8000
LOG="$OMLX_BASE/logs/moefit-serve.log"
PIDFILE="$OMLX_BASE/moefit-serve.pid"
BENCH=/tmp/moefit-bench
REPO=~/work/moefit
MODEL=Qwen3.8-Flash-Next-oQ4e-mtp

stop_omlx() {
  if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    kill "$(cat "$PIDFILE")" || true
    for i in $(seq 1 60); do
      kill -0 "$(cat "$PIDFILE")" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      kill -9 "$(cat "$PIDFILE")" || true
    fi
    rm -f "$PIDFILE"
  fi
  # also clear any stray omlx-server on :8000 (not 8317)
  pids=$(lsof -tiTCP:$PORT -sTCP:LISTEN 2>/dev/null || true)
  if [ -n "${pids:-}" ]; then
    kill $pids 2>/dev/null || true
    sleep 2
  fi
}

start_omlx() {
  local tag="$1"; shift
  mkdir -p "$OMLX_BASE/logs" "$BENCH"
  : > "$BENCH/serve_${tag}.log"
  # inherit caller env (OMLX_ADMISSION etc)
  nohup env "$@" omlx serve --model-dir "$OMLX_BASE/models" --port "$PORT"     --max-concurrent-requests 1 --memory-guard aggressive     >"$LOG" 2>&1 &
  echo $! >"$PIDFILE"
  echo "started pid=$(cat "$PIDFILE") tag=$tag env=$*"
  for i in $(seq 1 180); do
    if curl -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
      echo ready
      return 0
    fi
    if ! kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "server died"; tail -40 "$LOG"; return 1
    fi
    sleep 1
  done
  echo timeout; tail -40 "$LOG"; return 1
}

echo "=== LIP ON arm restart ==="
stop_omlx
start_omlx lip_on OMLX_ADMISSION=1
# warm + collect
cd "$BENCH"
/usr/bin/python3 t9_collect.py --url http://127.0.0.1:8000/v1/chat/completions   --model "$MODEL" --max-tokens 1024 --runs 3   --label "T3 LIP ON (OMLX_ADMISSION=1, default IO_WORKERS) n1024"   --host m4max-36gb --ram-gib 36 --out t3_lip_on.json

echo "=== restore stock (ADMISSION unset) ==="
stop_omlx
start_omlx stock_restore
# optional quick confirm
curl -fsS http://127.0.0.1:8000/v1/models | head -c 200; echo
echo DONE_LIP_ARM
