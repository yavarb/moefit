#!/bin/bash
# serve_paging.sh - start oMLX with expert paging for a 24-64 GB Mac and wait
# until the OpenAI-compatible endpoint answers.
#
#   scripts/serve_paging.sh            # defaults below
#   PORT=8000 OMLX_BASE=~/.omlx MEMORY_GUARD=aggressive scripts/serve_paging.sh
#   scripts/serve_paging.sh stop       # stop the server this script started
#
# Models come from $OMLX_BASE/models (see configure_omlx_paging.py). The
# model itself loads on the first request, so the bench's warm-up request is
# what pays the load time.
set -euo pipefail
export PATH="/opt/homebrew/bin:$PATH"
PORT="${PORT:-8000}"
OMLX_BASE="${OMLX_BASE:-$HOME/.omlx}"
MEMORY_GUARD="${MEMORY_GUARD:-aggressive}"
MAX_CONCURRENT="${MAX_CONCURRENT:-1}"
LOG="$OMLX_BASE/logs/moefit-serve.log"
PIDFILE="$OMLX_BASE/moefit-serve.pid"

if [ "${1:-}" = "stop" ]; then
  if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    kill "$(cat "$PIDFILE")"; echo "stopped pid $(cat "$PIDFILE")"; rm -f "$PIDFILE"
  else
    echo "no server started by this script is running"
  fi
  exit 0
fi

command -v omlx >/dev/null || { echo "omlx not on PATH (brew install jundot/omlx/omlx)"; exit 1; }
if curl -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
  echo "something already answers on :$PORT"; curl -fsS "http://127.0.0.1:$PORT/v1/models"; echo
  exit 0
fi
mkdir -p "$OMLX_BASE/logs"
echo "starting: omlx serve --model-dir $OMLX_BASE/models --port $PORT --max-concurrent-requests $MAX_CONCURRENT --memory-guard $MEMORY_GUARD"
nohup omlx serve --model-dir "$OMLX_BASE/models" --port "$PORT" \
  --max-concurrent-requests "$MAX_CONCURRENT" --memory-guard "$MEMORY_GUARD" \
  >"$LOG" 2>&1 &
echo $! >"$PIDFILE"
echo "pid $(cat "$PIDFILE"), log $LOG"
for _ in $(seq 1 120); do
  if curl -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    echo "ready: http://127.0.0.1:$PORT/v1"
    curl -fsS "http://127.0.0.1:$PORT/v1/models" | python3 -c 'import json,sys; print("models:", [m["id"] for m in json.load(sys.stdin)["data"]])'
    exit 0
  fi
  if ! kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "server exited early; last log lines:"; tail -30 "$LOG"; exit 1
  fi
  sleep 1
done
echo "timed out waiting for :$PORT; see $LOG"; exit 1
