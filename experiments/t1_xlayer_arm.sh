#!/bin/bash
# T1 guarded cross-layer fetch arm on Santa Cruz. GUARDED per T4 runbook:
# refuses on any live SLOT_LOCK_*, holds SLOT_LOCK_T1 (trap-removed), installs
# the T1 file (superset of LIP-v3 d9850d99; all flags default off), restarts ONCE
# with MOEFIT_XLAYER_D=1 (OMLX_ADMISSION unset), verify-gates, runs flag-toggle
# arms with no restart, then restores d9850d99 + default env.
set -euo pipefail
export PATH="/opt/homebrew/bin:$PATH"
OMLX_BASE="$HOME/.omlx"; PORT=8000
LOG="$OMLX_BASE/logs/moefit-serve.log"; PIDFILE="$OMLX_BASE/moefit-serve.pid"
BENCH=/tmp/moefit-bench
PATCHED=/opt/homebrew/Cellar/omlx/0.7.0/libexec/lib/python3.11/site-packages/omlx/patches/moe_expert_offload.py
T1FILE=/tmp/omlx_t1_xlayer.py; T1MD5=f18b33cc8f4623dff21711b01024a122
RESTORE=/tmp/omlx_t2_sidecar_plus_lip_v3.py; RESTOREMD5=d9850d99ab0be7ee723fee72e360adbc

locks=$(ls "$BENCH"/SLOT_LOCK_* 2>/dev/null || true)
if [ -n "$locks" ]; then echo "REFUSED: live slot lock(s): $locks"; exit 2; fi
[ "$(md5 -q $T1FILE)" = "$T1MD5" ] || { echo "T1 file md5 mismatch"; exit 3; }
[ "$(md5 -q $RESTORE)" = "$RESTOREMD5" ] || { echo "restore file md5 mismatch"; exit 3; }
[ "$(pgrep -f omlx-server | wc -l | tr -d ' ')" = "1" ] || { echo "not exactly one omlx-server"; exit 4; }

stop_omlx() {
  if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    kill "$(cat "$PIDFILE")" || true
    for i in $(seq 1 60); do kill -0 "$(cat "$PIDFILE")" 2>/dev/null || break; sleep 1; done
    kill -9 "$(cat "$PIDFILE")" 2>/dev/null || true; rm -f "$PIDFILE"
  fi
  pids=$(lsof -tiTCP:$PORT -sTCP:LISTEN 2>/dev/null || true)
  if [ -n "${pids:-}" ]; then kill $pids 2>/dev/null || true; sleep 2; fi
}
wait_mem() {
  for i in $(seq 1 120); do
    f=$(memory_pressure -Q 2>/dev/null | grep -o 'free percentage: [0-9]*' | grep -o '[0-9]*')
    if [ -n "${f:-}" ] && [ "$f" -ge 70 ]; then echo "mem ok (${f}%)"; return 0; fi; sleep 2
  done; echo "mem wait timeout"
}
start_omlx() {
  nohup env "$@" omlx serve --model-dir "$OMLX_BASE/models" --port "$PORT" \
    --max-concurrent-requests 1 --memory-guard aggressive >"$LOG" 2>&1 &
  echo $! >"$PIDFILE"; echo "started pid=$(cat "$PIDFILE") env=$*"
  for i in $(seq 1 300); do
    curl -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { echo ready; return 0; }
    kill -0 "$(cat "$PIDFILE")" 2>/dev/null || { echo died; tail -5 "$LOG"; return 1; }
    sleep 1
  done; echo timeout; return 1
}
restore() {
  echo "=== restore d9850d99 + default env ==="
  rm -f /tmp/omlx_xlayer_on
  stop_omlx; cp "$RESTORE" "$PATCHED"; md5 -q "$PATCHED"; wait_mem; start_omlx
  rm -f "$BENCH/SLOT_LOCK_T1"
}

echo "lead_silicon T1 xlayer arm $(date)" > "$BENCH/SLOT_LOCK_T1"
trap 'restore' EXIT
rm -f /tmp/omlx_xlayer_on
cp "$T1FILE" "$PATCHED"; [ "$(md5 -q "$PATCHED")" = "$T1MD5" ] || exit 5
stop_omlx; wait_mem; rm -f /tmp/omlx_moe_stats.json
start_omlx MOEFIT_XLAYER_D=1
grep -q "T1 xlayer links=47" "$LOG" && echo "links=47 ok" || echo "WARN: links line not in log (log level?); counters checked by aba script"
sleep 3
cd "$BENCH"
/usr/bin/python3 /tmp/t1_xlayer_aba.py --arms "${ARMS:-A0,A,B,A2,B2,A3,B3}" --out "$BENCH/t1_xlayer_aba.json"
echo DONE_T1_XLAYER
