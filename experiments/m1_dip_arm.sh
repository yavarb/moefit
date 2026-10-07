#!/bin/bash
# M1 silicon A/B (LRU-insert vs BIP vs DIP) on M4 Max 36 GB. Owner: silicon_dbuf (T9).
# STAGED ONLY: runs after design_inventor posts DIP code-green + file md5.
#   DIPFILE=/tmp/omlx_m1_dip.py DIPMD5=<md5> bash m1_dip_arm.sh
# Guards (T4 runbook): refuses on any SLOT_LOCK_*, on prefetch flags, on !=1 omlx-server,
# on md5 mismatch. ONE restart with DEFAULT env (T9 IO pool default, OMLX_ADMISSION unset,
# no xlayer); policy toggled ONLY via /tmp/omlx_insert_policy between arms, no restart.
# EXIT trap restores d9850d99 + default env and clears the lock + policy flag.
set -euo pipefail
export PATH="/opt/homebrew/bin:$PATH"
: "${DIPFILE:?set DIPFILE}"; : "${DIPMD5:?set DIPMD5}"
OMLX_BASE="$HOME/.omlx"; PORT=8000
LOG="$OMLX_BASE/logs/moefit-serve.log"; PIDFILE="$OMLX_BASE/moefit-serve.pid"
BENCH=/tmp/moefit-bench
PATCHED=/opt/homebrew/Cellar/omlx/0.7.0/libexec/lib/python3.11/site-packages/omlx/patches/moe_expert_offload.py
RESTORE=/tmp/omlx_t2_sidecar_plus_lip_v3.py; RESTOREMD5=d9850d99ab0be7ee723fee72e360adbc

locks=$(ls "$BENCH"/SLOT_LOCK_* 2>/dev/null || true)
[ -z "$locks" ] || { echo "REFUSED: live slot lock(s): $locks"; exit 2; }
pgrep -f "collect_silicon_run|_aba.py|t6_|lip_arm|xlayer_arm" >/dev/null && { echo "REFUSED: bench process alive"; exit 2; }
for f in /tmp/omlx_sidecar_on /tmp/omlx_xlayer_on; do [ ! -e "$f" ] || { echo "REFUSED: $f present"; exit 2; }; done
[ "$(md5 -q "$DIPFILE")" = "$DIPMD5" ] || { echo "DIP file md5 mismatch"; exit 3; }
[ "$(md5 -q "$RESTORE")" = "$RESTOREMD5" ] || { echo "restore file md5 mismatch"; exit 3; }
[ "$(pgrep -f omlx-server | wc -l | tr -d ' ')" = "1" ] || { echo "not exactly one omlx-server"; exit 4; }

stop_omlx() {
  local P; P=$(cat "$PIDFILE" 2>/dev/null || true)
  local L; L=$(lsof -tiTCP:$PORT -sTCP:LISTEN 2>/dev/null || true)
  for Q in $P $L; do kill "$Q" 2>/dev/null || true; done
  for _ in $(seq 1 90); do lsof -tiTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1 || break; sleep 1; done
  lsof -tiTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1 && { echo "port still held"; return 1; }
  return 0
}
wait_mem() {
  for _ in $(seq 1 120); do
    f=$(memory_pressure -Q 2>/dev/null | grep -o 'free percentage: [0-9]*' | grep -o '[0-9]*')
    [ -n "${f:-}" ] && [ "$f" -ge 70 ] && { echo "mem ok ${f}%"; return 0; }; sleep 2
  done; echo "mem wait timeout"
}
start_omlx() {  # default env only
  nohup omlx serve --model-dir "$OMLX_BASE/models" --port "$PORT" \
    --max-concurrent-requests 1 --memory-guard aggressive >"$LOG" 2>&1 &
  local N=$!; echo $N >"$PIDFILE"
  for _ in $(seq 1 300); do
    curl -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
    kill -0 "$N" 2>/dev/null || { echo died; tail -5 "$LOG"; return 1; }; sleep 1
  done
  local O; O=$(lsof -tiTCP:$PORT -sTCP:LISTEN)
  if [ "$O" != "$N" ] && [ "$(ps -o ppid= -p "$O" | tr -d ' ')" != "$N" ]; then
    echo "listener $O is not launched pid $N"; return 5; fi
  grep -q "Address already in use" "$LOG" && { echo "bind failure"; return 5; }
  echo "server ready pid=$N"
}
restore() {
  echo "=== restore d9850d99 + default env ==="
  rm -f /tmp/omlx_insert_policy
  stop_omlx || true; cp "$RESTORE" "$PATCHED"; md5 -q "$PATCHED"; wait_mem; start_omlx || true
  rm -f "$BENCH/SLOT_LOCK_M1"
}

echo "silicon_dbuf M1 DIP A/B $(date)" >"$BENCH/SLOT_LOCK_M1"
trap restore EXIT
rm -f /tmp/omlx_insert_policy
stop_omlx; cp "$DIPFILE" "$PATCHED"; [ "$(md5 -q "$PATCHED")" = "$DIPMD5" ] || exit 5
wait_mem; rm -f /tmp/omlx_moe_stats.json
start_omlx
sleep 3
cd "$BENCH"
/usr/bin/python3 /tmp/m1_dip_aba.py --rounds "${ROUNDS:-3}" --out "$BENCH/m1_dip_aba.json"
echo DONE_M1_DIP
