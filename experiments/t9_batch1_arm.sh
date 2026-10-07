#!/bin/bash
# T9 DB-isolating arm for Santa Cruz (silicon_dbuf).
#
# OMLX_MOE_OFFLOAD_IO_BATCH=1 keeps the 12-worker pool (slab-parallel reads +
# the decode overlap path stay ON) but shrinks the read-ahead window to one
# expert: expert i+1's reads are submitted only after expert i is installed
# (source-traced in _ensure_ids.prefetch on the installed file). So
#   ON(default, 15.72) - BATCH1  = the pure double-buffer / staged-install term
# which is the quantity T4's sim (+1.15) and prereg (+0.33..+1.10) priced.
#
# Usage on Santa Cruz (only when the box is idle and the queue owner agrees):
#   bash /tmp/moefit-bench/t9_batch1_arm.sh
# It restarts omlx with IO_BATCH=1, runs 3x n=1024 on the same 80-token prompt
# with the current repo collector (carries prompt_sha256), snapshots T2's
# counters, then RESTORES the default-env server. :8317 is never touched.
set -euo pipefail
export PATH=/opt/homebrew/bin:$PATH
B=/tmp/moefit-bench
PIDF=~/.omlx/moefit-serve.pid
LOG=~/.omlx/logs/moefit-serve.log
URL=http://127.0.0.1:8000/v1/chat/completions
MODEL=Qwen3.8-Flash-Next-oQ4e-mtp

if pgrep -f "collect_silicon_run|t9_collect|t2_sidecar_aba|t6_ssd_proof|score_lip|lip_on" >/dev/null; then
  echo "another bench is running; refusing"; pgrep -fl "collect|aba|t6_" ; exit 2
fi
[ -e /tmp/omlx_sidecar_on ] && { echo "sidecar flag present; refusing"; exit 2; }

restart() {  # $@ = extra env assignments
  local P; P=$(cat "$PIDF")
  cp "$LOG" "$B/serve_before_$(date +%H%M%S).log" || true
  kill "$P"; for _ in $(seq 1 60); do kill -0 "$P" 2>/dev/null || break; sleep 1; done
  kill -0 "$P" 2>/dev/null && { echo "old server $P did not exit"; exit 3; }
  env "$@" nohup omlx serve --model-dir ~/.omlx/models --port 8000 \
    --max-concurrent-requests 1 --memory-guard aggressive >"$LOG" 2>&1 &
  echo $! >"$PIDF"
  for _ in $(seq 1 180); do curl -fsS http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && break; sleep 1; done
  curl -fsS http://127.0.0.1:8000/v1/models >/dev/null || { echo "server not ready"; tail -20 "$LOG"; exit 4; }
  echo "server pid $(cat "$PIDF") env: $(ps eww -o command= -p "$(cat "$PIDF")" | tr ' ' '\n' | grep '^OMLX_MOE' || echo default)"
}

restart OMLX_MOE_OFFLOAD_IO_BATCH=1
cp /tmp/omlx_moe_stats.json "$B/t9_b1_stats0.json" 2>/dev/null || true
/usr/bin/python3 "$B/collect_silicon_run.py" --url "$URL" --model "$MODEL" --max-tokens 1024 --runs 3 \
  --label "T9 DB-isolating arm: OMLX_MOE_OFFLOAD_IO_BATCH=1 (workers 12, overlap on) n1024" \
  --host santacruz --ram-gib 36 --out "$B/t9_batch1_C.json"
sleep 2; cp /tmp/omlx_moe_stats.json "$B/t9_b1_stats1.json" 2>/dev/null || true
grep -m1 "wrapped 48 layers" "$LOG" || true
restart   # restore: default env, merged file unchanged
echo "restored; arm blob: $B/t9_batch1_C.json"
