#!/bin/bash
# M2c tiny-net correlator replay-OFF (F_L ablated) silicon A/B on M4 Max 36 GB / MBP.
# Decision (prereg t5_m2c_attribution_prereg.json + owner bar):
#   median Δtps >= +1.5  → KEEP correlator signal (falsifies instance-precision wash)
#   median Δtps <  +1.5  → DROP as exact-replay leak (GLM +32% caveat)
# ONE restart; EXIT restores d9850d99; SLOT_LOCK_T2; never kill non-omlx :8000.
set -euo pipefail
export PATH="/opt/homebrew/bin:$PATH"
OMLX_BASE="$HOME/.omlx"; PORT=8000
SERVE_LOG="$OMLX_BASE/logs/moefit-serve.log"; PIDFILE="$OMLX_BASE/moefit-serve.pid"
BENCH=/tmp/moefit-bench
LOCK=$BENCH/SLOT_LOCK_T2
REPO=/tmp/moefit-lab/repo
PATCHED=/opt/homebrew/Cellar/omlx/0.7.0/libexec/lib/python3.11/site-packages/omlx/patches/moe_expert_offload.py
# resolve patched path if Cellar layout differs
if [ ! -f "$PATCHED" ]; then
  PATCHED=$(python3 -c "import omlx.patches.moe_expert_offload as m; print(m.__file__)" 2>/dev/null || true)
fi
[ -n "${PATCHED:-}" ] && [ -f "$PATCHED" ] || { echo "cannot resolve moe_expert_offload.py"; exit 3; }
RESTORE=/tmp/omlx_t2_sidecar_plus_lip_v3.py
RESTORE_FALLBACK=/tmp/moefit-lab/repo/patches/omlx_t2_sidecar_plus_lip.py
RESTOREMD5=d9850d99ab0be7ee723fee72e360adbc
PATCH=$REPO/patches/omlx_t2_multi_router.py
PATCHMD5=980b20e02ad5468bdbf8c63780657c65
LOG=$BENCH/m2c_replay_off_ab.log
OUT=$BENCH/t2_nonrepeat_correlator_replay_off.json
OUT2=$BENCH/t2_nonrepeat_correlator_replay_off_onfirst.json
MAXTOK=${MAXTOK:-512}
HARNESS=$REPO/experiments/t2_nonrepeat_aba.py

mkdir -p "$BENCH" "$REPO/results/t2_silicon"
exec > >(tee -a "$LOG") 2>&1
echo "=== M2c replay-OFF A/B start $(date) ==="

locks=$(ls "$BENCH"/SLOT_LOCK_* 2>/dev/null || true)
if [ -n "$locks" ]; then echo "REFUSED: live slot lock(s): $locks"; exit 2; fi
[ -f "$PATCH" ] || { echo "missing patch $PATCH"; exit 3; }
[ "$(md5 -q "$PATCH")" = "$PATCHMD5" ] || { echo "patch md5 mismatch want=$PATCHMD5 got=$(md5 -q "$PATCH")"; exit 3; }
if [ ! -f "$RESTORE" ]; then
  [ -f "$RESTORE_FALLBACK" ] || { echo "missing restore $RESTORE and fallback"; exit 3; }
  cp "$RESTORE_FALLBACK" "$RESTORE"
  echo "staged restore from fallback -> $RESTORE"
fi
[ "$(md5 -q "$RESTORE")" = "$RESTOREMD5" ] || { echo "restore md5 mismatch"; exit 3; }

# Refuse if :8000 is a non-omlx listener (e.g. billing_platform) — no kills of foreign procs.
if lsof -tiTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1; then
  cmd=$(ps -p "$(lsof -tiTCP:$PORT -sTCP:LISTEN | head -1)" -o command= 2>/dev/null || true)
  case "$cmd" in
    *omlx*|*moe_expert*|*serve*) echo "port $PORT looks omlx-ish: $cmd" ;;
    *)
      echo "REFUSED: :$PORT held by non-omlx process — will not kill: $cmd"
      echo "Free the port (stop billing/other) or move that service, then re-run."
      exit 4
      ;;
  esac
fi

echo "lead_silicon/Opus M2c replay-OFF tiny-net correlator $(date -u +%Y-%m-%dT%H:%M:%SZ) pid=$$" > "$LOCK"
cp "$PATCH" "$BENCH/omlx_t2_multi_router.py"
cp "$HARNESS" "$BENCH/t2_nonrepeat_aba.py"

stop_omlx() {
  if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    kill "$(cat "$PIDFILE")" || true
    for i in $(seq 1 60); do kill -0 "$(cat "$PIDFILE")" 2>/dev/null || break; sleep 1; done
    kill -9 "$(cat "$PIDFILE")" 2>/dev/null || true; rm -f "$PIDFILE"
  fi
  # Only kill omlx listeners on PORT — never foreign apps.
  for pid in $(lsof -tiTCP:$PORT -sTCP:LISTEN 2>/dev/null || true); do
    cmd=$(ps -p "$pid" -o command= 2>/dev/null || true)
    case "$cmd" in
      *omlx*) kill "$pid" 2>/dev/null || true; sleep 2; kill -9 "$pid" 2>/dev/null || true ;;
      *) echo "leave non-omlx pid=$pid: $cmd" ;;
    esac
  done
}
wait_mem() {
  for i in $(seq 1 120); do
    f=$(memory_pressure -Q 2>/dev/null | grep -o 'free percentage: [0-9]*' | grep -o '[0-9]*' || true)
    if [ -n "${f:-}" ] && [ "$f" -ge 70 ]; then echo "mem ok (${f}%)"; return 0; fi
    sleep 2
  done
  echo "mem wait timeout"
}
start_omlx() {
  nohup env -u OMLX_ADMISSION "$@" omlx serve --model-dir "$OMLX_BASE/models" --port "$PORT" \
    --max-concurrent-requests 1 --memory-guard aggressive >"$SERVE_LOG" 2>&1 &
  echo $! >"$PIDFILE"
  echo "started pid=$(cat "$PIDFILE") env=$*"
  for i in $(seq 1 300); do
    curl -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { echo ready; return 0; }
    kill -0 "$(cat "$PIDFILE")" 2>/dev/null || { echo died; tail -20 "$SERVE_LOG"; return 1; }
    sleep 1
  done
  echo timeout; return 1
}
restore() {
  ec=$?
  echo "=== restore d9850d99 + default env (ec=$ec) $(date) ==="
  rm -f /tmp/omlx_sidecar_on /tmp/omlx_multi_router_on
  stop_omlx || true
  if [ -f "$RESTORE" ]; then
    cp "$RESTORE" "$PATCHED"
    echo "restored md5=$(md5 -q "$PATCHED") (want $RESTOREMD5)"
  fi
  wait_mem || true
  # only restart omlx if port free or already our pidfile path
  if ! lsof -tiTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1; then
    start_omlx || true
  else
    echo "skip post-restore start: :$PORT still held"
  fi
  echo "post-restore health: $(curl -sf -m3 http://127.0.0.1:$PORT/health || echo FAIL)"
  rm -f "$LOCK"
  # copy artifacts into repo results
  cp -f "$OUT" "$REPO/results/t2_silicon/t2_nonrepeat_correlator_replay_off.json" 2>/dev/null || true
  cp -f "$OUT2" "$REPO/results/t2_silicon/t2_nonrepeat_correlator_replay_off_onfirst.json" 2>/dev/null || true
  cp -f "$LOG" "$REPO/results/t2_silicon/m2c_replay_off_ab.log" 2>/dev/null || true
  echo "=== M2c replay-OFF A/B end $(date) lock cleared ==="
  exit $ec
}
trap restore EXIT

echo "=== install multi_router+$PATCHMD5 with MOEFIT_FL=0 (replay OFF) ==="
stop_omlx
cp "$PATCH" "$PATCHED"
[ "$(md5 -q "$PATCHED")" = "$PATCHMD5" ] || exit 5
wait_mem
# Correlator+Markov ON; exact-replay F_L DISABLED
start_omlx MOEFIT_CORRELATOR=1 MOEFIT_MULTI_ROUTER=1 MOEFIT_FL=0 \
  MOEFIT_PF_THETA=0.37 MOEFIT_CORR_THETA=0.55 MOEFIT_PF_CAP=8

echo "=== pre-warm ==="
python3 - <<'PYW' || echo "prewarm soft-fail"
import json, sys, urllib.request
url = "http://127.0.0.1:8000/v1/chat/completions"
body = json.dumps(dict(
    model="Qwen3.8-Flash-Next-oQ4e-mtp", stream=True, temperature=0, max_tokens=32,
    stream_options={"include_usage": True},
    messages=[{"role": "user", "content": "Say hello in one short sentence."}],
)).encode()
req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=1800) as r:
        n = sum(1 for _ in r)
    print(f"prewarm ok lines={n}")
except Exception as e:
    print(f"prewarm FAIL: {type(e).__name__}: {e}")
    sys.exit(1)
PYW
sleep 2
ls -la /tmp/omlx_moe_stats.json || echo "stats still missing"

echo "=== NON-repeat A/B replay-OFF max_tokens=$MAXTOK ==="
rm -f /tmp/omlx_sidecar_on
python3 "$HARNESS" --max-tokens "$MAXTOK" --out "$OUT"

# Leak-free attribution pass (lead_silicon): MOEFIT_FL=0 still leaves Markov1 (exact route-set
# transitions, trained even while the flag is OFF) and the online correlator learning from the
# OFF decode of the SAME greedy prompt. ON-first on 4 fresh holdout prompts removes that path:
# ON never follows a decode of its own prompt. Same process, no restart, flag toggle only.
echo "=== LEAK-FREE ON-first pass, holdout prompts, max_tokens=$MAXTOK ==="
python3 "$HARNESS" --max-tokens "$MAXTOK" --out "$OUT2" --on-first --prompt-set holdout --no-warm || echo "onfirst pass failed"

echo "=== verdict vs prereg / owner bar ==="
python3 - <<'PY'
import json
from pathlib import Path
p = Path("/tmp/moefit-bench/t2_nonrepeat_correlator_replay_off.json")
d = json.loads(p.read_text())
delta = d.get("median_delta_tps")
pct = d.get("median_pct")
hashes = d.get("all_hash_match")
print("summary", {k: d[k] for k in d if k not in ("runs", "pairs")})
# Owner bar: >=+1.5 tok/s with replay disabled else DROP as replay
if delta is None:
    verdict = "INCONCLUSIVE_NO_DELTA"
elif delta >= 1.5 and hashes:
    verdict = "KEEP_CORRELATOR_SIGNAL"
elif delta >= 1.5 and not hashes:
    verdict = "HASH_FAIL"
else:
    verdict = "DROP_AS_REPLAY"  # includes wash <1.5
print("OWNER_VERDICT", verdict, f"delta={delta} pct={pct} hashes={hashes}")
print("PREREG_NOTE: <=0.5 => replay-attributed; 0.5-1.5 wash; >=1.5 falsifies instance wash")
# write verdict sidecar
out = {
    "kind": "m2c_replay_off_ab_verdict",
    "owner_bar": ">=+1.5 tok/s bit-identical else DROP_AS_REPLAY",
    "median_delta_tps": delta,
    "median_pct": pct,
    "all_hash_match": hashes,
    "waste_rates": d.get("waste_rates"),
    "owner_verdict": verdict,
    "prereg": "results/t5_m2c_attribution_prereg.json",
    "env": "MOEFIT_FL=0 MOEFIT_CORRELATOR=1 MOEFIT_MULTI_ROUTER=1",
    "patch_md5": "980b20e02ad5468bdbf8c63780657c65",
}
Path("/tmp/moefit-bench/m2c_replay_off_verdict.json").write_text(json.dumps(out, indent=2)+"\n")
Path("/tmp/moefit-lab/repo/results/t2_silicon/m2c_replay_off_verdict.json").write_text(json.dumps(out, indent=2)+"\n")
print(json.dumps(out, indent=2))
PY
test -f "$OUT"
