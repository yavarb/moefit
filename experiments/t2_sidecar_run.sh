#!/bin/bash
# T2 sidecar A/B/A runner with slot lock (bash, not zsh: empty glob is fine).
B=/tmp/moefit-bench
shopt -s nullglob
for L in $B/SLOT_LOCK_*; do echo "held: $L"; cat "$L"; exit 2; done
echo "design_inventor T2 sidecar A/B/A pid $(pgrep -f omlx-server | tr '\n' ' ') $(date)" > $B/SLOT_LOCK_T2
trap 'rm -f $B/SLOT_LOCK_T2 /tmp/omlx_sidecar_on' EXIT
md5 -q /opt/homebrew/Cellar/omlx/0.7.0/libexec/lib/python3.11/site-packages/omlx/patches/moe_expert_offload.py
cd /tmp && python3 /tmp/t2_sidecar_aba.py --arms "${ARMS:-A0,A,B,A2,B2}" --out $B/t2_sidecar_aba.json
