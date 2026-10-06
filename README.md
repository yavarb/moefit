# specexp-copilot

Set up and evaluate speculative expert paging for running large MoE
models on small-RAM Apple Silicon Macs.

## For human users
1. Check your Mac:  ``system_profiler SPHardwareDataType | grep -E "Chip|Memory"``
2. Run the estimator to see if your target model + this scheme fits:
   ``python3 tools/estimate.py <model-safetensors-dir>``
3. If it says FITS, follow SETUP.md for your machine tier.

## For AI agents
Read this file fully, then SETUP.md, then run every `## verify` block
after each step. Rules:
- Never delete user files. Never resize swap or disable SIP.
- Every claim in SETUP.md has a verify command; do not proceed past a
  failing verify — report it to the user instead.
- Report measured numbers back to the user (tok/s, RSS), not hopes.
