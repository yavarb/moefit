# T3 LIP admission patch — exact diff for installed oMLX 0.7.0 (md5 d420b305)

Applies to /opt/homebrew/Cellar/omlx/0.7.0/libexec/lib/python3.11/site-packages/
omlx/patches/moe_expert_offload.py (the file pid 96103 loads — same md5 T9/T4/
design_inventor verified). Env-gated: stock behavior bit-for-bit unless
OMLX_ADMISSION=1 is set at process start. THREE hunks; the install choke
point is `_install`, used by BOTH the serial fallback and the windowed
read-ahead path, so one edit covers both arms of the DB toggle.

Pre-registration: results/T3_ADMISSION_SILICON_PREREG.md (prediction
-0.05..+0.25 tps on the 15.55 baseline; WASH expected; decision rules fixed).
Apply discipline: backup file + md5 + py_compile before restart (per
T4_DBUF_SILICON_PATCH.md).

## Hunk 1 — __init__ (after `self.hits = self.misses = 0`, line ~410)

```python
        self.hits = self.misses = 0
        self._admit = os.environ.get("OMLX_ADMISSION", "0") == "1"
        self._miss_hist = {}  # expert id -> lifetime miss count (LIP arm)
```

## Hunk 2 — _ensure_ids miss branch (after `self.misses += 1`, line ~490)

```python
                self.misses += 1
                if self._admit:
                    self._miss_hist[e] = self._miss_hist.get(e, 0) + 1
```

## Hunk 3 — _install tail (after `self.warm = ...`, before `return slot`)

Current code:

```python
        self.slot_of[e] = slot
        self.map[e] = slot
        self.warm = len(self.slot_of) == self.n_experts
        return slot
```

Patched:

```python
        self.slot_of[e] = slot
        self.map[e] = slot
        if (
            self._admit
            and self._miss_hist.get(e, 0) == 1
            and not self.free
        ):
            # LIP insertion (Qureshi ISCA'07): a FIRST-lifetime miss still
            # installs (bytes identical, map identical) but is placed at the
            # LRU position — the next eviction victim — instead of MRU.
            # One-shot pollution is evicted first; the resident core and
            # recurrent misses are protected. A hit or recurrent miss
            # re-inserts MRU via the existing hit path. Cache-fullness
            # bypass: while free slots remain, insert MRU as stock (warm).
            slot_val = self.slot_of.pop(e)
            rest = list(self.slot_of.items())
            self.slot_of.clear()
            self.slot_of[e] = slot_val
            self.slot_of.update(rest)
        self.warm = len(self.slot_of) == self.n_experts
        return slot
```

## Semantics notes

- `slot_of` is a plain dict whose INSERTION ORDER is the LRU order (hits
  re-insert via pop+assign; victim = next(iter(slot_of))). The front-insert
  rebuild is O(cap)=143 dict ops, only on first-lifetime misses — negligible
  vs 0.52 ms/expert IO.
- Everything else (reads, slot writes, map, counters, windowed read-ahead,
  decode-overlap path) is untouched; installs carry byte-identical payloads,
  so the T4 bit-exactness gate applies unchanged.
- Composes with the DB env arms: LIP is orthogonal to IO_WORKERS. If arms
  must be isolated, run LIP on the RESTORE restart (default env + LIP ON)
  as a fourth A/B/A pair: stock(default env) vs default env + OMLX_ADMISSION=1.
- Restore-after: unset OMLX_ADMISSION (or leave OFF) — default is stock.

## Scoring

Same-prompt A/B/A x3, n>=1024, collector blobs (runs[*].tokens,
finish_reason, stream_integrity), fail-closed scorer gates. Verdict per the
prereg decision rules: |d| < 0.30 tps = WASH (expected; the sim gain is a
multi-prompt steady-state effect the single-prompt bench cannot resolve);
>= +0.30 = TRANSFERS; ON < OFF by > 0.30 = falsifies single-prompt transfer.
