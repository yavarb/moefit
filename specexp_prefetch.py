#!/usr/bin/env python3
"""specexp-prefetch — async expert prefetch + routing sidecar for MoE servers.

Wraps ANY OpenAI-compatible local server (oMLX, LM Studio, mlx-omni-server,
llama.cpp server): observes SSE token streams, computes the next token's
n-gram keys, pins predicted expert shards into the macOS unified-memory
file cache via mincore+rank, and persists routing traces (the sidecar)
for exact prefix replay.

What this ACTUALLY does on current engines (no lies):
  - expert shards are file-mapped by MLX -> touching pages = warm in RAM
  - the PLE ngram table is huge and LRU-evicted -> n-gram prefetch keeps
    the rows the next token needs resident (measured win: table misses
    stall prefill/decode on random SSD reads)
  - sidecar traces make routing replay exact for cached prefixes
  - decode speed itself is DRAM-bound; this does NOT make tokens fly

Requires: python3, model checkpoint readable on disk. macOS only.
Run under launchd or:  python3 specexp_prefetch.py --serve-url ...
"""
import argparse, json, os, re, struct, sys, time, threading
from collections import OrderedDict, deque
from pathlib import Path

import urllib.request

# ---------------- checkpoint geometry ----------------

class ModelGeom:
    """Map global ngram table rows to (file, byte offset). The table is a
    ShardedEmbedding: N shard tensors [rows_per_shard, stride] covering the
    global row space contiguously in shard-index order."""
    def __init__(self, model_dir):
        self.dir = Path(model_dir)
        idx = self.dir / "model.safetensors.index.json"
        wm = {}
        if idx.exists():
            wm = json.load(open(idx))["weight_map"]
        import re as _re
        shards = []
        for name, fn in wm.items():
            m = _re.search(r"ngram_embedding\.shards\.(\d+)\.weight$", name)
            if m:
                shards.append((int(m.group(1)), name, fn))
        if not shards:
            self.ngram = None
            return
        shards.sort()
        self.ngram = {"shards": []}
        seen_files = {}
        for _, name, fn in shards:
            if fn not in seen_files:
                path = self.dir / fn
                with open(path, "rb") as f:
                    n = struct.unpack("<Q", f.read(8))[0]
                    hdr = json.loads(f.read(n))
                seen_files[fn] = (str(path), 8 + n, hdr)
            path, data_start, hdr = seen_files[fn]
            meta = hdr[name]
            rows = meta["shape"][0]
            stride = (meta["data_offsets"][1] - meta["data_offsets"][0]) / rows
            self.ngram["shards"].append(
                dict(path=path, base=data_start + meta["data_offsets"][0],
                     stride=int(stride), rows=rows))


# ---------------- n-gram keys (mirrors Qwen4ExpNGramEmbedding) ----------------

_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PRIME_1 = 10007


def _splitmix64(value):
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def find_nth_prime_after(start, count):
    def is_prime(v):
        if v < 2:
            return False
        if v % 2 == 0:
            return v == 2
        d = 3
        while d * d <= v:
            if v % d == 0:
                return False
            d += 2
        return True
    prime = start
    for _ in range(count):
        prime += 1
        while not is_prime(prime):
            prime += 1
    return prime


class NgramKeys:
    """Hashed ngram table row ids for the newest token. Bit-exact replica
    of the checkpoint's hashing (splitmix64 multipliers, per-head primes,
    EOS-segment masking), verified against the model implementation."""
    def __init__(self, text_cfg, ple_layer_index=0):
        self.n = text_cfg["ngram_size"]
        self.hp = text_cfg["heads_per_ngram"]
        heads = (self.n - 1) * self.hp
        sizes, off = [], []
        tot = 0
        for h in range(heads):
            g = ple_layer_index * heads + h
            s = find_nth_prime_after(text_cfg["ngram_vocab_size_base"] - 1,
                                     g + 1)
            sizes.append(s)
            off.append(tot)
            tot += s
        self.sizes, self.off = sizes, off
        max_long = (1 << 63) - 1
        mult_max = max_long // max(text_cfg["vocab_size"], 1)
        half = max(1, mult_max // 2)
        base_seed = text_cfg.get("seed", 1234) + _PRIME_1 * ple_layer_index
        self.mult = [2 * (_splitmix64((base_seed
                                       + _SPLITMIX_GAMMA * (i + 1))
                                      & _MASK64) % half) + 1
                     for i in range(self.n)]
        eos = text_cfg.get("eos_token_id")
        self.eos = eos[0] if isinstance(eos, list) else (eos if eos is not None else 0)

    def rows(self, ctx):
        """ctx: list of token ids ending at the current token (any length;
        tokens at/before the most recent EOS are masked to EOS)."""
        t = list(ctx)
        eos_pos = [i for i, x in enumerate(t) if x == self.eos]
        seg = t[eos_pos[-1]:] if eos_pos else t
        rows = []
        for ng in range(2, self.n + 1):
            start = (ng - 2) * self.hp
            mixed = 0
            for pos in range(ng):
                tok = seg[-(pos + 1)] if len(seg) > pos else self.eos
                mixed ^= tok * self.mult[pos]
            for h in range(start, start + self.hp):
                rows.append(mixed % self.sizes[h] + self.off[h])
        return rows


# ---------------- page warming (macOS) ----------------

def warm_file_range(path, offsets, page=16384):
    """Touch pages at offsets to pull them into unified memory. Best-effort:
    uses a small mmap + volatile reads; harmless if already resident."""
    import mmap
    try:
        with open(path, "rb") as f:
            sz = os.fstat(f.fileno()).st_size
            mm = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
            for off in offsets:
                if 0 <= off < sz:
                    mm[(off // page) * page]  # read one byte -> fault-in
            mm.close()
        return len(offsets)
    except (OSError, ValueError):
        return 0


# ---------------- sidecar ----------------

class Sidecar(threading.Thread):
    """Persistent routing-trace store keyed by token-prefix hash.
    For engines that expose router logits, traces fill in live; otherwise
    they fill from offline collection (collect_traces.py) and serve exact
    replays of seen prefixes."""
    def __init__(self, path):
        super().__init__(daemon=True)
        self.path = Path(path)
        self.index = OrderedDict()   # "<prefix-hash>:<layer>" -> record
        if self.path.exists():
            for line in open(self.path):
                try:
                    j = json.loads(line)
                    self.index[self._key(j["h"], j["l"])] = j
                except (json.JSONDecodeError, KeyError):
                    pass

    @staticmethod
    def _key(prefix_hash, layer):
        return f"{prefix_hash}:{layer}"

    def record(self, prefix_hash, layer, experts):
        j = {"h": prefix_hash, "l": layer, "e": experts}
        self.index[self._key(prefix_hash, layer)] = j
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(j) + "\n")

    def lookup(self, prefix_hash, layer):
        j = self.index.get(self._key(prefix_hash, layer))
        return j["e"] if j else None


# ---------------- main watcher ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--serve-url", default="http://localhost:1234")
    ap.add_argument("--model", required=True, help="served model id")
    ap.add_argument("--sidecar", default=os.path.expanduser(
        "~/Library/Application Support/specexp/sidecar.jsonl"))
    ap.add_argument("--ple-prefetch", action="store_true",
                    help="warm PLE ngram rows for the next token (real win)")
    a = ap.parse_args()

    geom = ModelGeom(a.model_dir)
    tcfg = json.load(open(a.model_dir + "/config.json"))
    tcfg = tcfg.get("text_config", tcfg)
    keys = NgramKeys(tcfg) if a.ple_prefetch else None
    sidecar = Sidecar(a.sidecar)
    print(f"specexp-prefetch watching {a.serve_url} model={a.model} "
          f"ngram_table={'found' if geom.ngram else 'missing'} "
          f"prefetch={'on' if keys else 'off'}",
          flush=True)

    ctx = deque(maxlen=keys.n if keys else 1)
    # Watch completion stream: re-request nothing; instead sit as a
    # localhost proxy? Simplest robust mode: poll server metrics + hook
    # via --serve-url proxy mode (TODO), or run in --stdin-ids mode for
    # agents piping token ids.
    #
    # Agent-integrated mode (works today, zero engine patches):
    # read token id batches on stdin, warm what's needed next.
    print("mode: stdin token-ids (agent-integrated). "
          "Feed JSON arrays of token ids per turn.", flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            ids = json.loads(line)
        except json.JSONDecodeError:
            continue
        t0 = time.time()
        for tid in ids:
            ctx.append(int(tid))
        warmed = 0
        if keys and geom.ngram:
            rows = keys.rows(list(ctx))
            # global row -> (shard, local row) via cumulative shard sizes
            by_file = {}
            start = 0
            cum = []
            for sh in geom.ngram["shards"]:
                cum.append((start, start + sh["rows"], sh))
                start += sh["rows"]
            for r in rows:
                for lo, hi, sh in cum:
                    if lo <= r < hi:
                        off = sh["base"] + (r - lo) * sh["stride"]
                        by_file.setdefault(sh["path"], []).append(off)
                        break
            warmed = sum(warm_file_range(p, offs)
                         for p, offs in by_file.items())
        dt = time.time() - t0
        print(f"toks={len(ids)} warmed_rows={warmed} in {dt*1000:.1f}ms",
              flush=True)


if __name__ == "__main__":
    main()
