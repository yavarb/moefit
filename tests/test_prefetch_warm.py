"""PLE row warming must cover every token of a multi-token turn and every
page a row spans.

Before: the stdin loop appended all ids, then warmed rows for the LAST
token only, so a 200-token prompt fed as one array warmed 16 rows instead
of 3200; and warm_file_range touched only the first page of each row, so a
row straddling a 16 KiB page boundary left its tail cold.

Uses a fake 2-shard n-gram table in a temp dir and a small text_config so
NgramKeys hashing runs unchanged. Stdlib only:
    python3 tests/test_prefetch_warm.py
"""
import json, os, struct, sys, tempfile
from collections import deque

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import specexp_prefetch as spf  # noqa: E402

CFG = dict(ngram_size=3, heads_per_ngram=8, ngram_vocab_size_base=1000,
           vocab_size=1000, eos_token_id=0, seed=1234)


def write_model(d, rows_per_shard, stride):
    hdr, off = {}, 0
    for i in range(2):
        name = f"model.language_model.ngram_embedding.shards.{i}.weight"
        hdr[name] = {"dtype": "F16", "shape": [rows_per_shard, stride // 2],
                     "data_offsets": [off, off + rows_per_shard * stride]}
        off += rows_per_shard * stride
    h = json.dumps(hdr).encode()
    with open(os.path.join(d, "model.safetensors"), "wb") as f:
        f.write(struct.pack("<Q", len(h))); f.write(h); f.write(b"\1" * off)
    json.dump({"text_config": CFG}, open(os.path.join(d, "config.json"), "w"))


def main():
    fails = 0
    keys = spf.NgramKeys(CFG)
    rows_per_token = (CFG["ngram_size"] - 1) * CFG["heads_per_ngram"]
    total_rows = sum(keys.sizes)
    with tempfile.TemporaryDirectory() as d:
        stride = 5000                           # straddles 16 KiB pages
        write_model(d, (total_rows + 1) // 2, stride)
        geom = spf.ModelGeom(d)
        if geom.ngram is None:
            print("FAIL: single-file model.safetensors not recognised"); fails += 1
            sys.exit(1)
        ids = [5, 17, 42, 42, 9, 0, 3, 8]
        ctx = deque(maxlen=keys.n)
        warmed = spf.warm_tokens(keys, geom, ctx, ids)
        want = rows_per_token * len(ids)
        print(f"fed {len(ids)} tokens: warmed_rows={warmed} expected={want}")
        if warmed != want:
            print("FAIL: not every token position was warmed"); fails += 1
        # determinism + range check
        r1 = keys.rows([5, 17, 42]); r2 = keys.rows([5, 17, 42])
        if r1 != r2 or len(r1) != rows_per_token:
            print("FAIL: rows not deterministic or wrong count"); fails += 1
        if not all(0 <= r < total_rows for r in r1):
            print("FAIL: row id outside table"); fails += 1
        # page coverage: a range crossing a page boundary touches both pages
        touched_pages = []
        import mmap
        class Spy:
            def __init__(self, inner): self.inner = inner
            def __getitem__(self, i): touched_pages.append(i); return self.inner[i]
            def close(self): self.inner.close()
        real_mmap = mmap.mmap
        mmap.mmap = lambda *a, **k: Spy(real_mmap(*a, **k))
        try:
            path = geom.ngram["shards"][0]["path"]
            spf.warm_file_range(path, [(16384 - 100, 5000)])
        finally:
            mmap.mmap = real_mmap
        print("pages touched for a straddling row:", sorted(set(touched_pages)))
        if sorted(set(touched_pages)) != [0, 16384]:
            print("FAIL: straddling row did not touch both pages"); fails += 1
    if fails:
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
