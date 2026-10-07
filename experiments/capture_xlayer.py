"""Capture per-layer MoE inputs + real routes during greedy decode (real model).

Purpose: evaluate CROSS-LAYER route prediction for a guarded L+1 expert
prefetch pipeline. For every decode token and every MoE layer L we store
  x_L    : the MoE block input (post attention/hyper-connection mix), fp16
  idx_L  : the real top-k expert ids, scores_L : renormalized gate probs
plus every layer's router weight W_L (bf16 -> fp16). Offline, the predictor
for layer L+d is topk(softmax(W_{L+d} @ x_L)) - computable the moment layer
L's MoE input exists, i.e. ~d layers before L+d's demand reads.

Decode positions only (T==1 calls) to bound memory; prefill is not stored.
Runs on a Mac that can hold the model (the 128 GB MBP); never on Santa Cruz.

usage: ~/.hermes/cache/scratch/omlxenv/bin/python3 experiments/capture_xlayer.py \
          --n-prompts 8 --max-tokens 160 --out ~/.hermes/cache/scratch/xlayer.npz
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from moefit.loader import load, DEFAULT_MODEL  # noqa: E402

CORPUS = Path(__file__).resolve().parents[1] / "data" / "prompts.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-prompts", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--split", default="holdout")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    prompts = [json.loads(l) for l in CORPUS.read_text().splitlines() if l.strip()]
    prompts = [p for p in prompts if p["split"] == a.split][: a.n_prompts]
    t0 = time.time()
    Ld = load(str(DEFAULT_MODEL))
    print(f"loaded {time.time()-t0:.0f}s expert_layers={len(Ld.expert_layers)}", flush=True)
    import mlx.core as mx
    from mlx_vlm.generate import stream_generate

    text = Ld.model.language_model.model
    blocks = {li: text.layers[li].mlp for li in Ld.expert_layers}
    ids = {id(b): li for li, b in blocks.items()}
    W = np.stack([np.asarray(blocks[li].gate.weight.astype(mx.float16))
                  for li in Ld.expert_layers])
    cls = type(next(iter(blocks.values())))
    orig = cls.__call__
    rec = {"on": False, "x": [], "idx": [], "sc": [], "layer": []}

    def patched(self_blk, x, *aa, **kw):
        li = ids.get(id(self_blk))
        if li is not None and rec["on"] and x.shape[-2] == 1 and x.ndim == 3:
            g = mx.softmax(self_blk.gate(x), axis=-1, precise=True)
            k = self_blk.top_k
            inds = mx.argpartition(g, kth=-k, axis=-1)[..., -k:]
            sc = mx.take_along_axis(g, inds, axis=-1)
            sc = sc / sc.sum(axis=-1, keepdims=True)
            xf = x.astype(mx.float16)
            mx.eval(inds, sc, xf)
            rec["x"].append(np.asarray(xf).reshape(-1))
            rec["idx"].append(np.asarray(inds).reshape(-1).astype(np.int16))
            rec["sc"].append(np.asarray(sc.astype(mx.float16)).reshape(-1))
            rec["layer"].append(li)
        return orig(self_blk, x, *aa, **kw)

    cls.__call__ = patched
    tok = Ld.tokenizer
    out = {"W": W, "expert_layers": np.array(Ld.expert_layers)}
    try:
        for pi, item in enumerate(prompts):
            render = tok.apply_chat_template([{"role": "user", "content": item["text"]}],
                                             add_generation_prompt=True, tokenize=False)
            for k in ("x", "idx", "sc", "layer"):
                rec[k].clear()
            rec["on"] = True
            t1 = time.time(); n = 0
            try:
                for resp in stream_generate(Ld.model, Ld.processor, render,
                                            max_tokens=a.max_tokens, temperature=0.0):
                    n += 1
            finally:
                rec["on"] = False
            lay = np.array(rec["layer"])
            nl = len(Ld.expert_layers)
            T = len(lay) // nl
            if T * nl != len(lay) or not (lay[:nl] == np.array(Ld.expert_layers)).all():
                print("WARN layer order mismatch", len(lay), flush=True)
            X = np.stack(rec["x"])[: T * nl].reshape(T, nl, -1)
            I = np.stack(rec["idx"])[: T * nl].reshape(T, nl, -1)
            S = np.stack(rec["sc"])[: T * nl].reshape(T, nl, -1)
            tag = f"p{pi}_{item['id']}"
            out[f"{tag}|x"] = X
            out[f"{tag}|idx"] = I
            out[f"{tag}|sc"] = S
            print(f"[{pi+1}/{len(prompts)}] {item['id']}: gen={n} decode_rows={T} "
                  f"x={X.shape} {time.time()-t1:.0f}s", flush=True)
            mx.clear_cache()
            np.savez(a.out, **out)   # incremental: survive external kills
    finally:
        cls.__call__ = orig
    np.savez(a.out, **out)
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
