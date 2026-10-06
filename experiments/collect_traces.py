"""Collect real router traces from Qwen3.8-Flash-Next on this Mac.

usage:
  .venv/bin/python experiments/collect_traces.py --split build
  .venv/bin/python experiments/collect_traces.py --split holdout

Greedy decode with the RouterTrace installed. Writes
results/traces/<split>.npz: per prompt the full token id sequence
(context+generation) and per-layer router top-k idx/score rows in token
order (prefill rows then one per generated token).
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

# MLX default wired pool is ~75% of unified memory; a 99 GB checkpoint
# needs most of a 128 GB Mac's pool wired.
os.environ.setdefault("MLX_MAX_WIRED_LIMIT", str(int(115e9)))

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from specexp.loader import load, DEFAULT_MODEL
from specexp.trace import RouterTrace
from specexp.ple_trace import PLETrace

CORPUS = Path(__file__).resolve().parents[1] / "data" / "prompts.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model-dir", default=str(DEFAULT_MODEL))
    ap.add_argument("--ple", action="store_true",
                    help="also capture PLE injection + ngram embedding rows")
    a = ap.parse_args()

    prompts = [json.loads(l) for l in CORPUS.read_text().splitlines() if l.strip()]
    prompts = [p for p in prompts if p["split"] == a.split]
    if a.limit:
        prompts = prompts[: a.limit]

    t0 = time.time()
    L = load(a.model_dir)
    print(f"loaded in {time.time()-t0:.0f}s; layers={L.num_layers} "
          f"expert_layers={len(L.expert_layers)} ple={L.ple_layers}", flush=True)

    import mlx.core as mx
    from mlx_vlm.generate import stream_generate

    tok = L.tokenizer
    use_chat = hasattr(tok, "apply_chat_template")

    outdir = Path(__file__).resolve().parents[1] / "results" / "traces"
    outdir.mkdir(parents=True, exist_ok=True)
    tr = RouterTrace(L.model, L.expert_layers)
    tr.install()
    pt = PLETrace(L.model) if a.ple else None
    if pt:
        pt.install()
    all_recs = {}
    try:
        for pi, item in enumerate(prompts):
            prompt = item["text"]
            if use_chat:
                try:
                    render = tok.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        add_generation_prompt=True, tokenize=False)
                except TypeError:
                    render = tok.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        add_generation_prompt=True)
            else:
                render = prompt
            tr.reset()
            tr.enabled = True
            if pt:
                pt.reset()
                pt.enabled = True
            gen_ids = []
            try:
                for resp in stream_generate(L.model, L.processor, render,
                                            max_tokens=a.max_tokens,
                                            temperature=0.0):
                    gen_ids.append(resp.token)
            finally:
                tr.enabled = False
                if pt:
                    pt.enabled = False
            ctx_ids = list(tok.encode(render, add_special_tokens=False))
            full_ids = ctx_ids + list(gen_ids)
            rec = {"ids": np.array(full_ids, dtype=np.int64)}
            for li, (idx, sc) in tr.stacked().items():
                # sanity: rows must equal tokens seen by the model
                rec[f"L{li}_idx"] = idx
                rec[f"L{li}_score"] = sc
            if pt:
                for name, arr in pt.stacked().items():
                    rec[f"PLE_{name}"] = arr
            tag = f"p{pi}_{item['id']}"
            all_recs.update({f"{tag}|{k}": v for k, v in rec.items()})
            n_rows = int(rec[f"L{L.expert_layers[0]}_idx"].shape[0]) if f"L{L.expert_layers[0]}_idx" in rec else -1
            print(f"[{pi+1}/{len(prompts)}] {item['id']}: ctx={len(ctx_ids)} "
                  f"gen={len(gen_ids)} rows={n_rows}", flush=True)
            mx.clear_cache()
    finally:
        tr.remove()

    out = outdir / f"{a.split}.npz"
    np.savez_compressed(out, **all_recs)
    print("wrote", out)


if __name__ == "__main__":
    main()
