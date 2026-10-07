#!/usr/bin/env python3
"""configure_omlx_paging.py - one-shot oMLX expert-paging setup from estimate.py.

Reads a Hugging Face-format MLX checkpoint directory, runs estimate.py on it,
turns the PAGING verdict into an oMLX resident fraction, links the checkpoint
into oMLX's model directory, and writes the per-model settings oMLX needs to
stream experts from SSD (``moe_expert_offload_enabled``) and keep the Qwen4-Exp
PLE n-gram table on SSD (``qwen4_ple_ssd_offload``).

Stdlib only. Never deletes anything: an existing model entry is updated in
place, other models' settings are kept, and a conflicting path under
``~/.omlx/models`` is reported instead of replaced.

usage:
    python3 scripts/configure_omlx_paging.py --model-dir ~/models/<ckpt> \
        [--model-id <served id>] [--omlx-base ~/.omlx] [--fraction F] \
        [--mtp] [--save-estimate results/estimate_<host>.json] [--dry-run]

Exit codes: 0 configured (or FULL: nothing to do), 2 NO-GO or checkpoint
incomplete, 3 conflicting path under the oMLX model dir.
"""
import argparse, json, os, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GIB = 2 ** 30
# oMLX sizes a resident load as checkpoint bytes x 1.05.
OMLX_OVERHEAD = 1.05
# Leave this share of the Metal working-set cap for KV cache, prefill
# buffers and the streaming slots. The rest may hold resident experts.
METAL_CAP_SHARE = 0.85
# A live oMLX ceiling (from a 507 "does not fit under the dynamic memory
# ceiling (N GB)" refusal) already excludes the tier reserve; its soft
# watermark is 95%, so keep a little under that.
LIVE_CEILING_SHARE = 0.92


def check_complete(model_dir: Path):
    """Every shard named by the index must exist; no .incomplete leftovers."""
    cfg = model_dir / "config.json"
    idx = model_dir / "model.safetensors.index.json"
    single = model_dir / "model.safetensors"
    if not cfg.exists():
        return ["config.json missing"]
    problems = []
    if idx.exists():
        wm = json.load(open(idx))["weight_map"]
        for fn in sorted(set(wm.values())):
            if not (model_dir / fn).exists():
                problems.append(f"shard missing: {fn}")
    elif not single.exists():
        problems.append("no model.safetensors(.index.json) yet")
    partial = list((model_dir / ".cache").rglob("*.incomplete"))
    if partial:
        problems.append(f"{len(partial)} .incomplete download file(s) under .cache")
    return problems


def run_estimate(model_dir: Path) -> dict:
    out = subprocess.run([sys.executable, str(ROOT / "estimate.py"), str(model_dir)],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"estimate.py failed:\n{out.stderr}")
    return json.loads(out.stdout)


def omlx_python():
    """Interpreter inside the oMLX install (brew keg), or None."""
    exe = shutil.which("omlx") or "/opt/homebrew/bin/omlx"
    try:
        real = Path(exe).resolve()
    except OSError:
        return None
    for cand in (real.parent / "python", real.parents[1] / "libexec" / "bin" / "python",
                 Path("/opt/homebrew/opt/omlx/libexec/bin/python")):
        if cand.exists():
            return str(cand)
    return None


def metal_cap_gib():
    """Apple's max_recommended_working_set_size: oMLX never plans above it."""
    py = omlx_python()
    if not py:
        return None
    try:
        out = subprocess.run(
            [py, "-c", "import mlx.core as mx;"
             "print(mx.device_info()['max_recommended_working_set_size'])"],
            capture_output=True, text=True, timeout=60)
        return int(out.stdout.strip()) / GIB
    except Exception:
        return None


def choose_fraction(est: dict, cap_gib, live_ceiling_gb=None):
    """Resident fraction from estimate.py, then clamped so the resident load
    (floor + resident experts, x1.05) fits under the Metal cap budget and,
    when given, under the live oMLX ceiling quoted by a 507 refusal."""
    n = est["experts_per_layer"]
    frac = est["resident_experts_per_layer"] / n
    reason = "estimate.py resident_experts_per_layer / experts_per_layer"
    floor = est["resident_floor_gib"] * OMLX_OVERHEAD
    routed = est["routed_experts_gib"] * OMLX_OVERHEAD
    limits = []
    if cap_gib:
        limits.append((cap_gib * METAL_CAP_SHARE,
                       f"{METAL_CAP_SHARE:.0%} of the {cap_gib:.1f} GiB Metal working-set cap"))
    if live_ceiling_gb:
        gib = live_ceiling_gb * 1e9 / GIB
        limits.append((gib * LIVE_CEILING_SHARE,
                       f"{LIVE_CEILING_SHARE:.0%} of the live {live_ceiling_gb:.1f} GB oMLX ceiling"))
    for budget, why in limits:
        by_cap = max(0.0, (budget - floor) / routed) if routed else 1.0
        if by_cap < frac:
            frac = by_cap
            reason = "clamped to " + why
    minimum = 8 / n  # oMLX keeps at least 8 experts (or top-k) per layer
    frac = max(minimum, int(frac * 100) / 100)  # floor to 2 decimals
    return round(min(frac, 1.0), 2), reason


def link_model(models_dir: Path, model_id: str, model_dir: Path, dry):
    link = models_dir / model_id
    if link.is_symlink() or link.exists():
        try:
            same = link.resolve() == model_dir.resolve()
        except OSError:
            same = False
        if same:
            return f"already linked: {link} -> {model_dir}"
        raise SystemExit(f"CONFLICT: {link} exists and does not point at {model_dir}; "
                         "pick another --model-id or move it yourself (nothing deleted)")
    if not dry:
        models_dir.mkdir(parents=True, exist_ok=True)
        os.symlink(model_dir, link)
    return f"linked {link} -> {model_dir}"


def write_settings(base: Path, model_id: str, updates: dict, dry):
    path = base / "model_settings.json"
    data = {"version": 1, "models": {}}
    if path.exists():
        data = json.load(open(path))
        data.setdefault("models", {})
    entry = dict(data["models"].get(model_id, {}))
    entry.update(updates)
    data["models"][model_id] = entry
    if not dry:
        base.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        tmp.replace(path)
    return path, entry


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--model-id", help="served id (default: checkpoint dir name)")
    ap.add_argument("--omlx-base", default=os.path.expanduser("~/.omlx"))
    ap.add_argument("--fraction", type=float, help="override resident fraction (0,1]")
    ap.add_argument("--ceiling-gb", type=float,
                    help="live oMLX memory ceiling in GB, from a 507 'does not fit "
                         "under the dynamic memory ceiling (N GB)' refusal; the "
                         "resident set is shrunk to fit it")
    ap.add_argument("--mtp", action="store_true",
                    help="keep oMLX Lightning MTP on (qwen4_exp allows it with "
                         "offload; the draft head's experts stay resident)")
    ap.add_argument("--save-estimate", help="write estimate.py JSON here")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="configure even if shards are still downloading")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    model_dir = Path(a.model_dir).expanduser().resolve()
    model_id = a.model_id or model_dir.name
    base = Path(a.omlx_base).expanduser()

    problems = check_complete(model_dir)
    if problems and not a.allow_incomplete:
        print("checkpoint not complete:\n  " + "\n  ".join(problems))
        sys.exit(2)

    est = run_estimate(model_dir)
    print(json.dumps(est, indent=1))
    if a.save_estimate and not a.dry_run:
        Path(a.save_estimate).parent.mkdir(parents=True, exist_ok=True)
        with open(a.save_estimate, "w") as f:
            json.dump(est, f, indent=1)
        print(f"estimate saved to {a.save_estimate}")

    verdict = est["verdict"]
    if verdict.startswith("NO-GO"):
        print(verdict)
        sys.exit(2)
    if verdict.startswith("FULL"):
        print("FULL: the model fits resident; load it normally, no paging settings written")
        sys.exit(0)

    cap = metal_cap_gib()
    if a.fraction:
        frac, why = a.fraction, "--fraction override"
    else:
        frac, why = choose_fraction(est, cap, a.ceiling_gb)
    cfg = json.load(open(model_dir / "config.json"))
    mtype = str(cfg.get("model_type") or cfg.get("text_config", {}).get("model_type") or "")
    is_qwen4 = mtype.lower().startswith("qwen4_exp")

    updates = {
        "moe_expert_offload_enabled": True,
        "moe_expert_offload_resident_fraction": frac,
        "mtp_enabled": bool(a.mtp),
    }
    if is_qwen4:
        updates["qwen4_ple_ssd_offload"] = True

    resident_gib = (est["resident_floor_gib"] + frac * est["routed_experts_gib"]) * OMLX_OVERHEAD
    print(f"\nresident fraction {frac} ({why})")
    print(f"  ~{frac * est['experts_per_layer']:.0f} of {est['experts_per_layer']} experts/layer in RAM, "
          f"resident load ~{resident_gib:.1f} GiB (x{OMLX_OVERHEAD} oMLX margin)")
    if cap:
        print(f"  Metal working-set cap {cap:.1f} GiB; budget {cap * METAL_CAP_SHARE:.1f} GiB")
    if is_qwen4:
        print(f"  PLE table {est['ple_table_gib']} GiB stays on SSD (qwen4_ple_ssd_offload)")

    print(link_model(base / "models", model_id, model_dir, a.dry_run))
    path, entry = write_settings(base, model_id, updates, a.dry_run)
    print(f"{'would write' if a.dry_run else 'wrote'} {path} [{model_id}]:")
    print(json.dumps(entry, indent=2))
    print("\nnext:\n  scripts/serve_paging.sh            # or:\n  omlx serve --model-dir "
          f"{base / 'models'} --port 8000 --max-concurrent-requests 1 --memory-guard aggressive")


if __name__ == "__main__":
    main()
