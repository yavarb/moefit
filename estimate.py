#!/usr/bin/env python3
"""estimate.py — will a MoE model run well on THIS Mac with expert paging?

Reads a HuggingFace-format model directory, classifies tensors into
routed experts vs the resident floor, reads this machine's chip/RAM,
and prints a concrete go/no-go with a capacity plan.

Zero dependencies beyond stdlib (parses the safetensors header itself).

usage: python3 estimate.py <model-dir> [--expert-tokens 10] [--json]
"""
import argparse, json, os, platform, re, struct, sys

DT_SIZE = {"F8_E4M3": 1, "F8_E5M2": 1, "BF16": 2, "F16": 2, "F32": 4,
           "F64": 8, "I8": 1, "U8": 1, "I16": 2, "U16": 2, "I32": 4,
           "U32": 4, "I64": 8, "U64": 8, "BOOL": 1}


def tensor_nbytes(meta):
    """Byte length of one tensor from its safetensors header entry.

    Prefer data_offsets (exact for every dtype, including the U32-packed
    weights of 4-bit MLX checkpoints); fall back to shape x dtype size.
    """
    offs = meta.get("data_offsets")
    if offs and len(offs) == 2 and offs[1] >= offs[0]:
        return int(offs[1]) - int(offs[0])
    numel = 1
    for d in meta["shape"]:
        numel *= d
    dt = meta["dtype"].upper()
    if dt not in DT_SIZE:
        raise SystemExit(f"unknown safetensors dtype {dt!r} and no "
                         f"data_offsets in header")
    return numel * DT_SIZE[dt]


def st_index(model_dir):
    idx = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(idx):
        with open(idx) as f:
            return json.load(f)["weight_map"]
    single = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(single):
        # read header for tensor names/sizes
        with open(single, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        return {k: "model.safetensors" for k in hdr if k != "__metadata__"}
    raise SystemExit(f"no safetensors found in {model_dir}")


def tensor_sizes(model_dir):
    wm = st_index(model_dir)
    files = sorted(set(wm.values()))
    out = {}
    for fn in files:
        path = os.path.join(model_dir, fn)
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        for k, meta in hdr.items():
            if k == "__metadata__":
                continue
            out[k] = tensor_nbytes(meta)
    return out


ROUTED_PAT = re.compile(
    r"(?:^|\.)\d+\.mlp\.(?:experts|switch_mlp|mlps)\.|experts\.\d+\.")


def classify(sizes):
    cats = {"routed": 0, "shared": 0, "ple": 0, "attn_floor": 0,
            "embed_head": 0, "other": 0}
    routed_per_layer = {}
    for k, v in sizes.items():
        if ROUTED_PAT.search(k) or "switch_mlp" in k:
            cats["routed"] += v
            m = re.search(r"layers\.(\d+)\.", k)
            if m:
                routed_per_layer.setdefault(int(m.group(1)), 0)
                routed_per_layer[int(m.group(1))] += v
        elif "ngram" in k or ".ple." in k or "ple_embedding" in k:
            cats["ple"] += v
        elif "shared_expert" in k:
            cats["shared"] += v
        elif any(t in k for t in ("embed_tokens", "lm_head", "embedding")):
            cats["embed_head"] += v
        elif ".mlp." in k or ".mlp_" in k:
            cats["other"] += v
        else:
            cats["attn_floor"] += v
    return cats, routed_per_layer


def machine():
    ram = None
    try:
        with open("/proc/meminfo") as f:  # non-mac fallback
            for line in f:
                if line.startswith("MemTotal"):
                    ram = int(line.split()[1]) * 1024
                    break
    except OSError:
        pass
    if ram is None and platform.system() == "Darwin":
        import subprocess
        try:
            ram = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]))
        except Exception:
            pass
    chip = platform.machine()
    if platform.system() == "Darwin":
        import subprocess
        try:
            out = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True).strip()
            if out:
                chip = out
        except Exception:
            pass
    return chip, ram


# measured decode DRAM demand: (floor + topk*expert) bytes/token at
# ~38% of peak DRAM BW sustained by MoE decode
DRAM_EFF = 0.385
PEAK_BW = {  # GB/s, approximate Apple SoC peaks
    "M4": 120, "M4 Pro": 273, "M4 Max": 546, "M3 Max": 400,
    "M2 Max": 400, "M1 Max": 400, "M3 Ultra": 800, "M2 Ultra": 800,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--expert-tokens", type=int, default=10)
    ap.add_argument("--os-reserve-gib", type=float, default=12.0)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    sizes = tensor_sizes(a.model_dir)
    cats, per_layer = classify(sizes)
    n_layers = max(per_layer) + 1 if per_layer else 0
    total = sum(sizes.values())
    chip, ram = machine()
    gib = 2**30
    usable = (ram / gib - a.os_reserve_gib) if ram else None
    bw = max((v for k, v in PEAK_BW.items() if k in chip), default=120)
    floor = (cats["shared"] + cats["attn_floor"] + cats["embed_head"]
             + cats["other"])
    experts_total = cats["routed"]
    # experts per layer avg, bytes/expert guess: routed/(layers*512)
    cfg = os.path.join(a.model_dir, "config.json")
    n_experts = 512
    if os.path.exists(cfg):
        c = json.load(open(cfg))
        tc = c.get("text_config", c)
        n_experts = tc.get("num_experts", n_experts)
    per_expert = experts_total / max(n_layers * n_experts, 1)
    tok_demand = floor + a.expert_tokens * n_layers * per_expert
    bw_need = tok_demand / gib * DRAM_EFF ** -1  # rough, bytes/token->GB/s at 1 tps

    out = dict(
        chip=chip, ram_gib=round((ram or 0) / gib, 1),
        usable_gib=round(usable, 1) if usable else None,
        model_total_gib=round(total / gib, 1),
        routed_experts_gib=round(experts_total / gib, 1),
        ple_table_gib=round(cats["ple"] / gib, 1),
        resident_floor_gib=round(floor / gib, 1),
        layers=n_layers, experts_per_layer=n_experts,
        expert_mib=round(per_expert / 2**20, 2),
        peak_dram_gbps=bw,
    )
    if usable is None:
        out["verdict"] = "UNKNOWN (could not read RAM)"
    elif floor / gib > usable:
        out["verdict"] = "NO-GO: non-expert floor exceeds usable RAM"
    else:
        cap_experts = int((usable - floor / gib) * gib / max(per_expert, 1))
        cap_per_layer = cap_experts // max(n_layers, 1)
        frac = min(1.0, cap_per_layer / max(n_experts, 1))
        out["resident_experts_per_layer"] = cap_per_layer
        out["expert_coverage_frac"] = round(frac, 3)
        if frac >= 0.999:
            out["verdict"] = ("FULL: model fits resident — plain loading, "
                              "no paging needed")
        else:
            tps_est = (bw * DRAM_EFF) / (tok_demand / gib)
            out["decode_ceiling_tps_bound"] = round(tps_est, 1)
            out["verdict"] = (
                "PAGING: keep %d experts/layer resident; predictions and "
                "prefix traces cover the rest from SSD. Decode is DRAM-bound"
                " at ~%s tok/s ceiling regardless of paging policy."
                % (cap_per_layer, round(tps_est, 1)))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
