"""Model loading for the Qwen3.8-Flash-Next-oQ4e checkpoint.

Drops the vision tower and (optionally) the MTP head to fit the language
model inside the Metal working-set on a 128 GB unified-memory Mac.
"""
from __future__ import annotations
import os
os.environ.setdefault("MLX_MAX_WIRED_LIMIT", str(int(115e9)))

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL = Path.home() / ".lmstudio/models/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp"


@dataclass
class Loaded:
    model: object
    tokenizer: object
    config: dict
    text_config: dict
    num_layers: int
    expert_layers: list[int]
    ple_layers: list[int]
    processor: object = None


def load(model_dir: str | os.PathLike = DEFAULT_MODEL,
         drop_vision: bool = True,
         drop_mtp: bool = True) -> Loaded:
    model_dir = Path(model_dir)
    cfg = json.loads((model_dir / "config.json").read_text())
    tcfg = cfg.get("text_config", cfg)
    num_layers = tcfg["num_hidden_layers"]

    # PLE layers as present in the checkpoint (0-based indices).
    idx = json.loads((model_dir / "model.safetensors.index.json").read_text())
    ple = sorted({int(m.group(1)) for k in idx["weight_map"]
                  if (m := re.search(r"layers\.(\d+)\.ple\.", k))})

    # qwen4_exp lives in mlx_vlm (as in oMLX production), not PyPI mlx_lm.
    # The checkpoint is 4-bit-affine; oMLX vendors a qwen4_exp build that
    # reads it directly (PyPI mlx-vlm expects FP8-form PLE shards). We use
    # the same vendored tree (Apache-2.0, from oMLX) via package-path
    # injection, and drop config.model_file for that type — same two
    # behaviors as oMLX apply_mlx_vlm_qwen4_exp_compat_patch.
    import sys, os
    from pathlib import Path as _P
    # oMLX runtime deps for the vendored arch (imported last-resort; this
    # venv keeps priority). Override with MOEFIT_OMLX_SITE=... if it moves.
    omlx_site = os.environ.get(
        "MOEFIT_OMLX_SITE",
        "~/.hermes/cache/scratch/omlxenv/lib/python3.13/site-packages")
    if Path(omlx_site).exists() and omlx_site not in sys.path:
        sys.path.append(omlx_site)
    vendor = _P(__file__).resolve().parents[1] / "vendor_mlx_vlm"
    if str(vendor) not in sys.path:
        import mlx_vlm, mlx_vlm.models
        mlx_vlm.__path__.insert(0, str(vendor))
        mlx_vlm.models.__path__.insert(0, str(vendor / "models"))
        import mlx_vlm.models.qwen4_exp  # noqa: F401
        import mlx_vlm.utils as _u
        if not getattr(_u.get_model_and_args, "_moefit_bypass", False):
            _orig_gma = _u.get_model_and_args
            def _gma(config, model_path=None):
                if str(config.get("model_type", "")).lower() == "qwen4_exp":
                    config = {k: v for k, v in config.items() if k != "model_file"}
                    return _orig_gma(config, model_path=None)
                return _orig_gma(config, model_path=model_path)
            _gma._moefit_bypass = True
            _u.get_model_and_args = _gma
    from mlx_vlm import load as mlx_load
    model, proc = mlx_load(str(model_dir))

    # Strip tensors that were loaded but are unused for text-only decode.
    if drop_vision:
        _strip(model, lambda p: p.startswith("vision_tower") or ".visual" in p)
    if drop_mtp:
        _strip(model, lambda p: p.startswith("mtp"))

    from mlx.nn import Module
    import mlx.core as mx
    mx.eval(model.parameters())
    gc_ok = _gc()

    sparse = tcfg.get("decoder_sparse_step", 1)
    mlp_only = set(tcfg.get("mlp_only_layers", []))
    expert_layers = [i for i in range(num_layers)
                     if i not in mlp_only and (i + 1) % sparse == 0]

    tok = proc.tokenizer if hasattr(proc, "tokenizer") else proc
    return Loaded(model=model, tokenizer=tok, processor=proc,
                  config=cfg, text_config=tcfg,
                  num_layers=num_layers, expert_layers=expert_layers,
                  ple_layers=ple)


def _strip(model, pred):
    """Detach submodules whose attribute path matches pred (MLX-native:
    children are plain instance attributes; setting to None frees them)."""
    from mlx.nn import Module
    removed = []

    def walk(mod, prefix=""):
        for name in list(vars(mod)):
            child = getattr(mod, name, None)
            if not isinstance(child, Module):
                continue
            path = f"{prefix}{name}"
            if pred(path):
                setattr(mod, name, None)
                removed.append(path)
            else:
                walk(child, f"{path}.")
        # ModuleList children: index-based names
        for name in list(vars(mod)):
            child = getattr(mod, name, None)
            if isinstance(child, (list, tuple)):
                for i, sub in enumerate(list(child)):
                    if isinstance(sub, Module):
                        if pred(f"{name}.{i}"):
                            child[i] = None
                            removed.append(f"{name}.{i}")
                        else:
                            walk(sub, f"{name}.{i}.")

    walk(model)
    return removed


def _gc():
    import gc
    gc.collect()
    try:
        import mlx.core as mx
        mx.clear_cache()
        mx.reset_peak_memory() if hasattr(mx, "reset_peak_memory") else mx.metal.reset_peak_memory()
    except Exception:
        pass
    return True
