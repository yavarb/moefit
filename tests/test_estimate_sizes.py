"""estimate.py must size tensors from the safetensors header correctly.

Repro for the dtype bug: MLX 4-bit affine checkpoints store packed weights
as U32. The dtype table in estimate.py had no U32 entry and fell back to
2 bytes/element, so every packed weight tensor was counted at half its
real size. The header's data_offsets give the exact byte length, so the
test builds a real single-file safetensors with a U32 tensor and a BF16
tensor and compares estimate.tensor_sizes() to the on-disk byte lengths.

run: python3 tests/test_estimate_sizes.py
"""
import json, os, struct, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import estimate  # noqa: E402


def write_safetensors(path, tensors):
    """tensors: list of (name, dtype, shape, nbytes). Writes zeros."""
    hdr, off = {}, 0
    for name, dtype, shape, nbytes in tensors:
        hdr[name] = {"dtype": dtype, "shape": shape,
                     "data_offsets": [off, off + nbytes]}
        off += nbytes
    h = json.dumps(hdr).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        f.write(b"\0" * off)


def main():
    with tempfile.TemporaryDirectory() as d:
        # one routed-expert packed weight: [512, 1024, 64] u32 = 128 MiB
        # one bf16 scales tensor: [512, 1024, 8] bf16 = 8 MiB
        tensors = [
            ("model.layers.0.mlp.experts.gate_proj.weight", "U32",
             [512, 1024, 64], 512 * 1024 * 64 * 4),
            ("model.layers.0.mlp.experts.gate_proj.scales", "BF16",
             [512, 1024, 8], 512 * 1024 * 8 * 2),
            ("model.layers.0.self_attn.q_proj.weight", "F8_E4M3",
             [2048, 2048], 2048 * 2048),
        ]
        write_safetensors(os.path.join(d, "model.safetensors"), tensors)
        sizes = estimate.tensor_sizes(d)
        bad = []
        for name, dtype, shape, nbytes in tensors:
            got = sizes[name]
            flag = "OK " if got == nbytes else "BAD"
            if got != nbytes:
                bad.append(name)
            print(f"{flag} {name:50s} {dtype:7s} expected={nbytes:>12d} "
                  f"got={got:>12d}")
        total_expected = sum(t[3] for t in tensors)
        total_got = sum(sizes.values())
        print(f"total expected={total_expected} got={total_got} "
              f"ratio={total_got / total_expected:.3f}")
        if bad:
            print("FAIL: mis-sized tensors:", bad)
            sys.exit(1)
        print("PASS")


if __name__ == "__main__":
    main()
