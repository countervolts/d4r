#!/usr/bin/env python3
"""Experimental preset-L decoder: compute only its reconstruction head columns.

Copies a current compiled L model, appends unchanged fc2 columns in a 16-channel
layout and builds the specialized decoder. No floating-point weight conversion.
The destination must not exist. See docs/l-amd-redesign.md.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))
import l_model  # noqa: E402,F401
from test_m import defines  # noqa: E402

# Half 0: gates, features, padding. Half 1: mix, covariance, padding.
COLUMNS = [0, 1, 12, 4, 5, 8, 9, 0, 13, 16, 17, 20, 21, 24, 0, 0]


def specialize(src, dst, wpw, extra, half_input=False):
    if dst.exists():
        raise ValueError(f"destination already exists: {dst}")
    head = bytearray((src / "offsets.bin").read_bytes())
    if head[:8] not in (b"D4RL0001", b"D4RL0002"):
        raise ValueError("requires a compiled preset-L model")
    if (len(head) - 40) % 296:
        raise ValueError("incompatible L metadata")
    block = len(head) - 296
    base8, base16, *offsets = struct.unpack_from("<74I", head, block)
    if not all(offsets[32:36]):
        raise ValueError("model needs the current f32 bias tiles")
    weights = bytearray((src / "weights8.bin").read_bytes())
    if len(weights) != struct.unpack_from("<I", head, 8)[0]:
        raise ValueError("invalid weight size")
    for chunk in range(4):
        start = base8 + offsets[22 + 3 * chunk]
        raw = np.frombuffer(bytes(weights[start:start + 1024]), np.uint8)
        matrix = raw.reshape(2, 2, 16, 16).transpose(0, 3, 1, 2).reshape(32, 32)
        selected = matrix[:, COLUMNS]
        packed = selected.reshape(2, 16, 1, 16).transpose(0, 2, 3, 1).copy().ravel()
        decoded = packed.reshape(2, 1, 16, 16).transpose(0, 3, 1, 2).reshape(32, 16)
        assert np.array_equal(decoded, selected)
        weights.extend(bytes((-len(weights)) % 64))
        offsets[44 + chunk] = len(weights) - base8
        weights.extend(packed.tobytes())
    struct.pack_into("<74I", head, block, base8, base16, *offsets)
    struct.pack_into("<I", head, 8, len(weights))
    weights16 = bytearray((src / "weights16.bin").read_bytes())
    if half_input:
        # These copies decode the original FP8 codes exactly; only activations
        # change precision in this experiment. Slot offsets are in f16 elements.
        code = np.arange(256, dtype=np.uint32)
        exponent, mantissa = (code & 127) >> 3, code & 7
        decoded = np.where(exponent == 0, mantissa / 512.0,
                           (1.0 + mantissa / 8.0) * np.exp2(exponent.astype(float) - 7))
        decoded = np.copysign(decoded, np.where(code & 128, -1.0, 1.0))
        decoded[(code & 127) == 127] = np.nan
        decoded = decoded.astype(np.float16)
        for position in (40, block):
            b8, b16, *o = struct.unpack_from("<74I", head, position)
            sources = [o[4], o[5], o[8], o[9], *[o[20 + 3 * c] for c in range(4)]]
            for slot, source in zip(range(49, 57), sources):
                raw = np.frombuffer(bytes(weights[b8 + source:b8 + source + 1024]), np.uint8)
                weights16.extend(bytes((-len(weights16)) % 64))
                o[slot] = (len(weights16) - b16) // 2
                weights16.extend(decoded[raw].tobytes())
            struct.pack_into("<74I", head, position, b8, b16, *o)
        struct.pack_into("<I", head, 12, len(weights16))
    shutil.copytree(src, dst)
    (dst / "offsets.bin").write_bytes(head)
    (dst / "weights8.bin").write_bytes(weights)
    (dst / "weights16.bin").write_bytes(weights16)
    flags = ["-DM_RELAXED", f"-DWPW={wpw}", "-DM_RESIDUAL_F32", "-DM_F32ACT",
             "-DM_RES32", "-DM_PK16", "-DM_BIAS32", "-DM_LDAU", "-DM_L_HEAD16", *extra]
    if half_input:
        flags.append("-DM_L_HALF_INPUT")
    for planes, suffix in ((0, ""), (1, "_planes")):
        subprocess.run(["glslangValidator", "-V", "--target-env", "vulkan1.3",
                        *defines("dec0l"), f"-DPLANES={planes}", *flags,
                        str(ROOT / "engine/layer_m.comp"), "-o", str(dst / f"dec0_l{suffix}.spv")], check=True)
        if half_input:
            subprocess.run(["glslangValidator", "-V", "--target-env", "vulkan1.3",
                            *defines("enc0l"), f"-DPLANES={planes}", *flags,
                            str(ROOT / "engine/layer_m.comp"), "-o", str(dst / f"enc0_l{suffix}.spv")], check=True)
    manifest = json.loads((dst / "model.json").read_text())
    manifest["experimental_head16"] = {"columns": COLUMNS, "wpw": wpw, "defines": flags,
                                       "source_model": str(src.resolve())}
    manifest["files"] = {name: hashlib.sha256((dst / name).read_bytes()).hexdigest()
                         for name in manifest["files"]}
    (dst / "model.json").write_text(json.dumps(manifest, indent=1) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--wpw", type=int, choices=(1, 2), default=2)
    parser.add_argument("--half-input", action="store_true", help="experimental FP16 normalized inputs and decoded weights in both outer stages")
    parser.add_argument("--define", action="append", default=[], help="extra shader flag, e.g. --define=-DM_SCALAR_WAVE")
    args = parser.parse_args()
    specialize(args.source, args.destination, args.wpw, args.define, args.half_input)
