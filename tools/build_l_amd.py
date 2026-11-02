#!/usr/bin/env python3
"""Build a preset-L AMD quality-budget variant in a separate model directory.

The variants change arithmetic intentionally. Evaluate final RGB PSNR against
the actual reference sequence and benchmark the same inputs before selecting one.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from build_l_head16 import ROOT, specialize
from test_m import defines

PROFILES = {
    "layout": [],
    "head": ["M_AMD_HEAD"],
    "norm": ["M_AMD_HEAD", "M_AMD_NORM"],
    "softmax": ["M_AMD_HEAD", "M_AMD_NORM", "M_AMD_SOFTMAX"],
    "pv8": ["M_AMD_HEAD", "M_AMD_NORM", "M_AMD_PV8"],
    "native": ["M_AMD_HEAD", "M_AMD_NORM", "M_AMD_SOFTMAX", "M_AMD_PV8"],
    "gelu": ["M_AMD_HEAD", "M_AMD_NORM", "M_AMD_SOFTMAX", "M_AMD_PV8", "M_AMD_GELU"],
    "head-gelu": ["M_AMD_HEAD", "M_AMD_GELU"],
    "balanced": ["M_AMD_HEAD", "M_AMD_NORM"],
    "fast": ["M_AMD_HEAD", "M_AMD_NORM", "M_AMD_SOFTMAX", "M_AMD_PV8", "M_AMD_GELU"],
}
NATIVE_PROFILES = {
    "balanced": ["SWIN_AMD_SAT8"],
    # The all-layer hard-GELU experiment misses the temporal 40 dB gate.
    # Keep it in the outer C32 blocks; use hardware FP8 saturation throughout.
    "fast": ["SWIN_AMD_SAT8"],
    "native": ["SWIN_AMD_SOFTMAX", "SWIN_AMD_PV8"],
    "gelu": ["SWIN_AMD_SOFTMAX", "SWIN_AMD_PV8", "SWIN_AMD_GELU"],
}


def build(source, destination, profile, extra, native_layers=None, history_bilinear=False):
    ldr = ["-DLDR=1"] if (source / "ldr").exists() else []
    native = []
    if native_layers:
        original = json.loads((source / "model.json").read_text())
        native = [name for name in original["files"] if name.startswith("hip/") and name.endswith(".hsaco")]
        if not native:
            raise ValueError("source model has no native HIP layers")
        for name in native:
            if not (native_layers / Path(name).name).is_file():
                raise ValueError(f"native layer missing: {name}")
    flags = ["-DM_L_HEAD16_WAVE", "-DM_L_EXPAND_DIRECT"]
    flags += ["-D" + name for name in PROFILES[profile]] + extra
    specialize(source, destination, 2, flags)
    manifest = json.loads((destination / "model.json").read_text())
    # The head builder specializes the decoder. Apply the same arithmetic
    # profile to the full-resolution encoder, with identical resource bindings.
    flags = manifest["experimental_head16"]["defines"]
    for planes, suffix in ((0, ""), (1, "_planes")):
        subprocess.run(["glslangValidator", "-V", "--target-env", "vulkan1.3",
                        *defines("enc0l"), f"-DPLANES={planes}", *flags,
                        *ldr,
                        str(ROOT / "engine/layer_m.comp"), "-o",
                        str(destination / f"enc0_l{suffix}.spv")], check=True)
    manifest["experimental_amd"] = {"profile": profile, "quality_metric": "reference RGB PSNR",
                                    "bit_exact_required": False}
    if history_bilinear:
        for name, tile in (("post_l", 11), ("post_l_q", 16), ("post_l_c", 12)):
            subprocess.run(["glslangValidator", "-V", "--target-env", "vulkan1.3",
                            "-DPRESET_L=1", f"-DTILE_W={tile}", f"-DTILE_H={tile}",
                            "-DM_AMD_HISTORY_BILINEAR", "-DM_AMD_POST_MATH",
                            *ldr,
                            str(ROOT / "engine/post_m.comp"), "-o", str(destination / f"{name}.spv")], check=True)
        manifest["experimental_amd"]["history_filter"] = "bilinear"
    if native_layers:
        for name in native:
            shutil.copy2(native_layers / Path(name).name, destination / name)
        manifest["experimental_amd"]["native_layers"] = str(native_layers.resolve())
    manifest["files"] = {name: hashlib.sha256((destination / name).read_bytes()).hexdigest()
                         for name in manifest["files"]}
    (destination / "model.json").write_text(json.dumps(manifest, indent=1) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--profile", choices=PROFILES, default="fast")
    parser.add_argument("--define", action="append", default=[])
    native = parser.add_mutually_exclusive_group()
    native.add_argument("--native-layers", type=Path, help="directory of separately built AMD HIP code objects")
    native.add_argument("--build-native", action="store_true", help="build the profile's HIP kernels with ROCm")
    parser.add_argument("--arch", choices=("gfx1200", "gfx1201"), default="gfx1201")
    parser.add_argument("--history-bilinear", action="store_true", help="experimental one-tap history filter instead of bicubic")
    args = parser.parse_args()
    if args.destination.exists():
        parser.error("destination must not exist")
    if args.build_native:
        if args.profile not in NATIVE_PROFILES:
            parser.error("this profile has no native HIP configuration; supply --native-layers")
        args.native_layers = args.destination.parent / (args.destination.name + "-hip")
        if args.native_layers.exists():
            parser.error("native build directory must not exist")
        env = dict(os.environ, D4R_VKHIP_LAYERS="1", D4R_GPU_ARCH=args.arch, D4R_NATIVE_FP8="1",
                   D4R_PREFER_ACCURACY="0", D4R_SWIN_CFLAGS=" ".join("-D" + f for f in NATIVE_PROFILES[args.profile]))
        subprocess.run(["bash", str(ROOT / "kernels/build.sh"), "m", str(args.native_layers.resolve())],
                       check=True, env=env)
    build(args.source, args.destination, args.profile, args.define, args.native_layers, args.history_bilinear)
