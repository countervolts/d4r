#!/usr/bin/env python3
"""Build the RDNA4 outer-block dataflow redesign for preset L.

The default folds encoder norm 2 and both decoder norms, keeps features and V
in registers, quantizes the skip/merge producer once, and exchanges the global
skip in contiguous 8-channel planes. Validate every output against a reference
with tools/bench_l_amd.py --min-psnr 40. No bit-equality criterion is used.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from build_l_amd import ROOT, build
from fold_l_norm_gains import SELECTIONS, fold

DEFAULT_FOLDS = ("enc:n2", "dec:n1", "dec:n2")


def build_outer(source, destination, native_layers, features="fused", selected=DEFAULT_FOLDS, dot2=False):
    original = json.loads((source / "model.json").read_text())
    if original.get("folded_outer_norms") or (source / "enc_features_l.spv").exists():
        raise ValueError("start from a model without folded gains or a split feature pass")
    flags = ["-DM_AMD_REGISTER_INPUT", "-DM_AMD_REGISTER_V", "-DM_AMD_PACKED_SKIP"]
    flags += ["-DM_AMD_FOLD_" + item.replace(":", "_").upper() for item in selected]
    if features == "split":
        flags.append("-DM_AMD_SPLIT_FEATURES")
    if dot2:
        flags.append("-DM_AMD_NORM_DOT2")
    build(source, destination, "fast", flags, native_layers)
    fold(destination, selected)
    manifest = json.loads((destination / "model.json").read_text())
    if features == "split":
        ldr = ["-DLDR=1"] if (source / "ldr").exists() else []
        shader = destination / "enc_features_l.spv"
        subprocess.run(["glslangValidator", "-V", "--target-env", "vulkan1.3",
                        "-DC=32", "-DKIND=3", "-DNH=2", "-DXC=64", "-DTEXL=1",
                        *manifest["experimental_head16"]["defines"], *ldr, "-DM_AMD_FEATURE_PASS",
                        str(ROOT / "engine/layer_m.comp"), "-o", str(shader)], check=True)
        manifest["files"][shader.name] = hashlib.sha256(shader.read_bytes()).hexdigest()
    manifest["experimental_outer"] = {"architecture": "RDNA4", "features": features,
                                      "skip_layout": "channel8_planes", "folds": list(selected),
                                      "normalization_dot2": dot2, "quality_target_db": 40,
                                      "quality_validated": False}
    (destination / "model.json").write_text(json.dumps(manifest, indent=1) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--native-layers", type=Path, help="reuse an existing SWIN_AMD_SAT8 HIP build")
    parser.add_argument("--arch", choices=("gfx1200", "gfx1201"), default="gfx1201")
    parser.add_argument("--features", choices=("fused", "split"), default="fused",
                        help="split is an RGP comparison experiment; slower on the measured gfx1201")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--fold", choices=SELECTIONS, action="append")
    selection.add_argument("--no-fold", action="store_true")
    parser.add_argument("--norm-dot2", action="store_true", help="experimental mixed-FP16 dot / FP32 accumulation norm")
    args = parser.parse_args()
    if args.destination.exists():
        parser.error("destination must not exist")
    if not args.native_layers:
        args.native_layers = args.destination.parent / (args.destination.name + "-hip")
        if args.native_layers.exists():
            parser.error("native build directory must not exist")
        env = dict(os.environ, D4R_VKHIP_LAYERS="1", D4R_GPU_ARCH=args.arch, D4R_NATIVE_FP8="1",
                   D4R_PREFER_ACCURACY="0", D4R_SWIN_CFLAGS="-DSWIN_AMD_SAT8")
        subprocess.run(["bash", str(ROOT / "kernels/build.sh"), "m", str(args.native_layers.resolve())],
                       env=env, check=True)
    selected = [] if args.no_fold else args.fold or DEFAULT_FOLDS
    build_outer(args.source, args.destination, args.native_layers, args.features, selected, args.norm_dot2)
