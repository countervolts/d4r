#!/usr/bin/env python3
"""Compare PE section metadata and aligned byte blocks for local NGX DLLs."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from inspect_ngx_dll import PEImage


def version_of(image: PEImage) -> str:
    for resource in image.resources():
        info = resource.get("version_info", {}).get("strings", {})
        if "FileVersion" in info:
            return info["FileVersion"]
    return "unknown"


def compare_sections(a: PEImage, b: PEImage, block_size: int) -> None:
    by_name_a = {section["name"]: section for section in a.sections}
    by_name_b = {section["name"]: section for section in b.sections}
    common = [name for name in by_name_a if name in by_name_b]

    hash_a = hashlib.sha256(a.data).hexdigest()
    hash_b = hashlib.sha256(b.data).hexdigest()
    print(f"A: {a.path.name} version={version_of(a)} size={len(a.data)} sha256={hash_a}")
    print(f"B: {b.path.name} version={version_of(b)} size={len(b.data)} sha256={hash_b}")
    print(f"Aligned comparison uses {block_size}-byte blocks at matching section offsets.")

    for name in common:
        sa = by_name_a[name]
        sb = by_name_b[name]
        start_a, start_b = sa["raw_offset"], sb["raw_offset"]
        size_a, size_b = sa["raw_size"], sb["raw_size"]
        overlap = min(size_a, size_b)
        prefix = 0
        ba = a.data[start_a : start_a + overlap]
        bb = b.data[start_b : start_b + overlap]
        for x, y in zip(ba, bb):
            if x != y:
                break
            prefix += 1

        blocks = overlap // block_size
        equal_blocks = 0
        for index in range(blocks):
            lo = index * block_size
            hi = lo + block_size
            if ba[lo:hi] == bb[lo:hi]:
                equal_blocks += 1

        aligned_percent = 100.0 * equal_blocks / blocks if blocks else 0.0
        print(
            f"{name:8} A={size_a:>10} B={size_b:>10} "
            f"aligned-identical={equal_blocks}/{blocks} ({aligned_percent:.2f}%) "
            f"common-prefix={prefix} bytes entropy={sa['entropy']:.3f}/{sb['entropy']:.3f} "
            f"section-sha256={sa['sha256'][:12]}/{sb['sha256'][:12]}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dll_a", type=Path)
    parser.add_argument("dll_b", type=Path)
    parser.add_argument("--block-size", type=int, default=4096)
    args = parser.parse_args()
    if args.block_size < 64 or args.block_size & (args.block_size - 1):
        parser.error("--block-size must be a power of two and at least 64")

    image_a = PEImage(args.dll_a, string_limit=0)
    image_b = PEImage(args.dll_b, string_limit=0)
    compare_sections(image_a, image_b, args.block_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
