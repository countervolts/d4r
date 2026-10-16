#!/usr/bin/env python3
"""Append the RDNA3 (RR_GFX11) widened copy of a native stage's weight file and print its f16 element offset (W_WIDE).

usage: widen_weights.py WEIGHTS.bin [OUT.bin]   (default: in place)

Element W_WIDE + j of the file's f16 view is byte j of the original image as ZLUDA widens an FP8 operand on gfx11
(rr_gfx11.glsl): the magnitude at bits 7..14, the sign at bit 15, the NaN code with bit 14 set too, i.e. the binary16
of value * 2^-8. Bytes that are not FP8 (tables, binary16 data) are widened as well and never read. W_WIDE is the
original size rounded up to 512 bytes, halved.
"""
import sys
from pathlib import Path

import numpy as np


def widen(image):
    b = np.frombuffer(image, np.uint8).astype(np.uint16)
    mag = b & 0x7f
    mag = mag | ((mag + 1) & 0x80)
    return ((mag << 7) | ((b & 0x80) << 8)).astype('<u2').tobytes()


def main():
    src = Path(sys.argv[1]); dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src
    image = src.read_bytes()
    base = (len(image) + 511) // 512 * 512
    dst.write_bytes(image + bytes(base - len(image)) + widen(image))
    print(base // 2)


if __name__ == '__main__':
    main()
