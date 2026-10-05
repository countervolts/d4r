#!/usr/bin/env python3
"""Dump the mma statements of a swin entry's phase range with their A/B/C/D registers.

Usage: swin_mma.py LAYER LO_STMT [HI_STMT]
LAYER is a rr_layer_spec alias (enc0, ...).  One line per mma, with the
statement index, the physical PTX line and the register operands in order.
"""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rr"))
import rr_layer_spec as R  # noqa: E402


def main():
    layer = sys.argv[1] if len(sys.argv) > 1 else "enc0"
    lo = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    hi = int(sys.argv[3]) if len(sys.argv) > 3 else 10 ** 9
    path, entry = R.resolve(layer, None, None)
    lines = open(path).read().split("\n")
    first, last = next((a, b) for (nm, a, b) in R.find_entries(lines) if nm == entry)
    stmts, _ = R.linearise(lines, first, last)
    for s in stmts:
        if s.is_mma and lo <= s.idx <= hi:
            print(f"s{s.idx} L{s.line}: " + " ".join(re.findall(r"%r\d+", s.text)))


if __name__ == "__main__":
    main()