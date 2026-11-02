#!/usr/bin/env python3
"""Replace three K reconstruction-prefetch integer divisions exactly.

Generated proprietary shaders are input only; this pass contains no shader or model data.
For n < 1024 and 1 <= d <= 24, floor(n * ceil(65536/d) / 65536) == floor(n/d).
All other bit patterns retain the original signed integer division.
"""
import argparse
from pathlib import Path
import re

RECIPROCALS = (0,) + tuple((65536 + d - 1) // d for d in range(1, 25))
PATTERN = re.compile(r"uint\(int\((r\d+)\) / int\(r7\)\)")
NAME = "D4R_PREFETCH_DIV"

def quotient(n: int, d: int) -> int:
    if not 0 <= n < 1024 or not 1 <= d <= 24:
        raise ValueError("outside exact lookup domain")
    return (n * RECIPROCALS[d]) >> 16

def rewrite(source: str) -> str:
    if NAME in source:
        raise ValueError("prefetch division already specialized")
    if source.count("void main()") != 1:
        raise ValueError("expected one reconstruction entry point")
    result, count = PATTERN.subn(lambda m: f"uint({NAME}({m[1]}, r7))", source)
    if count != 3:
        raise ValueError("expected three cooperative-prefetch divisions")
    helper = """// Exact integer quotient for the small cooperative prefetch rectangle.
// Guard preserves original signed division for every out-of-domain argument.
int D4R_PREFETCH_DIV(uint n, uint d) {
    const uint reciprocal[25] = uint[25](""" + ", ".join(f"{x}u" for x in RECIPROCALS) + """);
    if(n<1024u && d>=1u && d<=24u) return int((n * reciprocal[d]) >> 16u);
    return int(n) / int(d);
}
"""
    return result.replace("void main()", helper + "\nvoid main()", 1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(rewrite(args.source.read_text()))
