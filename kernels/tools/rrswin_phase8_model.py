#!/usr/bin/env python3
"""Host model of enc0 phase 8 from the oracle's own operands, to localise the pilot's residual.

Reads the combined p8c dump (32 operand registers then 48 accumulator registers per lane per
block) and does everything the pilot does except on the host, in f64, from the *documented*
layouts:
  - the A fragments (the first 24 registers) staged into an activation image with the layout the
    enc0 body documents,
  - the weights taken from the weight image through waddr (the same lookup the engine's prep uses),
  - D = the sum over k of weight * activation, compared with the oracle's own D.
If this reproduces the oracle's D, the operand model is right and any disagreement the pilot shows
is the engine or its glue; if it does not, the operand model is what is wrong.

usage: phase8_model.py <oracle-p8c-dir>
"""
import bisect
import pathlib
import struct
import sys

sys.path.insert(0, "/home/wouter/git/d4r/kernels/rr")
import rr_layer_spec as R  # noqa: E402

BASE = 46000000
SLOT = 16384
ORE = 32          # operand registers per lane
AREGS = 24        # A fragments (6 m-tiles of 4)
WEIGHT_BASE = 21696


def find(dirname, needle):
    for p in sorted(pathlib.Path(dirname).iterdir()):
        if "swin_enc0_kernel" in p.name and needle in p.name:
            return p
    raise SystemExit(f"no dump with {needle} in {dirname}")


def e4m3_to_f32(code):
    mag = code & 0x7F
    if mag >= 8:
        bits = (((mag >> 3) + 120) << 23) | ((mag & 7) << 20)
    else:
        bits = struct.unpack('<I', struct.pack('<f', float(mag) * 2.0 ** -9))[0]
    return struct.unpack('<f', struct.pack('<I', bits | ((code & 0x80) << 24)))[0]


def waddr(k, n):
    return (512 * (n >> 4) + 64 * (n & 7) + 16 * ((k >> 1) & 3) + 8 * ((n >> 3) & 1)
            + 4 * ((k >> 3) & 1) + 2 * ((k >> 4) & 1) + (k & 1))


def f16(bits):
    return struct.unpack('<e', struct.pack('<H', bits & 0xFFFF))[0]


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "/tmp/d4r-rr-perf/d-p8c"
    dump = find(d, "arg048").read_bytes()
    wimg = find(d, "arg040").read_bytes()

    block, lane = 0, 0
    off = BASE + block * SLOT + lane * (ORE + 48) * 4

    # Each lane supplies only eight K positions for each token row. Assemble
    # all 32 lanes before taking a dot product; the combined payload has 80
    # registers per lane, not 32.
    img = [[0] * 32 for _ in range(96)]
    for source_lane in range(32):
        source_off = BASE + block * SLOT + source_lane * (ORE + 48) * 4
        ops = struct.unpack(f'<{ORE}I', dump[source_off:source_off + ORE * 4])
        for mi in range(6):
            f = ops[4 * mi:4 * mi + 4]
            for j in range(4):
                k0 = 2 * (source_lane & 3) + (j & 1) + 16 * ((j >> 1) & 1)
                k2 = 8 + k0
                img[16 * mi + (source_lane >> 2)][k0] = (f[0] >> (8 * j)) & 0xFF
                img[16 * mi + (source_lane >> 2) + 8][k0] = (f[1] >> (8 * j)) & 0xFF
                img[16 * mi + (source_lane >> 2)][k2] = (f[2] >> (8 * j)) & 0xFF
                img[16 * mi + (source_lane >> 2) + 8][k2] = (f[3] >> (8 * j)) & 0xFF

    # the oracle's D, this lane
    doff = off + ORE * 4
    acc = struct.unpack('<48I', dump[doff:doff + 48 * 4])
    g, t = lane >> 2, lane & 3

    # This lane holds only the columns 2t and 2t+1 of each 8-wide n-tile, and only the token rows
    # 16*mi + 8*rh + g -- reading any other (weight, token) from its registers reads another
    # lane's data.
    print("weight  token   model      oracle     delta")
    for wt in (2 * t, 2 * t + 1):
        for mi in (0, 1):                   # this lane holds one token row per row half
          for rh in (0, 1):
            tok = 16 * mi + 8 * rh + g
            total = 0.0
            for k in range(32):
                total += e4m3_to_f32(wimg[WEIGHT_BASE + waddr(k, wt)]) * e4m3_to_f32(img[tok][k])
            total = struct.unpack('<e', struct.pack('<e', total))[0]
            widx = wt // 8
            reg = (mi * 4 + widx) * 2 + rh
            pair = acc[reg]
            c = wt % 2
            want = f16((pair >> 16) if c else (pair & 0xFFFF))
            # which (register, half) of this lane does the model's value actually equal?
            hits = []
            for rb in range(48):
                for hh in (0, 1):
                    v = f16((acc[rb] >> 16) if hh else (acc[rb] & 0xFFFF))
                    if abs(v - total) < 1e-4 * (1 + abs(total)):
                        hits.append(f"r{rb}.{hh}")
            print(f"{wt:6d} {tok:6d}  {total:+.6f}  {want:+.6f}  delta {total - want:+.6f}  matches {hits[:4]}")


if __name__ == "__main__":
    main()
