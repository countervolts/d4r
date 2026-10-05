#!/usr/bin/env python3
"""Solve phase 8's activation row from the oracle, to derive the fragment map instead of assuming it.

For one token row, the oracle's accumulator gives 32 equations (one per weight column):

    D(tok, wt) = sum_k weight(k, wt) * a(k)

with `weight` read from the weight image through the corroborated waddr (the PTX's own loads put
phase 8's B at 21696/22208).  That is a 32x32 linear system in the 32 unknown activation values
a(k) of that token row, so it can be solved directly -- no assumption about the fragment layout.
The solution is then snapped to e4m3 codes and compared with the bytes the lanes actually hold,
which names the byte that belongs at each k: the fragment map, derived.

usage: phase8_fit.py <oracle-p8c-dir> [token]
"""
import pathlib
import struct
import sys

import numpy as np

BASE = 46000000
SLOT = 16384
ORE = 32
ACC_OFF = 32 * 4
WEIGHT_BASE = 21696
TOKEN_TILES = 6          # the activations' 96 rows
WEIGHT_TILES = 4         # the weights' 32 rows
ROWS = 96
COLS = 32


def find(dirname, needle):
    for p in sorted(pathlib.Path(dirname).iterdir()):
        if "swin_enc0_kernel" in p.name and needle in p.name:
            return p
    raise SystemExit(f"no dump with {needle} in {dirname}")


def e4m3_codes():
    """Every e4m3 code and its value."""
    out = []
    for code in range(256):
        mag = code & 0x7F
        if (code & 0x7F) == 0x7F:
            continue                     # NaN
        if mag >= 8:
            bits = (((mag >> 3) + 120) << 23) | ((mag & 7) << 20)
        else:
            bits = struct.unpack('<I', struct.pack('<f', float(mag) * 2.0 ** -9))[0]
        value = struct.unpack('<f', struct.pack('<I', bits | ((code & 0x80) << 24)))[0]
        out.append((code, value))
    return out


def waddr(k, n):
    return (512 * (n >> 4) + 64 * (n & 7) + 16 * ((k >> 1) & 3) + 8 * ((n >> 3) & 1)
            + 4 * ((k >> 3) & 1) + 2 * ((k >> 4) & 1) + (k & 1))


def f16(bits):
    return struct.unpack('<e', struct.pack('<H', bits & 0xFFFF))[0]


def snap_error(W, D):
    """Max distance from the solved activation values to the nearest e4m3 code.

    The true activation values are e4m3, so a correct (weight map, D map) pair makes this small;
    a wrong one leaves values scattered, as the 15.8 measured for the assumed pair.  This is the
    discriminating test -- the least-squares residual is not, because the system is square.
    """
    a, *_ = np.linalg.lstsq(W, D, rcond=None)
    codes = e4m3_codes()
    errs = [min(abs(cv[1] - a[k]) for cv in codes) for k in range(32)]
    # the mean is the useful statistic: the max is the same for many wrong permutations, because
    # they differ only in how many values land close, not in the worst one
    return float(np.mean(errs))


def test_zluda_map(dump, wimg, tok, D, W):
    """Test the map ZLUDA's own lowering documents, rather than the body's comment.

    zluda_ptx_impl.cpp's e4m3_a_k16_fragment says: "upper holds row groupID (4 columns from
    4*threadID), lower holds row groupID+8", and the k16 halves go to registers {x,y} and {z,w}.
    That makes an A byte j of register x the k = 4*(lane&3) + j of row (lane>>2), not the
    k = 2*t + (j&1) + 16*((j>>1)&1) rrswin_enc0.hip claims.  This assembles the row from those
    bytes and compares it with the value solved out of the oracle.
    """
    a, *_ = np.linalg.lstsq(W, D, rcond=None)
    codes = e4m3_codes()
    solved = [min(codes, key=lambda cv: abs(cv[1] - a[k]))[0] for k in range(32)]
    held = {}
    for lane in range(32):
        if (lane >> 2) != (tok % 8):
            continue
        high = 1 if tok >= 8 else 0
        ops = struct.unpack(f'<{ORE}I', dump[BASE + lane * ORE * 4:BASE + lane * ORE * 4 + ORE * 4])
        mi = tok // 16
        rx = 4 * mi + (2 if high else 0)
        rz = 4 * mi + (3 if high else 1)
        for j in range(4):
            held[4 * (lane & 3) + j] = (ops[rx] >> (8 * j)) & 0xFF
            held[16 + 4 * (lane & 3) + j] = (ops[rz] >> (8 * j)) & 0xFF
    hit = sum(1 for k in range(32) if held.get(k) == solved[k])
    return hit, solved, held


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "/tmp/d4r-rr-perf/d-p8c"
    tok = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    dump = find(d, "arg048").read_bytes()
    wimg = find(d, "arg040").read_bytes()

    # the oracle's D for this token: rows 16*mi + 8*rh + g live in lane M with g = M>>2, t = M&3
    mi, h, g = tok // 16, (tok % 16) // 8, (tok % 16) % 8
    rh = h                       # the register pair's row half is the lane's half
    D = np.zeros(COLS)
    lane_bytes = {}
    for widx in range(WEIGHT_TILES):
        for t in range(4):
            M = 16 * h + t       # derived: lane = 16*((tok%16)/8) + (wt%8)/2
            off = BASE + M * ORE * 4 + ACC_OFF
            acc = struct.unpack('<48I', dump[off:off + 48 * 4])
            for c in range(2):
                wt = 8 * widx + 2 * t + c
                reg = (mi * 4 + widx) * 2 + rh
                D[wt] = f16((acc[reg] >> 16) if c else (acc[reg] & 0xFFFF))
            # this lane's A bytes, for naming the solved values
            ops = struct.unpack(f'<{ORE}I', dump[BASE + M * ORE * 4:BASE + M * ORE * 4 + ORE * 4])
            lane_bytes[M] = [((ops[4 * mi + r] >> (8 * j)) & 0xFF) for r in range(4) for j in range(4)]

    W = np.zeros((COLS, 32))
    for wt in range(COLS):
        for k in range(32):
            W[wt, k] = e4m3_codes_value(wimg[WEIGHT_BASE + waddr(k, wt)])

    a, *_ = np.linalg.lstsq(W, D, rcond=None)
    residual = float(np.abs(W @ a - D).max())
    # The system is square -- 32 weight columns for 32 unknown k -- so a small residual proves
    # nothing by itself: it can be solved exactly whether or not the model is right.  The real
    # test is whether the *solution* lands on e4m3 codes, since the true activation values are
    # e4m3.  Snap error is that measure.
    codes_all = e4m3_codes()
    _ = snap_error  # used by the search below
    snap_err = max(min(abs(cv[1] - a[k]) for cv in codes_all) for k in range(32))
    print(f"token {tok}: solved 32 activation values, residual {residual:.3e} "
          f"(square system, so not evidence), max snap-to-e4m3 error {snap_err:.3e} "
          f"({'consistent with e4m3 inputs' if snap_err < 1e-3 else 'NOT e4m3: the weights or the D mapping is wrong'})")

    codes = e4m3_codes()
    snaps = []
    for k in range(32):
        best = min(codes, key=lambda cv: abs(cv[1] - a[k]))
        snaps.append(best[0])
    print("solved and snapped activation row (k: code):")
    print(" ".join(f"{c:02x}" for c in snaps))
    # Derive the token -> lane assignment: for each candidate token row, solve and see whether
    # the resulting codes are exactly the bytes some lane group holds.
    print("token -> lane-group derivation (which solved row is a permutation of held bytes):")
    for cand in range(96):
        cmi, crh, cg = cand // 16, (cand % 16) // 8, (cand % 16) % 8
        Dc = np.zeros(COLS)
        held = []
        for widx in range(WEIGHT_TILES):
            for t in range(4):
                M = 16 * crh + t
                off = BASE + M * ORE * 4 + ACC_OFF
                acc = struct.unpack('<48I', dump[off:off + 48 * 4])
                for c in range(2):
                    wt = 8 * widx + 2 * t + c
                    reg = (cmi * 4 + widx) * 2 + crh
                    Dc[wt] = f16((acc[reg] >> 16) if c else (acc[reg] & 0xFFFF))
                ops = struct.unpack(f'<{ORE}I', dump[BASE + M * ORE * 4:BASE + M * ORE * 4 + ORE * 4])
                held += [((ops[4 * cmi + (0 if crh == 0 else 1)] >> (8 * j)) & 0xFF) for j in range(4)]
                held += [((ops[4 * cmi + (2 if crh == 0 else 3)] >> (8 * j)) & 0xFF) for j in range(4)]
        ac, *_ = np.linalg.lstsq(W, Dc, rcond=None)
        res = float(np.abs(W @ ac - Dc).max())
        snaps_c = [min(codes, key=lambda cv: abs(cv[1] - ac[k]))[0] for k in range(32)]
        if sorted(snaps_c) == sorted(held) and res < 1e-9:
            print(f"  token row {cand:2d} (mi {cmi}, rh {crh}, g {cg}) solves to exactly the 32 bytes "
                  f"held by lanes {4*cg}..{4*cg+3} of m-tile {cmi} row half {crh}")
            break
    else:
        print("  no token row's solution matched the bytes held for its group")

    # Derive the register-to-row assignment: solve every row exactly, then ask which
    # (lane, m-tile, row half) register set holds that row's values.
    print("register-to-row derivation:")
    solved = {}
    for cand in range(96):
        cmi, crh, cg = cand // 16, (cand % 16) // 8, (cand % 16) % 8
        Dc = np.zeros(COLS)
        for widx in range(WEIGHT_TILES):
            for t in range(4):
                M = 4 * cg + t
                off = BASE + M * ORE * 4 + ACC_OFF
                acc = struct.unpack('<48I', dump[off:off + 48 * 4])
                for c in range(2):
                    Dc[8 * widx + 2 * t + c] = f16((acc[((cmi * 4 + widx) * 2 + crh)] >> (16 if c else 0)) & 0xFFFF)
        ac, *_ = np.linalg.lstsq(W, Dc, rcond=None)
        solved[cand] = [min(codes, key=lambda cv: abs(cv[1] - ac[k]))[0] for k in range(32)]

    hits = {}
    for lane in range(32):
        ops = struct.unpack(f'<{ORE}I', dump[BASE + lane * ORE * 4:BASE + lane * ORE * 4 + ORE * 4])
        for mi in range(6):
            for rh in (0, 1):
                regs = (1, 3) if rh else (0, 2)
                held = [((ops[4 * mi + r] >> (8 * j)) & 0xFF) for r in regs for j in range(4)]
                for cand in range(96):
                    want = list(solved[cand])
                    ok = all(want.count(b) >= held.count(b) for b in set(held))
                    if ok:
                        hits.setdefault(cand, []).append((lane, mi, rh))
    for cand in sorted(hits)[:6]:
        (lane, mi, rh) = hits[cand][0]
        print(f"  row {cand:2d} (mi {cand//16}, rh {(cand%16)//8}, g {(cand%16)%8}): held by "
              f"lane {lane:2d} m-tile {mi} row-half {rh}  ({len(hits[cand])} candidates)")

    # Search the plausible variants of the two maps and report the snap error for each.  The
    # correct pair makes the solution land on e4m3 codes.
    print("map candidates (max snap-to-e4m3 error of the solved activation row):")
    import itertools

    def wmap(k, wt, variant):
        if variant == 0:
            return waddr(k, wt)
        if variant == 1:                       # k16 halves swapped
            return waddr((k + 8) % 32 if (k % 16) < 8 else (k - 8), wt)
        if variant == 2:                       # the two (k>>1)&3 / (k>>3)&1 terms exchanged
            n_ = wt
            return (512 * (n_ >> 4) + 64 * (n_ & 7) + 16 * ((k >> 3) & 1) + 8 * ((n_ >> 3) & 1)
                    + 4 * ((k >> 1) & 3) + 2 * ((k >> 4) & 1) + (k & 1))
        if variant == 3:                       # the k&1 / (k>>1)&1 terms exchanged
            n_ = wt
            return (512 * (n_ >> 4) + 64 * (n_ & 7) + 16 * ((k >> 1) & 3) + 8 * ((n_ >> 3) & 1)
                    + 4 * ((k >> 3) & 1) + 2 * ((k >> 4) & 1) + ((k >> 1) & 1) * 1)
        return waddr(k, wt)

    def dmap(tok, wt, variant):
        mi, rh, g = tok // 16, (tok % 16) // 8, (tok % 16) % 8
        M = 4 * g + ((wt % 8) // 2)
        off = BASE + M * ORE * 4 + ACC_OFF
        acc = struct.unpack('<48I', dump[off:off + 48 * 4])
        if variant == 0:
            widx, c = wt // 8, wt % 2
            reg = (mi * 4 + widx) * 2 + rh
        elif variant == 1:                     # n-tile major instead of m-tile major
            widx, c = wt // 8, wt % 2
            reg = (widx * 6 + mi) * 2 + rh
        elif variant == 2:                     # row halves swapped
            widx, c = wt // 8, wt % 2
            reg = (mi * 4 + widx) * 2 + (1 - rh)
        else:
            return None
        return f16((acc[reg] >> 16) if c else (acc[reg] & 0xFFFF))

    for wv, dv in itertools.product(range(4), range(3)):
        Wv = np.zeros((COLS, 32))
        for wt in range(COLS):
            for k in range(32):
                Wv[wt, k] = e4m3_codes_value(wimg[WEIGHT_BASE + wmap(k, wt, wv)])
        Dv = np.zeros(COLS)
        ok = True
        for wt in range(COLS):
            v = dmap(tok, wt, dv)
            if v is None:
                ok = False
                break
            Dv[wt] = v
        if not ok:
            continue
        print(f"  weight map {wv}, D map {dv}: {snap_error(Wv, Dv):.3e}")

    # The body's waddr is a hand-derived inverse, i.e. a bit permutation of the k index onto the
    # byte-offset bits.  Search the permutation instead of trusting it: 5 k bits onto the offset
    # bits {0,1,2,4,5} (offset bit 3 belongs to n), with the n terms fixed.  The correct
    # permutation makes the solved activation row land exactly on e4m3 codes.
    # NOTE: this search's encoding of the permutation is not yet right -- candidates cluster
    # around 0.43 rather than reaching ~0, so the winner found here is not trustworthy.  It does
    # establish the thing that matters at this stage: the maps in rrswin_enc0.hip give a snap
    # error of 15.8, i.e. they are *wrong*, and the e4m3-snap error is the test that shows it.
    print("k-bit permutation search (offset bits {0,1,2,4,5} <- k bits):")
    # Both maps are searched together: the k-bit permutation (5! of them) and the D register
    # order, since a wrong D map makes every permutation look wrong.
    def dval(order, wt):
        mi, rh, g = tok // 16, (tok % 16) // 8, (tok % 16) % 8
        M = 16 * rh + ((wt % 8) // 2)      # derived lane mapping
        off = BASE + M * ORE * 4 + ACC_OFF
        acc = struct.unpack('<48I', dump[off:off + 48 * 4])
        widx, c = wt // 8, wt % 2
        if order == 0:
            reg = (mi * 4 + widx) * 2 + rh
        elif order == 1:
            reg = (widx * 6 + mi) * 2 + rh
        elif order == 2:
            reg = (mi * 4 + widx) * 2 + (1 - rh)
        else:
            reg = (mi * 4 + (3 - widx)) * 2 + rh
        return f16((acc[reg] >> 16) if c else (acc[reg] & 0xFFFF))

    # A k-bit permutation only relabels which k holds which value, so it cannot change the
    # *set* of solved values and the snap statistic is blind to it.  The test that does work is
    # permutation-invariant and uses the registers themselves: do the solved values equal the
    # bytes the lanes hold?  Only the D map can be searched this way.
    print("D-map search by multiset match with the held bytes:")
    Wbase = np.zeros((COLS, 32))
    for wt in range(COLS):
        for k in range(32):
            Wbase[wt, k] = e4m3_codes_value(wimg[WEIGHT_BASE + waddr(k, wt)])
    held_all = set()
    for lane in range(32):
        ops = struct.unpack(f'<{ORE}I', dump[BASE + lane * ORE * 4:BASE + lane * ORE * 4 + ORE * 4])
        for mi in range(6):
            for r in range(4):
                for j in range(4):
                    held_all.add((ops[4 * mi + r] >> (8 * j)) & 0xFF)
    for order in range(4):
        Dv = np.array([dval(order, wt) for wt in range(COLS)])
        a, *_ = np.linalg.lstsq(Wbase, Dv, rcond=None)
        codes_all = e4m3_codes()
        snaps = [min(codes_all, key=lambda cv: abs(cv[1] - a[k]))[0] for k in range(32)]
        inside = sum(1 for c in snaps if c in held_all)
        # strict: the solved row must equal the union of the bytes the four lanes of this token's
        # group hold for this row half, as a multiset
        g_ = (tok % 16) % 8
        strict = []
        for lane in range(4 * g_, 4 * g_ + 4):
            ops = struct.unpack(f'<{ORE}I', dump[BASE + lane * ORE * 4:BASE + lane * ORE * 4 + ORE * 4])
            regs = (1, 3) if order == 2 else (0, 2)
            for r in regs:
                for j in range(4):
                    strict.append((ops[4 * (tok // 16) + r] >> (8 * j)) & 0xFF)
        exact = sorted(snaps) == sorted(strict)
        print(f"  D order {order}: {inside:2d} of 32 solved values are values the payload holds"
              f"{'  EXACT multiset match with the group\'s held bytes' if exact else ''}")
    # Derive the placement: which byte position holds each k.  The solved row gives the value for
    # every k in the *weight* labelling (the one waddr uses, which the exact-w-values test just
    # corroborated).  The payload gives the bytes and their (lane, register, byte) positions.  A
    # position is the k label whose solved value it holds, so matching them derives the map the
    # engine must use for its activation image to be consistent with the weights.
    hit, solved, held = test_zluda_map(dump, wimg, tok, D, W)
    print(f"ZLUDA's documented map (k = 4*(lane&3)+j, row = lane>>2): {hit} of 32 k match exactly")
    if hit < 32:
        print("  k : solved  held")
        for k in range(32):
            mark = "" if held.get(k) == solved[k] else "   <-- differs"
            print(f"  {k:2d}: {solved[k]:02x}      {held.get(k, 0):02x}{mark}")

    print("placement: which byte position holds each k label")
    a_all, *_ = np.linalg.lstsq(Wbase, D, rcond=None)
    codes_all = e4m3_codes()
    label_of = {}
    for k in range(32):
        snapped = min(codes_all, key=lambda cv: abs(cv[1] - a_all[k]))[0]
        label_of.setdefault(snapped, []).append(k)
    # The solved row's values appear somewhere in the payload but not in the group I assumed, so
    # the row-to-group mapping is what to derive: try every (g, row half) and report which group's
    # bytes are exactly the solved row's values.
    # wmma_layout.h says "A operand lane l = row r" with r = l & 15, while rrswin_enc0.hip's
    # header says row g = lane >> 2.  They are different conventions, so test the other one: a
    # row's 16 bytes per lane over two lanes (m and m+16) rather than four lanes of a group.
    print("  row-to-pair search (row = lane & 15, 16 bytes per lane):")
    solved_multi = sorted(label_of.keys())
    for row in range(16):
        held = []
        for lane in (row, row + 16):
            ops = struct.unpack(f'<{ORE}I', dump[BASE + lane * ORE * 4:BASE + lane * ORE * 4 + ORE * 4])
            for rr in range(4):
                for j in range(4):
                    held.append((ops[4 * (tok // 16) + rr] >> (8 * j)) & 0xFF)
        hit = sum(1 for b in solved_multi if b in held)
        miss = [b for b in solved_multi if b not in held]
        if hit >= 28:
            print(f"    row {row:2d}: {hit} of 32 solved values held by lanes {row},{row+16}"
                  f"{'  EXACT' if not miss else ''}")

    print("  row-to-group search:")
    solved_multi = sorted(label_of.keys())
    for cand_g in range(8):
        for cand_r in (0, 1):
            held = []
            for lane in range(4 * cand_g, 4 * cand_g + 4):
                ops = struct.unpack(f'<{ORE}I', dump[BASE + lane * ORE * 4:BASE + lane * ORE * 4 + ORE * 4])
                regs = (1, 3) if cand_r else (0, 2)
                for rr in regs:
                    for j in range(4):
                        held.append((ops[4 * (tok // 16) + rr] >> (8 * j)) & 0xFF)
            hit = sum(1 for b in solved_multi if b in held)
            if hit >= 28:
                print(f"    g {cand_g}, row half {cand_r}: {hit} of {len(solved_multi)} solved values held")

    g_ = (tok % 16) % 8
    print("  group lanes " + " ".join(str(4 * g_ + t) for t in range(4)) + ":")
    for lane in range(4 * g_, 4 * g_ + 4):
        ops = struct.unpack(f'<{ORE}I', dump[BASE + lane * ORE * 4:BASE + lane * ORE * 4 + ORE * 4])
        for r in range(4):
            row = "[row g]" if r in (0, 2) else "[row g+8]"
            ks = []
            for j in range(4):
                b = (ops[4 * (tok // 16) + r] >> (8 * j)) & 0xFF
                cand = label_of.get(b, [])
                ks.append(f"{b:02x}->k{cand[0]}" if len(cand) == 1 else f"{b:02x}->?")
            print(f"    lane {lane:2d} reg {r} {row}: " + "  ".join(ks))

    print("bytes held by the lanes of this token's group:")
    for M in sorted(lane_bytes):
        print(f"  lane {M:2d}: " + " ".join(f"{b:02x}" for b in lane_bytes[M]))


def e4m3_codes_value(code):
    mag = code & 0x7F
    if mag >= 8:
        bits = (((mag >> 3) + 120) << 23) | ((mag & 7) << 20)
    else:
        bits = struct.unpack('<I', struct.pack('<f', float(mag) * 2.0 ** -9))[0]
    return struct.unpack('<f', struct.pack('<I', bits | ((code & 0x80) << 24)))[0]


if __name__ == "__main__":
    main()
