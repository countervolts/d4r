#!/usr/bin/env python3
"""Splice recipe for cuda_dldn_engine_swin_dec1_kernel (nvngx_dlssd.dll 310.7).

kernels/tex/make_ptx.py's per-layer hook imports this module by path and returns
`recipe()`.  Everything here is self-contained: nothing is imported from
make_ptx.py or rr_layer_spec.py.

The cut replaces the whole 26-phase mma region of the entry -- physical lines
1486-38472 of ~/.cache/d4r-rr-corpus/0022-PREPASS_ENTRYPOINT_NAME.ptx, statement
range s403-s15988 -- with one `call.uni d4r_swin_dec1`.  NVIDIA's texture
prologue (lines 1-1485, which stage phase 1's A fragments and its sixteen bias
seeds) and its surface epilogue (line 38473 to the end, which stages phase 26's
accumulators through shared memory, softmaxes over them, gathers three textures
and writes param_0+112) stay in PTX.

The callee is handed, in this order:

  * the 36 A registers of phase 1 (nine .f16.e4m3 fragments of four b32, the
    first mma's operand order),
  * the sixteen C bias seeds phase 1 reads from the weight image at +24576..24816
    (every other mma's C operand is another mma's D register, which the body
    computes itself, so no other C register is passed),
  * the weight image pointer (%rd3, the `cvta.to.global` of param_0+40),
  * the parameter block pointer (%rd8, param_0 itself),
  * the address of the entry's 6400-byte .shared scratch.

`call.uni` return values cannot come back to PTX (ZLUDA lowers them to an LLVM
return type with no .param-space object), so the ten values the epilogue reads
out of the deleted region cross the cut through the shared scratch, 40 bytes per
lane: phase 26's eight accumulator registers %r18794 %r18795 %r18804 %r18805
%r18834 %r18835 %r18844 %r18845 (E26 stages them at s16000..s16019 and reads them
back at s16085..s16100), plus %r18991 (= 32*tid.z, computed inside the cut at
line 38396) and %r18993 (the shared base, line 38394).  rr_layer_spec.py's
`live_across` reports only six of these because dest_regs() reads the first
register of an mma's D list only; the four second halves are read by E26 too.

D4R_RRSWIN_DEBUG=<stage> builds NVIDIA's own body with that stage's registers
stored into the plane arena at +DEBUG_BASE, which kernels/rr/rrswin_dec1.hip
reproduces with -DD4R_RRSWIN_DEBUG_STAGE=<n>.
"""

import os

MODULE = "0022-PREPASS_ENTRYPOINT_NAME.ptx"
CALLEE = "d4r_swin_dec1"

# the mma region: the first mma's first physical line, and the line after the last
# mma's last physical line
CUT_CALL = (1486, 38473)

# phase 1's A fragments, in the first mma's operand order (s403 reads %r1084..)
A_REGS = [
    "%r1084", "%r1085", "%r1086", "%r1087",
    "%r1104", "%r1105", "%r1106", "%r1107",
    "%r1124", "%r1125", "%r1126", "%r1127",
    "%r1564", "%r1565", "%r1566", "%r1567",
    "%r1584", "%r1585", "%r1586", "%r1587",
    "%r1604", "%r1605", "%r1606", "%r1607",
    "%r2044", "%r2045", "%r2046", "%r2047",
    "%r2064", "%r2065", "%r2066", "%r2067",
    "%r2084", "%r2085", "%r2086", "%r2087",
]

# phase 1's sixteen bias seeds, in weight-image offset order (+24576, +24592, ...)
C_REGS = [
    "%r1621", "%r1631", "%r1681", "%r1691",
    "%r1741", "%r1751", "%r1801", "%r1811",
    "%r1861", "%r1871", "%r1921", "%r1931",
    "%r1981", "%r1991", "%r2041", "%r2051",
]

# what the epilogue reads back out of the deleted region
RET_REGS = ["%r18794", "%r18795", "%r18804", "%r18805",
            "%r18834", "%r18835", "%r18844", "%r18845",
            "%r18991", "%r18993"]
RET_STRIDE = 4 * len(RET_REGS)

SCRATCH = "_ZZ33cuda_dldn_engine_swin_dec1_kernel33DldnEngineSwinEncParamsStructBaseE4smem"

# registers the splice reads: %rd3 is the globalised weight image (s9), %rd8 is
# param_0 itself (s1)
W_REG = "%rd3"
V_REG = "%rd8"

# --- oracle debug stages --------------------------------------------------
# The payload goes into the param_0+8 feature-plane buffer, and at offset 0 from
# that pointer.  enc0's stages use param_0+48 + 46,000,000 because enc0's plane
# arena is 45,219,840 bytes; dec1's param_0+48 is a different, much smaller
# allocation (E25 only ever writes 8*ex*ey*4 = 7,536,640 bytes of it at
# 640x368), and 46,000,000 past either param_0+8 or param_0+48 faults with
# "page not present".  Offset 0 is safe because the entry reads param_0+8 only
# in the prologue -- s67-s324, before the cut at line 1486 -- so a store at any
# stage line is past this kernel's own last read of it, and the bridge dumps
# every argument right after the launch that produced it, so the payload is
# captured before anything else can overwrite it.
DEBUG_BASE = 0
DEBUG_GRID_X = 81            # ctaid.y * 81 + ctaid.x, one slot per block
DEBUG_MAX_BLOCK = 64
DEBUG_SLOT_FLOOR = 8192      # 32 lanes x 64 registers x 4 bytes


def debug_slot(nregs):
    """Bytes per block: 32 lanes x nregs x 4, at least the floor, a power of two
    so the splice's shift can form block * slot."""
    want = max(DEBUG_SLOT_FLOOR, 32 * nregs * 4)
    return 1 << (want - 1).bit_length()


def debug_block(regs):
    """PTX storing REGS (one u32 each) to plane arena + DEBUG_BASE +
    block*slot + lane*len(regs)*4.  make_ptx.py's debug branch inserts it into
    NVIDIA's own body; rrswin_dec1.hip's debug_store() writes the same bytes."""
    slot = debug_slot(len(regs))
    stores = "\n".join(
        "@%%pdbg1 st.global.v4.u32 [%%rddbg3+%d], {%s};"
        % (i * 16, ", ".join(regs[i * 4:i * 4 + 4]))
        for i in range((len(regs) + 3) // 4))
    return f"""// d4r debug stage
{{
.reg .pred %pdbg<2>;
.reg .b32 %rdbg<8>;
.reg .b64 %rddbg<4>;
ld.param.u64 %rddbg1, [cuda_dldn_engine_swin_dec1_kernel_param_0+8];
mov.u32 %rdbg1, %ctaid.x;
mov.u32 %rdbg2, %ctaid.y;
mad.lo.s32 %rdbg3, %rdbg2, {DEBUG_GRID_X}, %rdbg1;
setp.lt.u32 %pdbg1, %rdbg3, {DEBUG_MAX_BLOCK};
shl.b32 %rdbg4, %rdbg3, {slot.bit_length() - 1};
mov.u32 %rdbg5, %laneid;
mad.lo.s32 %rdbg6, %rdbg5, {len(regs) * 4}, %rdbg4;
add.s32 %rdbg7, %rdbg6, {DEBUG_BASE};
cvt.u64.u32 %rddbg2, %rdbg7;
add.s64 %rddbg3, %rddbg1, %rddbg2;
{stores}
}}
"""


# a1: phase 1's staged input -- the 36 A registers and the sixteen bias seeds, at
# the cut line itself, before the region has run.
A1_REGS = A_REGS + C_REGS

# p1: phase 1's 144 mma D pairs in mma order (statements s403..s546); the last mma's
# fourth line is 2490.
P1_REGS = [
    "%r652", "%r653", "%r662", "%r663",
    "%r672", "%r673", "%r682", "%r683",
    "%r692", "%r693", "%r702", "%r703",
    "%r712", "%r713", "%r722", "%r723",
    "%r732", "%r733", "%r742", "%r743",
    "%r752", "%r753", "%r762", "%r763",
    "%r772", "%r773", "%r782", "%r783",
    "%r792", "%r793", "%r802", "%r803",
    "%r812", "%r813", "%r822", "%r823",
    "%r832", "%r833", "%r842", "%r843",
    "%r852", "%r853", "%r862", "%r863",
    "%r872", "%r873", "%r882", "%r883",
    "%r892", "%r893", "%r902", "%r903",
    "%r912", "%r913", "%r922", "%r923",
    "%r932", "%r933", "%r942", "%r943",
    "%r952", "%r953", "%r962", "%r963",
    "%r972", "%r973", "%r982", "%r983",
    "%r992", "%r993", "%r1002", "%r1003",
    "%r1012", "%r1013", "%r1022", "%r1023",
    "%r1032", "%r1033", "%r1042", "%r1043",
    "%r1052", "%r1053", "%r1062", "%r1063",
    "%r1072", "%r1073", "%r1082", "%r1083",
    "%r1092", "%r1093", "%r1102", "%r1103",
    "%r1112", "%r1113", "%r1122", "%r1123",
    "%r1132", "%r1133", "%r1142", "%r1143",
    "%r1152", "%r1153", "%r1162", "%r1163",
    "%r1172", "%r1173", "%r1182", "%r1183",
    "%r1192", "%r1193", "%r1202", "%r1203",
    "%r1212", "%r1213", "%r1222", "%r1223",
    "%r1232", "%r1233", "%r1242", "%r1243",
    "%r1252", "%r1253", "%r1262", "%r1263",
    "%r1272", "%r1273", "%r1282", "%r1283",
    "%r1292", "%r1293", "%r1302", "%r1303",
    "%r1312", "%r1313", "%r1322", "%r1323",
    "%r1332", "%r1333", "%r1342", "%r1343",
    "%r1352", "%r1353", "%r1362", "%r1363",
    "%r1372", "%r1373", "%r1382", "%r1383",
    "%r1392", "%r1393", "%r1402", "%r1403",
    "%r1412", "%r1413", "%r1422", "%r1423",
    "%r1432", "%r1433", "%r1442", "%r1443",
    "%r1452", "%r1453", "%r1462", "%r1463",
    "%r1472", "%r1473", "%r1482", "%r1483",
    "%r1492", "%r1493", "%r1502", "%r1503",
    "%r1512", "%r1513", "%r1522", "%r1523",
    "%r1532", "%r1533", "%r1542", "%r1543",
    "%r1552", "%r1553", "%r1562", "%r1563",
    "%r1572", "%r1573", "%r1582", "%r1583",
    "%r1592", "%r1593", "%r1602", "%r1603",
    "%r1612", "%r1613", "%r1622", "%r1623",
    "%r1632", "%r1633", "%r1642", "%r1643",
    "%r1652", "%r1653", "%r1662", "%r1663",
    "%r1672", "%r1673", "%r1682", "%r1683",
    "%r1692", "%r1693", "%r1702", "%r1703",
    "%r1712", "%r1713", "%r1722", "%r1723",
    "%r1732", "%r1733", "%r1742", "%r1743",
    "%r1752", "%r1753", "%r1762", "%r1763",
    "%r1772", "%r1773", "%r1782", "%r1783",
    "%r1792", "%r1793", "%r1802", "%r1803",
    "%r1812", "%r1813", "%r1822", "%r1823",
    "%r1832", "%r1833", "%r1842", "%r1843",
    "%r1852", "%r1853", "%r1862", "%r1863",
    "%r1872", "%r1873", "%r1882", "%r1883",
    "%r1892", "%r1893", "%r1902", "%r1903",
    "%r1912", "%r1913", "%r1922", "%r1923",
    "%r1932", "%r1933", "%r1942", "%r1943",
    "%r1952", "%r1953", "%r1962", "%r1963",
    "%r1972", "%r1973", "%r1982", "%r1983",
    "%r1992", "%r1993", "%r2002", "%r2003",
    "%r2012", "%r2013", "%r2022", "%r2023",
    "%r2032", "%r2033", "%r2042", "%r2043",
    "%r2052", "%r2053", "%r2062", "%r2063",
    "%r2072", "%r2073", "%r2082", "%r2083",
]


def stages():
    """Oracle stage table: name -> (line after which the store block goes, registers).

    The names carry a `d1` prefix on purpose.  make_ptx.py's swin_debug() falls
    back from a `(layer, stage)` key to the bare `stage` name when it serves a
    layer that has no kernels/rr/swin_<layer>_debug_stages.py, and enc0's
    unkeyed table already owns "a", "p1".."p7", "p5o", "e5", "e5a", "e6" -- a
    dec1 stage called "p1" would silently resolve to enc0's line 4128.  Keeping
    the debug stages in this module (rather than in a *_debug_stages.py that
    would hand the splice to the shared block) also keeps the payload in the
    param_0+8 buffer: the shared swin_debug_block() hardcodes param_0+48, which
    is dec1's *output* plane and is rewritten in full by E25 after every stage
    line."""
    return {
        "d1a": (1485, A1_REGS),
        "d1p": (2490, P1_REGS),
    }

    # d1probe is not a stage: it is consumed by recipe() below, which splices
    # probe_block() instead of debug_block().
    return {
        "d1a": (1485, A1_REGS),
        "d1p": (2490, P1_REGS),
    }


def probe_block(regs):
    """The magic word and the low 32 bits of every `param_0+k` pointer, stored at
    `param_0+48 + 0`, spliced at the end of the entry.

    Two measured facts force this shape.  (a) The launch dump names each argument
    `base<pointer>+<extent>` but never says where inside the dumped block a given
    `param_0+k` pointer lands, so that offset has to be measured, not assumed.
    (b) `param_0+8` is *not* a usable payload buffer for dec1: a 68-byte store
    there at line 1485 kills the run with `HW Exception ... GPU Hang` (measured;
    build /tmp/d4r-dec1/orc-probe, log /tmp/d4r-rr-perf/d1-probe2), which is why
    the older `d1p` oracle stage hangs too.  `param_0+48` is the arena E25 itself
    writes, so it is mapped writable, and splicing at the end of the entry keeps
    E25's stores from overwriting the probe.

    Stored word 0 is 111, word 9 is 222 and word 15 is 333, so the signature is
    unambiguous against ambient buffer content.
    """
    assert len(regs) == 1
    return """// d4r pointer probe
{
.reg .b32 %rp<20>;
.reg .b64 %rq<32>;
ld.param.u64 %rq1, [cuda_dldn_engine_swin_dec1_kernel_param_0+0];
ld.param.u64 %rq2, [cuda_dldn_engine_swin_dec1_kernel_param_0+8];
ld.param.u64 %rq3, [cuda_dldn_engine_swin_dec1_kernel_param_0+24];
ld.param.u64 %rq4, [cuda_dldn_engine_swin_dec1_kernel_param_0+40];
ld.param.u64 %rq5, [cuda_dldn_engine_swin_dec1_kernel_param_0+48];
ld.param.u64 %rq6, [cuda_dldn_engine_swin_dec1_kernel_param_0+80];
ld.param.u64 %rq7, [cuda_dldn_engine_swin_dec1_kernel_param_0+88];
ld.param.u64 %rq8, [cuda_dldn_engine_swin_dec1_kernel_param_0+96];
ld.param.u64 %rq9, [cuda_dldn_engine_swin_dec1_kernel_param_0+104];
ld.param.u64 %rq10, [cuda_dldn_engine_swin_dec1_kernel_param_0+112];
ld.param.u64 %rq11, [cuda_dldn_engine_swin_dec1_kernel_param_0+120];
ld.param.u64 %rq12, [cuda_dldn_engine_swin_dec1_kernel_param_0+128];
cvta.to.global.u64 %rq29, %rq5;
cvta.to.global.u64 %rq17, %rq1;
cvta.to.global.u64 %rq18, %rq3;
cvta.to.global.u64 %rq19, %rq4;
cvta.to.global.u64 %rq20, %rq5;
cvta.to.global.u64 %rq21, %rq6;
cvta.to.global.u64 %rq22, %rq7;
cvta.to.global.u64 %rq23, %rq8;
cvta.to.global.u64 %rq24, %rq9;
cvta.to.global.u64 %rq25, %rq10;
cvta.to.global.u64 %rq26, %rq11;
cvta.to.global.u64 %rq27, %rq12;
mov.u32 %rp1, 111;
cvt.u32.u64 %rp2, %rq17;
cvt.u32.u64 %rp3, %rq29;
cvt.u32.u64 %rp4, %rq18;
cvt.u32.u64 %rp5, %rq19;
cvt.u32.u64 %rp6, %rq20;
cvt.u32.u64 %rp7, %rq21;
cvt.u32.u64 %rp8, %rq22;
cvt.u32.u64 %rp10, %rq23;
cvt.u32.u64 %rp11, %rq24;
cvt.u32.u64 %rp12, %rq25;
cvt.u32.u64 %rp13, %rq26;
cvt.u32.u64 %rp14, %rq27;
mov.u32 %rp9, 222;
mov.u32 %rp15, 333;
mov.u32 %rp16, %r2195;
mov.u32 %rp17, %r2196;
st.global.u32 [%rq29], %rp1;
st.global.u32 [%rq29+4], %rp2;
st.global.u32 [%rq29+8], %rp3;
st.global.u32 [%rq29+12], %rp4;
st.global.u32 [%rq29+16], %rp5;
st.global.u32 [%rq29+20], %rp6;
st.global.u32 [%rq29+24], %rp7;
st.global.u32 [%rq29+28], %rp8;
st.global.u32 [%rq29+32], %rp9;
st.global.u32 [%rq29+36], %rp10;
st.global.u32 [%rq29+40], %rp11;
st.global.u32 [%rq29+44], %rp12;
st.global.u32 [%rq29+48], %rp13;
st.global.u32 [%rq29+52], %rp14;
st.global.u32 [%rq29+56], %rp15;
st.global.u32 [%rq29+60], %rp16;
st.global.u32 [%rq29+64], %rp17;
}
"""


def cut_call():
    """The PTX that replaces the mma region."""
    args = A_REGS + C_REGS
    decl = "\n".join("\t.param .b32 p%d;" % i for i in range(len(args)))
    store = "\n".join("\tst.param.b32 [p%d+0], %s;" % (i, r)
                      for i, r in enumerate(args))
    decl += "\n\t.param .b64 pw;\n\t.param .b64 pv;\n\t.param .b32 ps;"
    store += (f"\n\tst.param.b64 [pw+0], {W_REG};"
              f"\n\tst.param.b64 [pv+0], {V_REG};"
              f"\n\tmov.u32 %rswin0, {SCRATCH};"
              "\n\tst.param.b32 [ps+0], %rswin0;")
    hel = ", ".join(["p%d" % i for i in range(len(args))] + ["pw", "pv", "ps"])
    gets = "\n".join("\tld.shared.b32 %s, [%%rswin1+%d];" % (r, 4 * i)
                     for i, r in enumerate(RET_REGS))
    return f"""{{
.reg .b32 %rswin<2>;
{decl}
{store}
\tcall.uni {CALLEE}, ({hel});
\tmad.lo.s32 %rswin1, %laneid, {RET_STRIDE}, %rswin0;
{gets}
}}"""


def extern():
    """The .extern .func declaration of the callee, matching cut_call()'s arguments."""
    args = ", ".join([".param .b32 d4r_a%d" % i
                      for i in range(len(A_REGS) + len(C_REGS))]
                     + [".param .b64 d4r_w", ".param .b64 d4r_v", ".param .b32 d4r_s"])
    return f".extern .func {CALLEE}\n(\n\t{args}\n)\n;\n"


def recipe():
    """make_ptx.py's recipe for this kernel."""
    stage = os.environ.get("D4R_RRSWIN_DEBUG", "")
    if stage:
        if stage == "d1probe":
            # D4R_RRSWIN_DEBUG=d1probe: measure where each `param_0+k` pointer
            # lands inside the block the launch dump writes, so every later stage
            # extracts its payload at a measured offset instead of an assumed one.
            # NOTE: make_ptx.py rejects any debug line that is not a phase's last
            # mma ("debug line N is not phase 1's last mma"), so the probe cannot be
            # spliced past the last mma at line 38472.  Splicing at 38472 instead
            # would be after every store, which is what this probe wants.
            # (First attempt, with the probe at param_0+8 + 0 / line 1485: run exit
            # 137, "Killed" by the 240 s harness timeout -- GPU hang.)
            return dict(file=MODULE, debug=(38472, [A_REGS[0]]), debug_block=probe_block)
        s = stages().get(stage)
        if s is None:
            if stage == "none":
                # D4R_RRSWIN_DEBUG=none: NVIDIA's own body, unedited -- the
                # baseline the cut and the debug stages are measured against.
                return dict(file=MODULE)
            raise SystemExit(f"swin_dec1_recipe: no debug stage {stage!r}")
        return dict(file=MODULE, debug=s, debug_block=debug_block)
    return dict(file=MODULE, cut_call=CUT_CALL, call=cut_call(), extern=extern())
