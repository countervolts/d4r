#!/usr/bin/env python3
"""Hand-edited PTX for the texture kernels: keep NVIDIA's code up to a cut line, then call a native tail.

usage: make_ptx.py KERNEL OUT.ptx
"""
import importlib.util
import os
import re
import sys
from pathlib import Path

# PTX extracted from nvngx_dlss.dll by kernels/tools/extract_dlss_ptx.py (kernels/build.sh does this)
PTX_DIR = Path(os.environ.get("D4R_DLSS_PTX_DIR", Path(__file__).resolve().parent.parent / "extracted" / "ptx"))

# Ray Reconstruction enc0 mirror cut: PTX keeps the prologue (textures, input staging, the A/B/bias
# register setup up to line 3963) and the surface epilogue (from line 44528); the 536 mma of the 22
# phases (lines 3964-44527) become one call to the native body, which is handed the six staged A
# fragments, the four bias seeds, the weight image and the parameter block. Phase 22's eight mma each
# write a *pair* of b32 (four f16 per lane), and the epilogue's `selp` chains (stmts 15595-15609,
# 15643-15676, 15718-15720) select between the halves on `laneid & 3`, so both halves of every pair
# must cross the cut -- returning only the even halves (the previous list) left the odd ones undefined
# in the spliced PTX and made byte-exact output impossible.
#
# The eighteen accumulators come back through shared memory, not `call.uni`'s return parameters.
# ZLUDA lowers those to the *function's* LLVM return type (llvm_zluda/ptx/src/pass/llvm/emit.rs,
# emit_call: one value, or an `get_or_create_struct_type` struct for several, distributed with
# LLVMBuildExtractValue). There is no .param-space object behind them and no pointer is ever formed,
# so the callee sees 0x0 in all eighteen slots and stores through them fault the launch; an 18 x i32
# C++ struct return lowers to an sret pointer, which is a different LLVM signature and will not link.
#
# So the callee is handed the address of `input_tensor` (3200 bytes of .shared, dead after the cut --
# its last ld.shared is line 3938 and no register derived from its base is read past it) and writes
# the eighteen values at `base + 72*laneid + 4*i`; the spliced PTX reads them back with eighteen
# ld.shared.b32. 72 * 32 lanes = 2304 bytes, inside the 3200. Per-block LDS is what makes this safe:
# a global scratch slot would be shared by all 14973 blocks, and sizing it per block and lane
# (14973 * 32 * 72 = 34.5 MB) would not fit past the 45,219,840 arena bytes the kernel writes either.
# enc0_tail already uses this shape (d4r_enc0_tail takes the shared base as a plain b32 argument).
SWIN_A_REGS = ["%r120", "%r136", "%r128", "%r144", "%r152", "%r168", "%r160", "%r176",
               "%r184", "%r200", "%r192", "%r208", "%r216", "%r232", "%r224", "%r240",
               "%r246", "%r257", "%r253", "%r1042", "%r1079", "%r1080", "%r1081", "%r1082"]
SWIN_BIAS_REGS = ["%r1086", "%r1096", "%r1106", "%r1116"]
SWIN_RET_REGS = ["%r17597", "%r18203"] + sum(
    ([f"%r{d}", f"%r{d + 1}"] for d in range(17634, 17706, 10)), [])

# the .shared scratch the accumulators cross the cut through: 18 b32 per lane, 72 bytes apart
SWIN_RET_SCRATCH = "_ZZ33cuda_dldn_engine_swin_enc0_kernel35DldnEngineFusedSwinEnc0ParamsStructE12input_tensor"
SWIN_RET_STRIDE = 4 * len(SWIN_RET_REGS)

SWIN_P2_A_REGS = ["%r1207", "%r1208", "%r1209", "%r1210", "%r1247", "%r1248", "%r1249", "%r1250",
                 "%r1287", "%r1288", "%r1289", "%r1290", "%r1327", "%r1328", "%r1329", "%r1330",
                 "%r1367", "%r1368", "%r1369", "%r1370", "%r1407", "%r1408", "%r1409", "%r1410"]

def swin_cut_call():
    """The native body is handed the staged A fragments, the four bias seeds, the weight image, the
    parameter block and the address of the shared scratch; it writes the eighteen accumulators there
    and this splice reads them back into the epilogue's registers."""
    args = SWIN_A_REGS + SWIN_BIAS_REGS
    decl = "\n".join(f"\t.param .b32 p{i};" for i in range(len(args)))
    store_args = "\n".join(f"\tst.param.b32 [p{i}+0], {r};" for i, r in enumerate(args))
    decl += "\n\t.param .b64 pw;\n\t.param .b64 pv;\n\t.param .b32 ps;"
    store_args += ("\n\tst.param.b64 [pw+0], %rd367;\n\tst.param.b64 [pv+0], %rd368;"
                   f"\n\tmov.u32 %rswin0, {SWIN_RET_SCRATCH};"
                   "\n\tst.param.b32 [ps+0], %rswin0;")
    hel = ", ".join([f"p{i}" for i in range(len(args))] + ["pw", "pv", "ps"])
    gets = "\n".join(f"\tld.shared.b32 {r}, [%rswin1+{4 * i}];"
                     for i, r in enumerate(SWIN_RET_REGS))
    return f"""{{
.reg .b32 %rswin<2>;
{decl}
{store_args}
\tcall.uni d4r_swin_enc0, ({hel});
\tmad.lo.s32 %rswin1, %laneid, {SWIN_RET_STRIDE}, %rswin0;
{gets}
}}"""


def swin_cut_extern():
    args = ", ".join([f".param .b32 d4r_a{i}" for i in range(len(SWIN_A_REGS) + len(SWIN_BIAS_REGS))]
                     + [".param .b64 d4r_w", ".param .b64 d4r_v", ".param .b32 d4r_s"])
    return f".extern .func d4r_swin_enc0\n(\n\t{args}\n)\n;\n"




ENC0_TAIL = """{{
	.param .b32 q0;
	st.param.b32 [q0+0], {smem};
	.param .b64 q1;
	st.param.b64 [q1+0], {w};
	.param .b64 q2;
	st.param.b64 [q2+0], {out};
	.param .b32 q3;
	st.param.b32 [q3+0], {dimx};
	.param .b32 q4;
	st.param.b32 [q4+0], {dimy};
	.param .b32 q5;
	st.param.b32 [q5+0], {ox};
	.param .b32 q6;
	st.param.b32 [q6+0], {oy};
	call.uni d4r_enc0_tail, (q0, q1, q2, q3, q4, q5, q6);
}}
ret;

}}
"""

SWIN_EXTERN = """.extern .func d4r_swin_enc0
(
	.param .b64 d4r_e0, .param .b32 d4r_e1, .param .b32 d4r_e2
)
;
"""

# NVIDIA's own entry takes the address of its parameter block (`mov.b64 %rd368, <param_0>`), so the
# native body is handed that pointer plus the two shared arrays' addresses. The registers are ones the
# entry already declares; the body between them is deleted.
SWIN_STUB = """mov.b64 %rd376, cuda_dldn_engine_swin_enc0_kernel_param_0;
mov.u32 %r18228, _ZZ33cuda_dldn_engine_swin_enc0_kernel35DldnEngineFusedSwinEnc0ParamsStructE11input_noise;
mov.u32 %r18227, _ZZ33cuda_dldn_engine_swin_enc0_kernel35DldnEngineFusedSwinEnc0ParamsStructE12input_tensor;
{
	.param .b64 q0;
	st.param.b64 [q0+0], %rd376;
	.param .b32 q1;
	st.param.b32 [q1+0], %r18228;
	.param .b32 q2;
	st.param.b32 [q2+0], %r18227;
	call.uni d4r_swin_enc0, (q0, q1, q2);
}
ret;"""

KERNELS = {
    # enc0: cut after the bar.warp.sync that ends the smem staging of the GEMM A operand
    "rrlite_enc0_4x4_mvhi_hdr_folded": dict(
        file="dlss-0185-00.ptx",
        cut_after=2548,
        cut_check="bar.warp.sync -1;",
        sust=True,
        extern=""".extern .func d4r_enc0_tail
(
	.param .b32 d4r_a0, .param .b64 d4r_a1, .param .b64 d4r_a2, .param .b32 d4r_a3,
	.param .b32 d4r_a4, .param .b32 d4r_a5, .param .b32 d4r_a6
)
;
""",
        tail=ENC0_TAIL.format(smem="%r259", w="%rd106", out="%rd78", dimx="%r1066", dimy="%r1067", ox="%r1230", oy="%r1233"),
    ),
    # dec0: the GEMM (from the B fragment loads to the e4m3 staging stores) becomes a native call; the
    # A loads before it are left dead. %r560 (shared base) is the only register of the range used later.
    "rrlite_dec0_4x4_folded": dict(
        file="dlss-0195-00.ptx",
        delete=(139, 587),
        delete_check=("mov.u32 %r17, %laneid;", "st.shared.v2.u16 [%r570+1168], {%rs63, %rs64};"),
        sust=True,
        extern=""".extern .func d4r_dec0_head
(
	.param .b32 d4r_a0, .param .b64 d4r_a1, .param .b64 d4r_a2, .param .b32 d4r_a3,
	.param .b32 d4r_a4, .param .b32 d4r_a5, .param .b32 d4r_a6
)
;
""",
        insert="""mov.u32 %r560, _ZZ22rrlite_dec0_4x4_foldedN6RRLite18Decoder0ParametersEE4smem;
{
	.param .b32 q0;
	st.param.b32 [q0+0], %r560;
	.param .b64 q1;
	st.param.b64 [q1+0], %rd22;
	.param .b64 q2;
	st.param.b64 [q2+0], %rd24;
	.param .b32 q3;
	st.param.b32 [q3+0], %r475;
	.param .b32 q4;
	st.param.b32 [q4+0], %r476;
	.param .b32 q5;
	st.param.b32 [q5+0], %r3;
	.param .b32 q6;
	st.param.b32 [q6+0], %r4;
	call.uni d4r_dec0_head, (q0, q1, q2, q3, q4, q5, q6);
}""",
    ),
    # Ray Reconstruction denoiser Swin layers (kernels/rr/rrswin_*.hip): the whole entry body of the
    # module's Shwin entry is replaced by a call into the native implementation, linked in as bitcode.
    # PREPASS_ENTRYPOINT_NAME, the module's other entry, is left untouched.
    "cuda_dldn_engine_swin_enc0_kernel": dict(
        file="0013-PREPASS_ENTRYPOINT_NAME.ptx",
        replace_body=True,
        extern=SWIN_EXTERN,
        stub=SWIN_STUB,
    ),
    # post and downsample: only the formatted surface stores become native (inline format conversion
    # instead of ZLUDA's per-channel surface_formatted_store_bits calls)
    "rrlite_post_3_2_mvhi_hdr_folded": dict(file="dlss-0176-00.ptx", sust=True, extern=""),
    "rrlite_downsample_kernel_static_hdr": dict(file="dlss-0196-00.ptx", sust=True, extern=""),
    # preset K (DLSS 4): the output kernel's 17 surface stores (13 formatted, 4 raw 32-bit)
    "hiluma_engine_output_depthinv_mvhi_hdr_max_v2_rel": dict(file="dlss-0078-00.ptx", sust=True, rz_round=True, extern=""),
    "hiluma_engine_input_depthinv_mvhi_hdr_v2_rel": dict(file="dlss-0062-00.ptx", sust=True, rz_round=True, extern=""),
}


SUST_EXTERN = """.extern .func d4r_sust_v2b16
(
	.param .b64 d4r_s0, .param .b32 d4r_s1, .param .b32 d4r_s2, .param .b16 d4r_s3, .param .b16 d4r_s4
)
;
"""
SUST4_EXTERN = """.extern .func d4r_sust_v4b16
(
	.param .b64 d4r_h0, .param .b32 d4r_h1, .param .b32 d4r_h2, .param .b16 d4r_h3,
	.param .b16 d4r_h4, .param .b16 d4r_h5, .param .b16 d4r_h6
)
;
"""
SUSTP_EXTERN = """.extern .func d4r_sust_p_v4b32
(
	.param .b64 d4r_t0, .param .b32 d4r_t1, .param .b32 d4r_t2, .param .b32 d4r_t3, .param .b32 d4r_t4,
	.param .b32 d4r_t5, .param .b32 d4r_t6
)
;
"""
SUSTP_RE = re.compile(r"^sust\.p\.2d\.v4\.b32\.zero \[(%rd\d+), \{(%r\d+),(%r\d+)\}\], \{(%f\d+),(%f\d+),(%f\d+),(%f\d+)\};$")
SUSTB32_EXTERN = """.extern .func d4r_sust_b32
(
	.param .b64 d4r_w0, .param .b32 d4r_w1, .param .b32 d4r_w2, .param .b32 d4r_w3
)
;
"""
SUSTB32_RE = re.compile(r"^sust\.b\.2d\.b32\.zero \[(%rd\d+), \{(%r\d+),(%r\d+)\}\], \{(%r\d+)\};$")
SUST_RE = re.compile(r"^sust\.b\.2d\.v2\.b16\.zero \[(%rd\d+), \{(%r\d+),(%r\d+)\}\], \{(%rs\d+),(%rs\d+)\};$")
SUST4_RE = re.compile(r"^sust\.b\.2d\.v4\.b16\.zero \[(%rd\d+), \{(%r\d+),\s*(%r\d+)\}\], \{(%rs\d+),\s*(%rs\d+),\s*(%rs\d+),\s*(%rs\d+)\};$")


SUST_KINDS = set(os.environ.get("D4R_SUST_KINDS", "b16,p,b32").split(","))


def replace_sust(lines):
    out, n = [], {"b16": 0, "v4b16": 0, "p": 0, "b32": 0}
    for line in lines:
        mh = SUST4_RE.match(line.strip()) if "b16" in SUST_KINDS else None
        if mh:
            s, x, y, *v = mh.groups()
            st = "\n".join(f"\t.param .b16 v{i};\n\tst.param.b16 [v{i}+0], {r};" for i, r in enumerate(v))
            out.append(f"""{{
	.param .b64 u0;
	st.param.b64 [u0+0], {s};
	.param .b32 u1;
	st.param.b32 [u1+0], {x};
	.param .b32 u2;
	st.param.b32 [u2+0], {y};
{st}
	call d4r_sust_v4b16, (u0, u1, u2, v0, v1, v2, v3);
}}""")
            n["v4b16"] += 1
            continue
        mb = SUSTB32_RE.match(line.strip()) if "b32" in SUST_KINDS else None
        if mb:
            s, x, y, v = mb.groups()
            out.append(f"""{{
	.param .b64 u0;
	st.param.b64 [u0+0], {s};
	.param .b32 u1;
	st.param.b32 [u1+0], {x};
	.param .b32 u2;
	st.param.b32 [u2+0], {y};
	.param .b32 u3;
	st.param.b32 [u3+0], {v};
	call d4r_sust_b32, (u0, u1, u2, u3);
}}""")
            n["b32"] += 1
            continue
        mp = SUSTP_RE.match(line.strip()) if "p" in SUST_KINDS else None
        if mp:
            s, x, y, *v = mp.groups()
            st = "\n".join(f"\t.param .b32 v{i};\n\tst.param.b32 [v{i}+0], {r};" for i, r in enumerate(v))
            out.append(f"""{{
	.param .b64 u0;
	st.param.b64 [u0+0], {s};
	.param .b32 u1;
	st.param.b32 [u1+0], {x};
	.param .b32 u2;
	st.param.b32 [u2+0], {y};
{st}
	call d4r_sust_p_v4b32, (u0, u1, u2, v0, v1, v2, v3);
}}""")
            n["p"] += 1
            continue
        m = SUST_RE.match(line.strip())
        if not m:
            out.append(line)
            continue
        s, x, y, a, b = m.groups()
        out.append(f"""{{
	.param .b64 u0;
	st.param.b64 [u0+0], {s};
	.param .b32 u1;
	st.param.b32 [u1+0], {x};
	.param .b32 u2;
	st.param.b32 [u2+0], {y};
	.param .b16 u3;
	st.param.b16 [u3+0], {a};
	.param .b16 u4;
	st.param.b16 [u4+0], {b};
	call d4r_sust_v2b16, (u0, u1, u2, u3, u4);
}}""")
        n["b16"] += 1
    return out, n


RZ_RE = re.compile(r"^add\.rz\.ftz\.f32 (%f\d+), (%f\d+), (%f\d+);$")
CVT_RE = re.compile(r"^cvt\.rzi\.f32\.f32 (%f\d+), (%f\d+);$")


def replace_rz_round(lines):
    """roundf idiom add.rz(x, copysign(0.5, x)) + cvt.rzi -> exact round-half-away in default rounding:
    t = trunc(x); r = |x - t| >= 0.5 ? t + 2 * copysign(0.5, x) : t (x - t is exact). Lets ZLUDA compile the
    kernel without the strict-FP (constrained) mode that one explicit rounding mode forces."""
    out, n, i = [], 0, 0
    while i < len(lines):
        m = RZ_RE.match(lines[i].strip())
        if m:
            d, x, sgn = m.groups()
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            c = CVT_RE.match(lines[j].strip()) if j < len(lines) else None
            if c and c.group(2) == d:
                r = c.group(1)
                out += [f"cvt.rzi.f32.f32 %rq{4*n}, {x};", f"sub.f32 %rq{4*n+1}, {x}, %rq{4*n};",
                        f"abs.f32 %rq{4*n+1}, %rq{4*n+1};", f"setp.ge.f32 %pq{n}, %rq{4*n+1}, 0f3F000000;",
                        f"add.f32 %rq{4*n+2}, {sgn}, {sgn};", f"add.f32 %rq{4*n+3}, %rq{4*n}, %rq{4*n+2};",
                        f"selp.f32 {r}, %rq{4*n+3}, %rq{4*n}, %pq{n};"]
                n += 1
                i = j + 1
                continue
            raise SystemExit(f"add.rz at line {i} not followed by its cvt.rzi")
        out.append(lines[i])
        i += 1
    return out, n


ENC0_REF = "rrlite_enc0_4x4_mvhi_hdr_folded"
# the other flag combinations of the kernels whose only native part is the surface-store rewrite
SUST_ONLY_RE = re.compile(r"^(hiluma_engine_output_depth(inv|reg)_mv(hi|lo)_(hdr|ldr)(_max)?_v[12]_rel|"
                          r"rrlite_post_3_[12]_mv(hi|lo)_(hdr|ldr)(_folded)?|"
                          r"rrlite_enc0_4x4_mv(hi|lo)_(hdr|ldr)|rrlite_dec0_4x4|"
                          r"rrlite_downsample_kernel_(static|dynamic)_(hdr|ldr)|"
                          r"cuda_dldn_engine_hkpn_output_kernel_transformer(_diamond_wallaby)?)$")
ENC0_RE = re.compile(r"^rrlite_enc0_4x4_mv(hi|lo)_(hdr|ldr)_folded$")


def read_lines(path):
    return path.read_text().replace("\r\n", "\n").split("\n")


def enc0_registers(lines):
    """Cut line (1-based) and the tail call's registers of an enc0 variant. The code after the third
    bar.warp.sync is the same GEMM in every flag combination, only register numbers differ: the shared
    base and weights pointer are read off by aligning that block with the reference variant; the output
    pointer and grid size are the kernel's loads of parameters +40 and +8; the block offsets are the two
    values halved (shr 31 / add) by the epilogue."""
    syncs = [i for i, l in enumerate(lines) if l.strip() == "bar.warp.sync -1;"]
    if len(syncs) != 3:
        sys.exit(f"enc0: expected 3 bar.warp.sync, found {len(syncs)}")
    cut = syncs[2] + 1
    before, after = lines[:cut], lines[cut:]
    out = [m.group(1) for m in (re.match(r"^ld\.param\.u64 (%rd\d+), \[%rd\d+\+40\];$", l.strip()) for l in before) if m]
    dims = [m.groups() for m in (re.match(r"^ld\.param\.v2\.u32 \{(%r\d+), (%r\d+)\}, \[%rd\d+\+8\];$", l.strip()) for l in before) if m]
    halves = []
    for a, b in zip(after, after[1:]):
        m = re.match(r"^shr\.u32 (%r\d+), (%r\d+), 31;$", a.strip())
        if m and re.match(rf"^add\.s32 %r\d+, {re.escape(m.group(2))}, {re.escape(m.group(1))};$", b.strip()):
            halves.append(m.group(2))
    if len(out) != 1 or len(dims) != 1 or len(halves) < 2:
        sys.exit(f"enc0: parameter loads or epilogue not found ({out}, {dims}, {halves[:2]})")
    return cut, dict(out=out[0], dimx=dims[0][0], dimy=dims[0][1], oy=halves[0], ox=halves[1])


REG_RE = re.compile(r"%(?:rd|rs|r|fd|f|p)\d+")


def enc0_variant(name):
    ref = read_lines(module_file(ENC0_REF, KERNELS[ENC0_REF]))
    path = module_file(name, {"file": ""})
    lines = read_lines(path)
    rcut, rregs = enc0_registers(ref)
    k = KERNELS[ENC0_REF]
    known = dict(smem="%r259", w="%rd106", out="%rd78", dimx="%r1066", dimy="%r1067", ox="%r1230", oy="%r1233")
    if rcut != k["cut_after"] or any(rregs[key] != known[key] for key in rregs):
        sys.exit(f"enc0: the register search does not reproduce {ENC0_REF} ({rcut}, {rregs})")
    cut, regs = enc0_registers(lines)
    # align the GEMM block (up to the reference's first closing brace) and map registers
    n = next(i for i, l in enumerate(ref[rcut:]) if l.startswith("}"))
    mapping = {}
    for a, b in zip(ref[rcut:rcut + n], lines[cut:cut + n]):
        if REG_RE.sub("R", a) != REG_RE.sub("R", b):
            sys.exit(f"enc0: {name} differs from {ENC0_REF} after the cut: {a!r} / {b!r}")
        for x, y in zip(REG_RE.findall(a), REG_RE.findall(b)):
            if mapping.setdefault(x, y) != y:
                sys.exit(f"enc0: inconsistent register mapping {x} -> {mapping[x]} / {y}")
    regs.update(smem=mapping[known["smem"]], w=mapping[known["w"]])
    return dict(file=path.name, cut_after=cut, cut_check="bar.warp.sync -1;", sust=True, extern=k["extern"],
                tail=ENC0_TAIL.format(**regs))


SWIN_DEBUG_BASE = 46000000  # bytes into the plane arena (arg048) the kernel never writes
SWIN_DEBUG_SLOT = 8192      # the per-block slot floor: 32 lanes x 64 registers x 4 bytes


def swin_debug_slot(nregs):
    """Per-block slot for a stage of NREGS registers: 32 lanes x NREGS x 4 bytes, at least
    SWIN_DEBUG_SLOT, rounded up to a power of two so the splice's `shl` can form block * slot."""
    want = max(SWIN_DEBUG_SLOT, 32 * nregs * 4)
    return 1 << (want - 1).bit_length()


def swin_p5_regs():
    """Phase 5's 48 mma D pairs in the mma statement order: m-tile major, n-tile minor, the row
    pair (row g, row g+8) each mma writes. 2566 + 120*mi + 10*tt is the first mma of group mi."""
    return sum(([f"%r{2566 + 120 * mi + 10 * tt + rh}" for rh in (0, 1)]
                for mi in range(4) for tt in range(12)), [])



def swin_p5_operand_regs():
    """Phase 5's 48 mma C pairs (the position bias: 24 tiles of four words, tile T at 2390 + 4*T) and
    the twelve B fragment pairs (the e4m3 keys, 2932 + 10*tt) -- the operands the 48 mma read."""
    return (sum(([f"%r{2390 + 4 * t + w}" for w in range(4)] for t in range(24)), [])
            + sum(([f"%r{2932 + 10 * t}", f"%r{2933 + 10 * t}"] for t in range(12)), []))


def swin_e5_regs():
    """Phase 6's twelve A fragments (four b32 each, mma operand order) and twelve B fragment pairs:
    the softmax probabilities and the transposed V that E5 packs."""
    return (sum(([f"%r{5884 + 120 * (a // 3) + 20 * (a % 3) + o}" for o in range(4)]
                for a in range(12)), [])
            + sum(([f"%r{6178 + 10 * b}", f"%r{6179 + 10 * b}"] for b in range(12)), []))


def swin_e5a_regs():
    """Phase 6's twelve A fragments alone: the softmax probabilities E5 packs."""
    return sum(([f"%r{5884 + 120 * (a // 3) + 20 * (a % 3) + o}" for o in range(4)]
                for a in range(12)), [])


def swin_p6_regs():
    """Phase 6's sixteen mma D pairs, each the third (last) k-step of its chain: m-tile major,
    n-tile minor, the row pair (row g, row g+8)."""
    return sum(([f"%r{5852 + 120 * mi + 10 * nt + 40 * (nt >> 1) + rh}" for rh in (0, 1)]
                for mi in range(4) for nt in range(4)), [])


def swin_e6_regs():
    """Phase 7's sixteen A registers (four fragments of four) and its thirty-two C registers (the
    residual plus the per-column table), m-tile major and n-tile minor."""
    return (sum(([f"%r{6366 + 40 * mi + o}" for o in range(4)] for mi in range(4)), [])
            + sum(([f"%r{6342 + 40 * mi + 10 * nt + rh}" for rh in (0, 1)]
                   for mi in range(4) for nt in range(4)), []))


def swin_p7_regs():
    """Phase 7's sixteen mma D pairs, m-tile major and n-tile minor."""
    return sum(([f"%r{6334 + 10 * (4 * mi + nt) + rh}" for rh in (0, 1)]
                for mi in range(4) for nt in range(4)), [])


# a debug stage: (line after which to insert, registers to store); the per-block slot follows from
# the register count, so a stage is reproducible from its tuple alone
SWIN_DEBUG_STAGES = {
    # the staged block input: the 24 A registers in the first mma's operand order plus the four bias seeds
    "a": (3962, ["%r120", "%r136", "%r128", "%r144", "%r152", "%r168", "%r160", "%r176",
                 "%r184", "%r200", "%r192", "%r208", "%r216", "%r232", "%r224", "%r240",
                 "%r246", "%r257", "%r253", "%r1042", "%r1079", "%r1080", "%r1081", "%r1082",
                 "%r1086", "%r1096", "%r1106", "%r1116"]),
    # phase 1's 24 accumulator register pairs (m-tile major, n-tile minor)
    "p1": (4128, ["%r877", "%r878", "%r887", "%r888", "%r897", "%r898", "%r907", "%r908",
                  "%r917", "%r918", "%r927", "%r928", "%r937", "%r938", "%r947", "%r948",
                  "%r957", "%r958", "%r967", "%r968", "%r977", "%r978", "%r987", "%r988",
                  "%r997", "%r998", "%r1007", "%r1008", "%r1017", "%r1018", "%r1027", "%r1028",
                  "%r1037", "%r1038", "%r1047", "%r1048", "%r1057", "%r1058", "%r1067", "%r1068",
                  "%r1077", "%r1078", "%r1087", "%r1088", "%r1097", "%r1098", "%r1107", "%r1108"]),
    # phase 2's accumulator pairs: its last mma's `;` is line 4489
    "p2": (4489, sum(([f"%r{d + 10 * i + j}" for i in range(4) for j in range(2)]
                     for d in range(1175, 1406, 40)), [])),
    # phase 3's (the fused Q projection); its last mma ends on line 7840
    "p3": (7840, sum(([f"%r{d + 10 * i + j}" for i in range(4) for j in range(2)]
                     for d in range(1900, 2131, 40)), [])),
    # phase 4's (V); its last mma ends on line 8021
    "p4": (8021, sum(([f"%r{d + 10 * i + j}" for i in range(4) for j in range(2)]
                     for d in range(2149, 2380, 40)), [])),
    # phase 2's A operands as the epilogue packed them (the last bias load is line 4323)
    "a2": (4323, SWIN_P2_A_REGS),
    # phase 5 (the score GEMM, Q·Kᵀ with the position bias); its last mma ends on line 8741
    "p5": (8741, swin_p5_regs()),
    # phase 5's C (bias) and B (key) operand registers, as the mma reads them
    "p5o": (8741, swin_p5_operand_regs()),
    # E5's output: phase 6's A and B fragments; the last pack is line 18883
    "e5": (18883, swin_e5_regs()),
    # E5's probabilities alone: the fragment's B side is a movmatrix of phase 4's D, which the
    # scalar body only reaches through phase 6's accumulators, so it is captured by p6 instead
    "e5a": (18883, swin_e5a_regs()),
    # phase 6 (P·V); its last mma ends on line 19220
    "p6": (19220, swin_p6_regs()),
    # E6's output: phase 7's A fragments and C seeds; the last pack is line 19643
    "e6": (19643, swin_e6_regs()),
    # phase 7 (the output projection); its last mma ends on line 19756
    "p7": (19756, swin_p7_regs()),
}

# Per-layer extra debug stages, so a layer can add its own without editing this file:
# kernels/rr/swin_<layer>_debug_stages.py exposing stages() -> {name: (line, [regs])}. They are keyed
# by (layer, stage) because two layers can use the same stage name; the entries above stay unkeyed
# and belong to enc0.
for _extra in sorted((Path(__file__).resolve().parent.parent / "rr").glob("swin_*_debug_stages.py")):
    _layer = _extra.stem[len("swin_"):-len("_debug_stages")]
    _spec = importlib.util.spec_from_file_location(_extra.stem, _extra)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    for _stage, _where in _mod.stages().items():
        SWIN_DEBUG_STAGES[(_layer, _stage)] = _where




def swin_debug_block(regs, entry="cuda_dldn_engine_swin_enc0_kernel", base=None):
    """PTX that stores REGS (u32 each) to the output buffer + base + block*slot + lane*len(regs)*4.

    The region must be past the bytes the kernel itself writes; enc0's plane arena is 45,219,840 bytes
    and a layer whose output buffer is smaller sets D4R_RRSWIN_DEBUG_BASE lower (both sides of the
    comparison read the same value). Blocks above 63 are skipped so the first 64 blocks (one warp
    each) land in distinct slots. Works on either side of the splice: it is inserted into NVIDIA's own
    PTX for the reference and reproduced by the native body's debug flag.
    """
    if base is None:
        base = int(os.environ.get("D4R_RRSWIN_DEBUG_BASE", SWIN_DEBUG_BASE))
    slot = swin_debug_slot(len(regs))
    stores = "\n".join(
        f"@%pdbg1 st.global.v4.u32 [%rddbg3+{i * 16}], {{{', '.join(regs[i * 4:i * 4 + 4])}}};"
        for i in range((len(regs) + 3) // 4))
    return f"""// d4r debug stage
{{
.reg .pred %pdbg<2>;
.reg .b32 %rdbg<8>;
.reg .b64 %rddbg<4>;
ld.param.u64 %rddbg1, [{entry}_param_0+48];
mov.u32 %rdbg1, %ctaid.x;
mov.u32 %rdbg2, %ctaid.y;
mad.lo.s32 %rdbg3, %rdbg2, 161, %rdbg1;
setp.lt.u32 %pdbg1, %rdbg3, 64;
shl.b32 %rdbg4, %rdbg3, {slot.bit_length() - 1};
mov.u32 %rdbg5, %laneid;
mad.lo.s32 %rdbg6, %rdbg5, {len(regs) * 4}, %rdbg4;
add.s32 %rdbg7, %rdbg6, {base};
cvt.u64.u32 %rddbg2, %rdbg7;
add.s64 %rddbg3, %rddbg1, %rddbg2;
{stores}
}}
"""


def swin_debug(name):
    """Debug recipe: keep NVIDIA's whole body and store one stage's accumulator registers."""
    stage = os.environ.get("D4R_RRSWIN_DEBUG", "")
    match = re.fullmatch(r"cuda_dldn_engine_swin_(\w+)_kernel", name)
    if not stage or match is None:
        return None
    where = SWIN_DEBUG_STAGES.get((match.group(1), stage))
    if where is None:
        where = SWIN_DEBUG_STAGES.get(stage)
    if where is None:
        return None
    return dict(file=module_file(name, {"file": ""}).name, debug=where,
                debug_block=lambda regs: swin_debug_block(regs, name))


def kernel_spec(name):
    d = swin_debug(name)
    if d:
        return d
    # Per-layer recipes: kernels/rr/swin_<layer>_recipe.py owns one denoiser layer's splice, so the
    # ten layers can be written in parallel without sharing this file. Falls through to the built-in
    # enc0 entry below when the module is absent.
    swin = re.fullmatch(r"cuda_dldn_engine_swin_(\w+)_kernel", name)
    if swin is not None:
        module_path = Path(__file__).resolve().parent.parent / "rr" / f"swin_{swin.group(1)}_recipe.py"
        if module_path.is_file():
            spec = importlib.util.spec_from_file_location(f"swin_{swin.group(1)}_recipe", module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.recipe()
    if name == "cuda_dldn_engine_swin_enc0_kernel" and os.environ.get("D4R_RRSWIN_CUT"):
        return dict(file="0013-PREPASS_ENTRYPOINT_NAME.ptx", cut_call=(3964, 44528), extern=swin_cut_extern())
    if name in KERNELS:
        return KERNELS[name]
    if ENC0_RE.match(name):
        return enc0_variant(name)
    if SUST_ONLY_RE.match(name):
        return dict(file="", sust=True, extern="", rz_round=name.startswith("hiluma_engine_output"))
    sys.exit(f"make_ptx: no recipe for {name}")


def module_file(name, k):
    """The extracted PTX module that defines NAME: the 310.7 file number, else any module that does
    (other DLSS versions number their modules differently)."""
    entry = re.compile(rf"\.entry\s+{re.escape(name)}\s*\(")
    known = PTX_DIR / k["file"]
    if k["file"] and known.is_file() and entry.search(known.read_text()):
        return known
    for path in sorted(PTX_DIR.glob("*.ptx")):
        if entry.search(path.read_text()):
            return path
    sys.exit(f"no PTX module in {PTX_DIR} defines {name}")


def main():
    name, out = sys.argv[1], Path(sys.argv[2])
    k = kernel_spec(name)
    lines = read_lines(module_file(name, k))
    if "cut_after" in k:
        cut = k["cut_after"]
        if lines[cut - 1].strip() != k["cut_check"]:
            sys.exit(f"line {cut} is {lines[cut - 1]!r}, expected {k['cut_check']!r}")
        head, tail = lines[:cut], k["tail"].split("\n")
    elif "delete" in k:
        a, b = k["delete"]
        if (lines[a - 1].strip(), lines[b - 1].strip()) != k["delete_check"]:
            sys.exit(f"lines {a}/{b} are {lines[a - 1]!r}/{lines[b - 1]!r}, expected {k['delete_check']}")
        head, tail = lines[:a - 1] + k["insert"].split("\n") + lines[b:], []
    elif "cut_call" in k:
        before, after = k["cut_call"]
        if not lines[before - 1].startswith("mma.sync.aligned.m16n8k32"):
            sys.exit(f"{name}: cut line {before} is not an mma: {lines[before - 1]!r}")
        # the line the cut ends on must be the last line of an mma statement; which mma that is
        # depends on the layer, so this only checks the shape, not enc0's registers
        if not lines[after - 2].rstrip().endswith(";"):
            sys.exit(f"{name}: cut end {after} is not the end of an mma statement: {lines[after - 2]!r}")
        head = lines[:before - 1] + k.get("call", swin_cut_call()).split("\n") + lines[after - 1:]
        tail = []
    elif "debug" in k:
        after, regs = k["debug"]
        window = "\n".join(lines[max(0, after - 4000):after])
        prev = lines[after - 1].rstrip()
        if regs[0] not in window or not (prev.endswith(";") or prev.endswith("//") or not prev):
            sys.exit(f"{name}: debug line {after} is not phase 1's last mma: {lines[after - 1]!r}")
        head, tail = lines[:after] + k.get("debug_block", swin_debug_block)(regs).split("\n") + lines[after:], []
    elif "replace_body" in k:
        e = next(i for i, l in enumerate(lines) if l.startswith(f".visible .entry {name}("))
        b0 = next(i for i in range(e, len(lines)) if lines[i].strip() == "{")
        j = b0 + 1
        while j < len(lines) and (not lines[j].strip()
                                 or lines[j].lstrip().startswith((".reg", ".shared", ".param", ".pragma", "//"))):
            j += 1
        if not lines[j - 1].lstrip().startswith("//") and lines[j - 1].strip() != "":
            sys.exit(f"{name}: could not find the end of the entry declarations (line {j} is {lines[j - 1]!r})")
        depth, b = 0, None
        for i in range(b0, len(lines)):
            depth += lines[i].count("{") - lines[i].count("}")
            if i > b0 and depth == 0:
                b = i
                break
        if b is None or lines[b].strip() != "}":
            sys.exit(f"{name}: entry closing brace not found")
        head, tail = lines[:j] + k["stub"].split("\n") + [""] + lines[b:], []
    else:
        head, tail = lines, []
    entry = next(i for i, l in enumerate(head) if l.startswith(".visible .entry"))
    body, n = replace_sust(head[entry:]) if k.get("sust") else (head[entry:], {"b16": 0, "p": 0, "b32": 0})
    if k.get("rz_round"):
        body, nrz = replace_rz_round(body)
        # temporaries for the rewrite, declared after the kernel's own .reg lines
        at = next(i for i, l in enumerate(body) if l.startswith(".reg"))
        body = body[:at] + [f".reg .f32 %rq<{4 * nrz + 1}>;", f".reg .pred %pq<{nrz + 1}>;"] + body[at:]
        n["rz"] = nrz
    extern = k.get("extern", "") + (SUST_EXTERN if n["b16"] else "") + (SUST4_EXTERN if n.get("v4b16") else "") + (SUSTP_EXTERN if n["p"] else "") + (SUSTB32_EXTERN if n["b32"] else "")
    text = "\n".join(head[:entry]) + "\n" + extern + "\n".join(body + tail)
    out.write_text(text)
    print(f"{out}: {len(text.splitlines())} lines, surface stores replaced: {n}")


if __name__ == "__main__":
    main()
