"""Enc0 cut recipes; isolated phase 8 preserves NVIDIA's other phases."""
import os
import sys
from pathlib import Path


def recipe():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tex"))
    import make_ptx

    if not os.environ.get("D4R_RRSWIN_PHASE8"):
        if os.environ.get("D4R_RRSWIN_CUT"):
            return dict(file="0013-PREPASS_ENTRYPOINT_NAME.ptx", cut_call=(3964, 44528),
                        extern=make_ptx.swin_cut_extern())
        return make_ptx.KERNELS["cuda_dldn_engine_swin_enc0_kernel"]

    calls = []
    for mi, base in enumerate((6784, 6824, 6864, 6904, 6944, 6984)):
        operands = [f"%r{base + i}" for i in range(4)]
        operands += [f"%r{6495 + i}" for i in range(8)]
        outputs = [f"%r{6503 + 40 * mi + 10 * nt + rh}" for nt in range(4) for rh in range(2)]
        declarations = "\n".join(f"\t.param .b32 p8a{i};" for i in range(12))
        stores = "\n".join(f"\tst.param.b32 [p8a{i}], {reg};" for i, reg in enumerate(operands))
        loads = "\n".join(f"\tld.shared.b32 {reg}, [%p8out+{4 * i}];" for i, reg in enumerate(outputs))
        args = ", ".join([f"p8a{i}" for i in range(12)] + ["p8s"])
        calls.append(f"""{{
\t.reg .b32 %p8out;
{declarations}
\t.param .b32 p8s;
{stores}
\tst.param.b32 [p8s], %p8base;
\tcall.uni d4r_swin_enc0_p8, ({args});
\tmad.lo.u32 %p8out, %laneid, 32, %p8base;
{loads}
}}""")
    # input_tensor's last read is in the prologue (line 3938); only the
    # disjoint input_noise array is read by the final surface epilogue.
    call = "{\n\t.reg .b32 %p8base;\n"
    call += f"\tmov.u32 %p8base, {make_ptx.SWIN_RET_SCRATCH};\n" + "\n".join(calls) + "\n}"
    params = ",\n\t".join([f".param .b32 a{i}" for i in range(12)]
                            + [".param .b32 scratch"])
    return dict(file="0013-PREPASS_ENTRYPOINT_NAME.ptx", cut_call=(19770, 19935),
                call=call, extern=f".extern .func d4r_swin_enc0_p8\n(\n\t{params}\n)\n;\n")
