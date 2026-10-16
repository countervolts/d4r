#!/usr/bin/env python3
"""Instrument a PTX kernel to report where non-finite values first appear.

Adds a trailing `.param .u64 d4r_dbg` to the entry and, after every
floating-point result written to a %-register (MMA outputs, f16/f16x2
arithmetic, conversions, rsqrt/rcp, ...), ORs a flag into
d4r_dbg[source_line] when the value is Inf or NaN (all-ones exponent):
bit 0 for an f16/f16x2 result, bit 1 for an f32 result.

With --max, each slot instead records the largest magnitude seen on that
line (f16 results as f16 bits, f32 results as f32 bits with bit 31 set to
tell them apart), which shows where values grow out of range.

Replay the instrumented kernel with replay_dlss_layer, passing a zero-filled
buffer of 4 * (line_count + 1) bytes as the extra argument, then list the
flagged lines with --report.

usage: instrument_ptx_nonfinite.py [--max] IN.ptx OUT.ptx KERNEL
       instrument_ptx_nonfinite.py --report [--max] IN.ptx DEBUG_BUFFER
"""
import re
import struct
import sys

# Instructions whose first operand (or brace list) is a floating-point result.
F16X2_OPS = re.compile(r"^(add|sub|mul|fma\.rn|min|max|neg|abs)\.f16x2$|^cvt\.rn\.f16x2\.e[45]m[23]x2$")
F16_OPS = re.compile(r"^(add|sub|mul|fma\.rn|min|max|neg|abs)\.f16$|^cvt\.rn\.f16\.f(32|64)$")
F32_OPS = re.compile(r"^(add|sub|mul|fma\.rn|div\.approx|rsqrt\.approx|rcp\.approx|sqrt\.approx|lg2\.approx|ex2\.approx|cvt\.f32\.f16)(\.ftz)?(\.f32)?$")
MMA_F16 = re.compile(r"^mma\.sync\.aligned\.m16n8k(16|32)\.row\.col\.f16\.")

INSTRUCTION = re.compile(r"^(\s*\{?\s*)(@!?%p\d+\s+)?([a-z][\w.:]*)\s+(.*)$")


MAX_MODE = False


def max_checks(offset, register, kind):
    """Record the largest magnitude written by one instruction."""
    if kind == "f16x2":
        return [
            f"and.b32 %d4rdbg_r0, {register}, 32767;",
            f"shr.u32 %d4rdbg_r1, {register}, 16;",
            "and.b32 %d4rdbg_r1, %d4rdbg_r1, 32767;",
            "max.u32 %d4rdbg_r0, %d4rdbg_r0, %d4rdbg_r1;",
            f"atom.global.max.u32 %d4rdbg_r1, [%d4rdbg_rd0+{offset}], %d4rdbg_r0;",
        ]
    if kind == "f16":
        return [
            f"cvt.u32.u16 %d4rdbg_r0, {register};",
            "and.b32 %d4rdbg_r0, %d4rdbg_r0, 32767;",
            f"atom.global.max.u32 %d4rdbg_r1, [%d4rdbg_rd0+{offset}], %d4rdbg_r0;",
        ]
    return [
        f"mov.b32 %d4rdbg_r0, {register};",
        "and.b32 %d4rdbg_r0, %d4rdbg_r0, 2147483647;",
        "shr.u32 %d4rdbg_r0, %d4rdbg_r0, 1;",
        "or.b32 %d4rdbg_r0, %d4rdbg_r0, 2147483648;",
        f"atom.global.max.u32 %d4rdbg_r1, [%d4rdbg_rd0+{offset}], %d4rdbg_r0;",
    ]


def checks_for(line_number, opcode, operands):
    """Return PTX lines checking the destination of one instruction."""
    offset = 4 * line_number
    if MMA_F16.match(opcode):
        destination = re.match(r"\{([^}]*)\}", operands)
        registers = [r.strip() for r in destination.group(1).split(",")] if destination else []
        kind = "f16x2"
    else:
        first = operands.split(",")[0].strip().rstrip(";")
        registers = [first]
        if F16X2_OPS.match(opcode):
            kind = "f16x2"
        elif F16_OPS.match(opcode):
            kind = "f16"
        elif F32_OPS.match(opcode) and opcode.endswith(("f32", "f16")):
            kind = "f32"
        else:
            return []
    lines = []
    for register in registers:
        if not register.startswith("%"):
            continue  # registers declared in nested inline-asm scopes
        if MAX_MODE:
            lines += max_checks(offset, register, kind)
            continue
        if kind == "f16x2":
            lines += [
                f"and.b32 %d4rdbg_r0, {register}, 2080406528;",  # 0x7c007c00
                "and.b32 %d4rdbg_r1, %d4rdbg_r0, 31744;",  # 0x7c00
                "setp.eq.u32 %d4rdbg_p0, %d4rdbg_r1, 31744;",
                "and.b32 %d4rdbg_r1, %d4rdbg_r0, 2080374784;",  # 0x7c000000
                "setp.eq.or.u32 %d4rdbg_p0, %d4rdbg_r1, 2080374784, %d4rdbg_p0;",
                f"@%d4rdbg_p0 atom.global.or.b32 %d4rdbg_r1, [%d4rdbg_rd0+{offset}], 1;",
            ]
        elif kind == "f16":
            lines += [
                f"cvt.u32.u16 %d4rdbg_r0, {register};",
                "and.b32 %d4rdbg_r0, %d4rdbg_r0, 31744;",
                "setp.eq.u32 %d4rdbg_p0, %d4rdbg_r0, 31744;",
                f"@%d4rdbg_p0 atom.global.or.b32 %d4rdbg_r1, [%d4rdbg_rd0+{offset}], 1;",
            ]
        else:
            lines += [
                f"mov.b32 %d4rdbg_r0, {register};",
                "and.b32 %d4rdbg_r0, %d4rdbg_r0, 2139095040;",  # 0x7f800000
                "setp.eq.u32 %d4rdbg_p0, %d4rdbg_r0, 2139095040;",
                f"@%d4rdbg_p0 atom.global.or.b32 %d4rdbg_r1, [%d4rdbg_rd0+{offset}], 2;",
            ]
    return lines


def instrument(source, kernel):
    lines = source.split("\n")
    output = []
    in_kernel = False
    entry_params_open = False
    body_started = False
    pending_statement = None  # multi-line mma statements
    for number, line in enumerate(lines, start=1):
        if re.match(rf"^\.visible \.entry {re.escape(kernel)}\(", line):
            in_kernel = True
            entry_params_open = True
            output.append(line)
            continue
        if in_kernel and entry_params_open and line.strip() == ")":
            output[-1] = output[-1] + ","
            output.append(".param .u64 d4r_dbg")
            output.append(line)
            entry_params_open = False
            continue
        if in_kernel and not body_started and line.strip() == "{":
            output.append(line)
            output.append(".reg .pred %d4rdbg_p0;")
            output.append(".reg .b32 %d4rdbg_r<2>;")
            output.append(".reg .b64 %d4rdbg_rd0;")
            output.append("ld.param.u64 %d4rdbg_rd0, [d4r_dbg];")
            output.append("cvta.to.global.u64 %d4rdbg_rd0, %d4rdbg_rd0;")
            body_started = True
            continue
        output.append(line)
        if not body_started:
            continue
        if line.strip() == "}" and pending_statement is None and number == len(lines):
            in_kernel = False
        statement = line if pending_statement is None else pending_statement[1] + " " + line
        start = number if pending_statement is None else pending_statement[0]
        if ";" not in line:
            match = INSTRUCTION.match(statement)
            if match and MMA_F16.match(match.group(3)):
                pending_statement = (start, statement)
            continue
        pending_statement = None
        match = INSTRUCTION.match(statement)
        if not match or match.group(2):
            continue  # skip predicated results: the check would run unconditionally
        checks = checks_for(start, match.group(3), match.group(4))
        if checks:
            output.extend(checks)
    return "\n".join(output), len(lines)


def half_value(bits):
    return struct.unpack("<e", struct.pack("<H", bits & 0xFFFF))[0]


def report_max(lines, values, limit):
    rows = [(index, value) for index, value in enumerate(values) if value]
    print(f"{len(rows)} source lines recorded magnitudes")
    for index, value in rows[:limit]:
        if value & 0x80000000:
            magnitude = struct.unpack("<f", struct.pack("<I", (value & 0x7FFFFFFF) << 1))[0]
            kind = "f32"
        else:
            magnitude = half_value(value)
            kind = "f16"
        text = lines[index - 1].strip() if 0 < index <= len(lines) else "?"
        print(f"  line {index:6d} [{kind}] max|x|={magnitude:<12.6g} {text[:90]}")


def report(ptx_path, debug_path, max_mode=False, limit=40):
    lines = open(ptx_path).read().split("\n")
    data = open(debug_path, "rb").read()
    flags = struct.unpack(f"<{len(data) // 4}I", data)
    if max_mode:
        report_max(lines, flags, limit)
        return
    hits = [(index, flag) for index, flag in enumerate(flags) if flag]
    print(f"{len(hits)} source lines produced non-finite values")
    for index, flag in hits[:40]:
        kind = "+".join(k for bit, k in ((1, "f16"), (2, "f32")) if flag & bit)
        text = lines[index - 1].strip() if 0 < index <= len(lines) else "?"
        print(f"  line {index:6d} [{kind}] {text[:110]}")


if __name__ == "__main__":
    arguments = sys.argv[1:]
    reporting = "--report" in arguments
    MAX_MODE = "--max" in arguments
    arguments = [a for a in arguments if a not in ("--report", "--max")]
    if reporting and len(arguments) == 2:
        report(arguments[0], arguments[1], MAX_MODE, limit=400)
    elif len(arguments) == 3:
        sys.argv = [sys.argv[0]] + arguments
        text, count = instrument(open(sys.argv[1]).read(), sys.argv[3])
        open(sys.argv[2], "w").write(text)
        print(f"instrumented {sys.argv[3]}: {count} source lines; debug buffer needs {4 * (count + 1)} bytes")
    else:
        print(__doc__)
        sys.exit(2)
