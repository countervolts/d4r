#!/usr/bin/env python3
"""Print the expression tree that defines a PTX register, followed back through arithmetic, pair packs/unpacks,
conversions and warp shuffles down to loads, constants and registers named as leaves.

usage: ptx_expr.py STAGE LINE %REG [--depth N] [--stop %r1,%r2]

LINE is a line at or after the register's definition (the search goes backwards from it). Brace-scoped temporaries
(`{ .reg .b16 hl, hu; ... }`) are resolved within their block. Shuffles print as `shfl.bfly^k(expr)`: the same
expression on the lane `laneid ^ k`.
"""
import argparse
import re
from pathlib import Path


def module_of(kernel):
    for p in sorted(Path('~/.cache/d4r-rr-corpus').expanduser().glob('*.ptx')):
        if f'.entry {kernel}(' in p.read_text(errors='replace'):
            return p


class Tracer:
    def __init__(self, lines, start):
        self.lines = lines
        self.start = start
        self.consts = {}
        self.cache = {}

    def find_def(self, reg, before, block_lo):
        """the statement (line index) that last writes REG before line BEFORE"""
        pat = re.compile(r'(^|[\s{,|])' + re.escape(reg) + r'([\s,|}]|$)')
        lo = block_lo if not reg.startswith('%') else self.start
        for i in range(before - 1, lo - 1, -1):
            s = re.sub(r'//.*', '', self.lines[i]).strip().lstrip('{').strip()
            s = re.sub(r'^@!?%p\d+\s+', '', s)
            if not s or s.startswith(('.', '$', '}')):
                continue
            parts = s.split(None, 1)
            if len(parts) < 2:
                continue
            rest = parts[1]
            dsts = ([x.strip() for x in rest[1:rest.index('}')].split(',')] if rest.startswith('{')
                    else [x.strip() for x in rest.split(',')[0].split('|')])
            if reg in dsts and not parts[0].startswith(('st.', 'sust', 'bra')):
                return i, parts[0], rest.rstrip(';').strip(), dsts
        return None

    def block_start(self, line):
        depth = 0
        for i in range(line, self.start, -1):
            depth += self.lines[i].count('}') - self.lines[i].count('{')
            if depth < 0:
                return i
        return self.start

    def expr(self, reg, before, depth, stop, half=None):
        reg = reg.strip()
        if re.fullmatch(r'-?\d+|0[fFdD][0-9A-Fa-f]+|0x[0-9a-fA-F]+', reg):
            return reg
        if reg in stop or depth <= 0:
            return reg + (f'.{half}' if half is not None else '')
        d = self.find_def(reg, before, self.block_start(before - 1))
        if d is None:
            return reg + '?'
        i, op, rest, dsts = d
        args = [a.strip() for a in re.split(r',(?![^{]*\})', rest)]
        srcs = args[1:] if not rest.startswith('{') else args[1:]
        e = lambda r, h=None: self.expr(r, i, depth - 1, stop, h)
        if op.startswith('mov.b32') and rest.startswith('{'):          # unpack: {lo, hi} <- r
            which = 'lo' if dsts.index(reg) == 0 else 'hi'
            return e(srcs[0], which)
        if op.startswith('mov.b32') and srcs and srcs[0].startswith('{'):   # pack r <- {lo, hi}
            lo, hi = [x.strip() for x in srcs[0][1:-1].split(',')]
            if half == 'lo':
                return e(lo)
            if half == 'hi':
                return e(hi)
            return f'pack({e(lo)}, {e(hi)})'
        if op.startswith(('mov.', 'cvt.')) and len(srcs) == 1:
            inner = e(srcs[0], half)
            return inner if op.startswith('mov.') else f'{op.split(".")[0]}.{op.split(".")[-2]}({inner})'
        if op.startswith('shfl.sync.bfly'):
            return f'shfl^{srcs[1]}({e(srcs[0], half)})'
        if op.startswith('ld.'):
            return f'load@L{i + 1}[{srcs[0]}]' + (f'.{half}' if half else '')
        name = op.split('.')[0]
        if op.endswith('x2') and half:
            return f'{name}(' + ', '.join(e(s, half) for s in srcs) + ')'
        return f'{name}(' + ', '.join(e(s, half) for s in srcs) + ')'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage')
    ap.add_argument('line', type=int)
    ap.add_argument('reg')
    ap.add_argument('--depth', type=int, default=30)
    ap.add_argument('--stop', default='')
    o = ap.parse_args()
    kernel = f'cuda_dldn_engine_swin_{o.stage}_kernel'
    lines = module_of(kernel).read_text().replace('\r', '').split('\n')
    k = next(i for i, l in enumerate(lines) if f'.entry {kernel}(' in l)
    t = Tracer(lines, k)
    print(t.expr(o.reg, o.line, o.depth, set(filter(None, o.stop.split(',')))))


if __name__ == '__main__':
    main()
