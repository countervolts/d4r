#!/usr/bin/env python3
"""Which earlier result each phase operand of a dumped layer (engine/rr/phase_dump.py) comes from.
usage: relations.py DUMP.npz [WAVE]"""
import sys
import numpy as np
sys.path.insert(0, __import__('os').path.dirname(__file__))
from rrmath import f16, q8

z = np.load(sys.argv[1]); w = int(sys.argv[2]) if len(sys.argv) > 2 else 0
phases = sorted({int(k.split('_')[0][1:]) for k in z.files})
P = lambda p, n: z[f'p{p}_w{w}_{n}']


def inter(n):
    """the C/D -> A reinterpretation within each 32 block: column 4m+j <- 2m+j, 4m+2+j <- 16+2m+j"""
    perm = []
    for b in range(n // 32):
        for m in range(8):
            perm += [32 * b + 2 * m, 32 * b + 2 * m + 1, 32 * b + 16 + 2 * m, 32 * b + 16 + 2 * m + 1]
    return perm


def rel(op, M, p):
    found = []
    if not M.any():
        return ['zero']
    for q in range(p - 1, 0, -1):
        D = P(q, 'D')
        if op == 'C':
            for name, X in (('D', D),):
                if X.shape == M.shape and np.array_equal(X, M):
                    found.append(f'{name}{q}')
            continue
        Q = q8(f16(D))
        cands = []
        if Q.shape[1] % 32 == 0:
            cands += [('q8(D%d)[:, inter]' % q, Q[:, inter(Q.shape[1])])]
        cands += [('q8(D%d)' % q, Q)]
        for name, X in cands:
            for tname, Y in ((name, X), (name + '.T', X.T)):
                if Y.shape == M.shape and np.array_equal(Y, M):
                    found.append(tname)
                elif Y.shape[1] == M.shape[1] and Y.shape[0] > M.shape[0] and np.array_equal(Y[:M.shape[0]], M):
                    found.append(tname + f'[:{M.shape[0]}]')
                if op == 'B' and Y.shape[0] == M.shape[0] and Y.shape[1] > M.shape[1] and np.array_equal(Y[:, :M.shape[1]], M):
                    found.append(tname + f'[:, :{M.shape[1]}]')
        if found:
            break
    return found or ['?']


for p in phases:
    A, B, C = P(p, 'A'), P(p, 'B'), P(p, 'C')
    print(f'phase {p:2d} {A.shape[0]}x{B.shape[1]}x{A.shape[1]}:  A <- {rel("A", A, p)}  B <- {rel("B", B, p)}  '
          f'C <- {rel("C", C, p)}')
