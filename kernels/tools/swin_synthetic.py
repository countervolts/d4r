#!/usr/bin/env python3
"""Generate deterministic, untrained M/L Swin replay fixtures with shifted/border windows.

Usage: swin_synthetic.py OUTPUT [--large] [--seed N]
No NVIDIA data is embedded. Run each fixture through dump_runner with control
and candidate HSACOs, compare all allocations byte-for-byte; swin_check.py can
also compare sampled blocks to the independent NumPy model. Synthetic timing
is not trained-model or end-to-end inference performance.
"""
from pathlib import Path
import sys, struct
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import swin_model as sm
from swin_check import LAYERS
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('output', type=Path)
parser.add_argument('--large', action='store_true', help='M 1280x720 render layer dimensions')
parser.add_argument('--seed', type=int, default=87001)
options=parser.parse_args()
root=options.output
if root.exists():
    parser.error('output must be a new directory')
rng=np.random.default_rng(options.seed)
for layer,(C,NH,NPM,CIN) in LAYERS.items():
    for shift in [0,4]:
        W,H=({'enc1':(640,360),'dec1':(640,360),'enc2':(320,180),'dec2':(320,180),'enc3_tube':(160,90)}[layer] if options.large else (24,16))
        d=root/f'{layer}-s{shift}';d.mkdir(parents=True,exist_ok=True)
        pre=4*C*CIN+8*C if CIN else 0
        L=sm.layout(C,NH,pre)
        total=L['end']+4*C*NPM+2*NPM
        # Small finite, signed E4M3 values keep random chains away from overflow.
        w=rng.choice(np.array([0,8,16,24,32,136,144,152,160],np.uint8),total)
        def half(off,n,scale=.1):w[off:off+2*n]=rng.uniform(-scale,scale,n).astype(np.float16).view(np.uint8)
        def gamma(off):w[off:off+2*C]=np.full(C,.125,np.float16).view(np.uint8)
        gamma(L['g1']);gamma(L['g2']);half(L['bo'],C);half(L['b2'],C)
        for off in L['head']:half(off+64*C,256)
        for off in L['b1']:half(off+32*C,32)
        if NPM:half(L['end']+4*C*NPM,NPM)
        if CIN:
            for q in range(4):half(4*C*CIN+2*q*C,C)
        codes=np.array([0,24,32,40,48,152,160,168,176],np.uint8)
        arrays=[w.tobytes(),rng.choice(codes,W*H*(CIN//4 if CIN else C)).tobytes(),rng.choice(codes,W*H*C).tobytes(),bytes([0xa5])*(W*H*C),bytes([0xa5])*(max(1,W*H*NPM//4))]
        ptr=[(i+1)*0x100000000 for i in range(5)]
        args=struct.pack('<Q4i4Q',ptr[0],shift,shift,W,H,ptr[1],ptr[2],ptr[3],ptr[4])
        if layer=='enc3_tube':args+=bytes(32)
        (d/'args.bin').write_bytes(args)
        kernel=f'rrlite_{layer}_4x4'
        lines=[f'kernel {kernel}',f'launch {(W+shift+7)//8} {(H+shift+7)//8} 1 32 1 8 0',f'args {len(args)}']
        for i,buf in enumerate(arrays):
            (d/f'alloc-{i}.bin').write_bytes(buf)
            lines.append(f'alloc {i} {hex(ptr[i])} {len(buf)}')
        for off,i in [(0,0),(24,1),(32,2),(40,3),(48,4)]:lines.append(f'pointer {off} {i} 0')
        (d/'manifest.txt').write_text('\n'.join(lines)+'\n')
