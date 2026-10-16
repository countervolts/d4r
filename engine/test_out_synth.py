#!/usr/bin/env python3
"""Compare the own K output kernel with the translated reference kernel on a captured frame whose motion and depth are
replaced by synthetic fields (the captured scenes barely move, so they do not exercise the reprojection).

usage: test_out_synth.py CAPTURE_DIR FRAME WORK_DIR KVK REF_SPV [seed]
  REF_SPV: vulkan/ptx2glsl.py's translation of the reference kernel, compiled (see vulkan/README.md).
"""
import glob, os, re, subprocess, sys
import numpy as np
K = 'hiluma_engine_output_depthinv_mvlo_hdr_max_v2_rel'
cap, frame, work, kvk, ref_spv = sys.argv[1:6]
seed = int(sys.argv[6]) if len(sys.argv) > 6 else 1
HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(f'{work}/ref', exist_ok=True)
rng = np.random.default_rng(seed)
seqs = sorted({int(re.match(r'launch-(\d+)-', os.path.basename(f)).group(1)) for f in glob.glob(f'{cap}/dump/launch-*-{K}-arg*')})
seq = seqs[int(frame)]
def shape_of(off):
    f = glob.glob(f'{cap}/dump/launch-{seq:03d}-{K}-arg{off}-*')[0]
    w, h, c = (int(x) for x in re.search(r'-(\d+x\d+x\d+)-fmt', f).group(1).split('x'))
    return f, w, h, c
def smooth(h, w, cells, amp):
    # bilinear upsampling of a coarse random grid: a smooth field with fractional values
    g = rng.uniform(-amp, amp, (cells + 1, cells + 1))
    y = np.linspace(0, cells, h)[:, None]; x = np.linspace(0, cells, w)[None, :]
    y0 = np.minimum(y.astype(int), cells - 1); x0 = np.minimum(x.astype(int), cells - 1); fy = y - y0; fx = x - x0
    return g[y0, x0] * (1 - fy) * (1 - fx) + g[y0, x0 + 1] * (1 - fy) * fx + g[y0 + 1, x0] * fy * (1 - fx) + g[y0 + 1, x0 + 1] * fy * fx
f, w, h, c = shape_of(240)
mv = np.stack([smooth(h, w, 6, 3.0), smooth(h, w, 6, 3.0)], axis=-1)
mv[rng.uniform(size=(h, w)) < 0.02] += rng.uniform(-8, 8, 2)            # outliers, also off-screen reprojections
mv.astype(np.float16).tofile(f'{work}/mv.bin')
f, w, h, c = shape_of(248)
depth = 0.3 + 0.1 * smooth(h, w, 5, 1.0)
blobs = smooth(h, w, 12, 1.0) > 0.35                                      # nearer objects with hard edges
depth[blobs] += 0.4
depth.astype(np.float32).tofile(f'{work}/depth.bin')
# reference: the translated kernel on the modified inputs
out = subprocess.run([sys.executable, f'{HERE}/../vulkan/kcase.py', 'make', cap, K, frame, f'{work}/ref', '200:linear', '224:linear', '232:linear'],
                     capture_output=True, text=True, check=True).stdout
man = open(f'{work}/ref/manifest').read().split('\n')
for i, l in enumerate(man):
    if l.startswith('tex 240 '): man[i] = re.sub(r'^tex 240 \S+', f'tex 240 {work}/mv.bin', l)
    if l.startswith('tex 248 '): man[i] = re.sub(r'^tex 248 \S+', f'tex 248 {work}/depth.bin', l)
    if l.startswith('surf '): man[i] = re.sub(r'^(surf \d+ \d+ \d+ \d+ \d+) \S+', r"\1 -", l)
open(f'{work}/ref/manifest', 'w').write('\n'.join(man))
subprocess.run([kvk, ref_spv, f'{work}/ref/manifest', '1', '64'], check=True, capture_output=True)
env = dict(os.environ, OUT_TEX_OVERRIDE=f'240={work}/mv.bin,248={work}/depth.bin', OUT_REF_DIR=f'{work}/ref')
print(subprocess.run([sys.executable, f'{HERE}/test_out.py', cap, frame, work, kvk], env=env, capture_output=True, text=True).stdout.strip())
