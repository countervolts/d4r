#!/usr/bin/env python3
"""Run the engine's own preset K layers on a captured frame and compare with what NGX's pipeline produced.

usage: test_k.py CAPTURE_DIR MODEL_PREFIX WORK_DIR [--chain] [--time N] [layer ...]
  CAPTURE_DIR/replay: layer launches before they ran (inputs, parameters); CAPTURE_DIR/dump: buffers after each launch.
  --chain feeds each layer the engine's own previous outputs instead of the captured inputs.
"""
import glob, json, os, re, struct, subprocess, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'kernels', 'tools'))
import pwin_model as pm
F16 = np.float16
ORDER = ['enc0', 'enc1', 'enc2', 'enc3', 'enc4', 'dec5', 'dec4', 'dec3', 'dec2', 'dec1', 'dec0']

def params(P, e):
    W, H = P.i32x2(0); sx, sy = P.i32x2(56)
    o = [0] * 108
    o[0], o[1], o[2], o[3] = e['g1'], e['bo'], e['g2'], e['b2']
    for h in range(e['H']):
        o[4 + 5 * h:7 + 5 * h] = e['qkv'][3 * h:3 * h + 3]; o[7 + 5 * h] = e['table'][h]; o[8 + 5 * h] = e['wo'][h]
    for c in range(e['C'] // 8):
        o[44 + 3 * c], o[45 + 3 * c], o[46 + 3 * c] = e['w1'][c], e['b1'][c], e['w2'][c]
    k = e['kind']
    if k in ('enc', 'enc0'):
        o[104], o[105] = e['merge_w'], e['merge_b']
    if k in ('dec', 'dec0'):
        o[104], o[105] = e['expand_w'], e['expand_b']
    if k == 'enc0':
        o[106], o[107] = e['embed_w'], e['embed_b']
    if k == 'dec0':
        o[106], o[107] = e['head_w'], e['head_b']
    return struct.pack('<4i108I', W, H, sx, sy, *o), (W, H)

def defines(e, extra=()):
    k = e['kind']
    kind = {'plain': 0, 'enc': 1, 'enc0': 1, 'dec': 2, 'dec0': 2}[k]
    n16 = e.get('merge_n', 16)
    return [f'-DC={e["C"]}', f'-DNH={e["H"]}', f'-DKIND={kind}', f'-DPOSATTN={int(e["posattn"])}', f'-DEMBED={int(k == "enc0")}',
            f'-DHEAD={int(k == "dec0")}', f'-DXC={e["X"] or 16}', f'-DMERGE_N={n16}', f'-DFOLD_GAINS={int(e.get("fold_gains", True))}',
            *extra]

def ulps(m, g):
    m, g = m.astype(np.float64), g.astype(np.float64)
    scale = np.maximum(np.abs(m), 2.0 ** -10)
    return np.abs(m - g) / 2.0 ** (np.floor(np.log2(scale)) - 10)

def report(tag, ref, got):
    u = ulps(ref, got); r, g = ref.astype(np.float64), got.astype(np.float64)
    if not np.abs(r).max():
        print(f'  {tag:12s} reference is all zero'); return 0.0
    psnr = 10 * np.log10(np.abs(r).max() ** 2 / max(((r - g) ** 2).mean(), 1e-30))
    print(f'  {tag:12s} {ref.size:9d} values: exact {100 * (ref.view(np.uint16) == got.view(np.uint16)).mean():5.1f}%  <=1ulp {100 * (u <= 1).mean():5.1f}%  '
          f'<=4ulp {100 * (u <= 4).mean():6.2f}%  max {u.max():8.1f} ulp  psnr {psnr:6.1f} dB  finite {100 * np.isfinite(got.astype(np.float32)).mean():.0f}%')
    return psnr

def main():
    a = sys.argv[1:]
    chain = '--chain' in a
    if chain: a.remove('--chain')
    reps = 1
    if '--time' in a:
        i = a.index('--time'); reps = int(a[i + 1]); del a[i:i + 2]
    cap, prefix, work = a[:3]
    layers = a[3:] or ORDER
    os.makedirs(work, exist_ok=True)
    index = json.load(open(prefix + '.json'))
    open(f'{work}/dummy.bin', 'wb').write(b'\0' * 16)
    prev = {}        # (kind of buffer, layer) -> file of the engine's own output
    total = 0.0
    for name in layers:
        e = index[name]
        d = sorted(glob.glob(f'{cap}/replay/replay-*-dltss_pwin_{name}_layer'))[0]   # the main launch: params and inputs
        seq = int(re.search(r'replay-(\d+)-', d).group(1))
        P = pm.Dump(d)
        pb, (W, H) = params(P, e)
        open(f'{work}/{name}.params', 'wb').write(pb)
        spv = f'{work}/{name}.spv'
        subprocess.run(['glslangValidator', '-V', '--target-env', 'vulkan1.3', *defines(e), f'{HERE}/layer.comp', '-o', spv], check=True, stdout=subprocess.DEVNULL)
        def slice_in(off, fn):
            buf, o = P.ptr(off); buf[o:].tofile(fn); return fn
        k, C, X = e['kind'], e['C'], e['X']
        fin = slice_in(8, f'{work}/{name}.in'); fskip = f'{work}/dummy.bin'
        if k in ('dec', 'dec0'):
            fskip = slice_in(16, f'{work}/{name}.skip')
        if chain:
            src = {'enc1': 'enc0', 'enc2': 'enc1', 'enc3': 'enc2', 'enc4': 'enc3', 'dec5': 'enc4', 'dec4': 'dec5', 'dec3': 'dec4', 'dec2': 'dec3', 'dec1': 'dec2', 'dec0': 'dec1'}
            skip = {'dec4': 'enc4', 'dec3': 'enc3', 'dec2': 'enc2', 'dec1': 'enc1', 'dec0': 'enc0'}   # skip of decN: the pre-merge output of encN
            if name in src and ('main', src[name]) in prev:
                fin = prev[('main', src[name])]
            if name in skip and ('full', skip[name]) in prev:
                fskip = prev[('full', skip[name])]
        outs = []    # (binding, param offset, bytes, tag)
        if k in ('enc', 'enc0'):
            outs = [(4, 24, (H // 2) * (W // 2) * X * 2, 'merged'), (5, 32, H * W * C * 2, 'full')]
        elif k == 'dec0':
            outs = [(4, 24, H * W * 40 * 2, 'head')]
        else:
            outs = [(4, 24, H * W * C * 2, 'full')]
        cmd = [f'{work}/evk', spv, str(P.grid[0]), str(P.grid[1]), str(reps), f'in:0={work}/{name}.params', f'in:1={prefix}.bin', f'in:2={fin}', f'in:3={fskip}']
        got = {}
        for b, off, n, tag in outs:
            cmd.append(f'out:{b}={work}/{name}.{tag}:{n}'); got[tag] = (off, n)
        if len(outs) == 1:
            cmd.append(f'out:5={work}/{name}.unused:16')
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f'{name} FAILED: {r.stderr[-300:]}')
        t = re.search(r'([0-9.]+) us per dispatch', r.stdout)
        if t: total += float(t.group(1))
        print(f'{name}: {e["kind"]} C={C} H={e["H"]} grid {P.grid[0]}x{P.grid[1]} tokens {W}x{H}' + (f'  {t.group(1)} us' if t else ''))
        for tag, (off, n) in got.items():
            # The post-launch dump holds each allocation once, named by its base address. A phase-split layer
            # writes its output in a later launch of the same layer (kernels/k/pwin_phase.h: NAME_post3 writes
            # out24 for the bottleneck, NAME_post4 the merged output), which the bridge launches right after the
            # main kernel, and the main launch's dump can only hold the output allocation's previous content.
            # Take the last of this layer's launches in this frame that dumped the output allocation.
            addr = P.u64(off)
            base = next(b for b, buf in P.allocs.values() if b <= addr < b + buf.size)
            f, launches = None, []
            for n2, k2 in [(seq, P.kernel)] + [(seq + j, f'{P.kernel}_post{j}') for j in range(1, 9)]:
                if not os.path.isdir(f'{cap}/replay/replay-{n2:06d}-{k2}'):
                    continue
                q = glob.glob(f'{cap}/dump/launch-{n2:03d}-{k2}-arg*-buffer-base0x{base:x}+*')
                launches.append(f'{k2}#{n2}' + ('' if q else '(no dump)'))
                if q: f = q[0]
            if f is None:
                raise ValueError(f'{name}.{tag}: no launch of {name} dumped the output allocation')
            o = addr - base
            ref = np.fromfile(f, dtype=np.uint8)[o:o + n].view(F16)
            if not np.abs(ref.astype(np.float32)).max():
                print(f'  {tag:12s} reference is all zero: {name} launches in this frame: {", ".join(launches)}')
            result = np.fromfile(f'{work}/{name}.{tag}', dtype=F16)[:n // 2]
            if result.shape != ref.shape or not np.isfinite(result).all():
                raise ValueError(f'{name}.{tag}: wrong size or nonfinite output')
            report(tag, ref, result)
            prev[('main' if tag in ('merged', 'head') or k in ('plain', 'dec') else 'full', name)] = f'{work}/{name}.{tag}'
            if tag == 'full' and k in ('enc', 'enc0'):
                prev[('full', name)] = f'{work}/{name}.{tag}'
    if reps > 1:
        print(f'sum of layers: {total / 1000:.3f} ms')

if __name__ == '__main__':
    main()
