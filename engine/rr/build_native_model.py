#!/usr/bin/env python3
"""build_native_model.sh --dll for hosts without bash (native Windows): a native Ray Reconstruction model directory
for preset D or E, everything taken from nvngx_dlssd.dll (its PTX, the argument-block templates, the weights).

usage: build_native_model.py --dll NVNGX_DLSSD_DLL D|E OUT_DIR

The same steps, defaults and outputs as build_native_model.sh --dll: the translated model (engine/compile_rr.py), the
weights (dll_weights.py), each native stage packed and compiled against its packing's header, rr.txt rewritten to the
native stages. Builds the default arithmetic only (D4R_RR_RELAXED level 3; D4R_RR_RELAXED=0-2 and D4R_RR_MIX=1 as
there); D4R_RR_GFX11=1 builds the RDNA3 variant (exact arithmetic, widened weights, as the bash script); the
capture checks stay in the bash script. OUT_DIR/native-work keeps the
generated sources compile_rr_dxil.py builds the D3D12 shaders from.
Everything written is derived from the user's own nvngx_dlssd.dll and stays on their machine.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

RR = Path(__file__).resolve().parent
ROOT = RR.parents[1]
SWIN = ('enc1', 'enc2', 'enc3', 'enc4', 'enc5', 'dec4', 'dec3', 'dec2', 'dec1')


def run(*command, log=None):
    with open(log, 'w') if log else open(os.devnull, 'w') as out:
        result = subprocess.run([str(c) for c in command], stdout=out, stderr=subprocess.STDOUT if log else None,
                                cwd=ROOT)
    if result.returncode:
        if log:
            print(''.join(Path(log).read_text(errors='replace').splitlines(True)[-5:]), file=sys.stderr, end='')
        raise SystemExit(f'build_native_model: failed: {" ".join(map(str, command[:3]))}')


def python(*args, log=None):
    run(sys.executable, *args, log=log)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dll', type=Path, required=True)
    p.add_argument('preset', choices=('D', 'E'))
    p.add_argument('out', type=Path)
    a = p.parse_args()
    dll, out = a.dll.resolve(), a.out.resolve()
    if not dll.is_file():
        raise SystemExit(f'no nvngx_dlssd.dll at {dll}')
    work = out / 'native-work'
    work.mkdir(parents=True, exist_ok=True)
    glslang = shutil.which('glslangValidator')
    if not glslang:
        raise SystemExit('glslangValidator is not on PATH')
    any_kernels = ['--any-kernels'] if os.environ.get('D4R_RR_ANY_KERNELS') else []

    python(ROOT / 'kernels/tools/extract_dlss_ptx.py', dll, work / 'ptx')
    python(ROOT / 'engine/compile_rr.py', '--dll', dll, '--preset', a.preset, out, '--ptx-dir', work / 'ptx',
           '--network-bindings-only', log=work / 'compile_rr.log')
    python(RR / 'dll_weights.py', dll, work / 'dll', '--model', '0' if a.preset == 'D' else '1', *any_kernels)
    rr_txt = out / 'rr.txt'
    preset = next(line.split()[1] for line in rr_txt.read_text().splitlines() if line.startswith('preset '))
    # the same defines as build_native_model.sh (see there): RR_EXACT_CONVERT, RR_PRESET_D, the relaxed levels
    defs = ['-DRR_EXACT_CONVERT'] + (['-DRR_PRESET_D'] if preset == '4' else [])
    relaxed = int(os.environ.get('D4R_RR_RELAXED', '3'))
    if relaxed > 0:
        defs.append(f'-DRR_RELAXED={min(relaxed, 2)}')
        defs += ['-DRR_NATIVE'] if relaxed >= 3 else []
    defs += ['-DRR_MIX'] if os.environ.get('D4R_RR_MIX') == '1' else []
    # D4R_RR_GFX11=1: RDNA3 (no FP8 matrices): engine/rr/rr_gfx11.glsl's exact arithmetic only, each native weight file
    # with a widened copy (W_WIDE), `arithmetic gfx11` in rr.txt (compile_rr.py reads the same variable)
    gfx11 = os.environ.get('D4R_RR_GFX11') == '1'
    if gfx11:
        defs = ['-DRR_EXACT_CONVERT', '-DRR_GFX11'] + (['-DRR_PRESET_D'] if preset == '4' else [])
        relaxed = 0
        rr_txt.write_text(re.sub(r'^(preset .*)$', r'\1\narithmetic gfx11', rr_txt.read_text(), count=1, flags=re.M),
                          newline='\n')

    def widen(weights):
        """RDNA3: append the widened copy and return its -DW_WIDE define (rr/widen_weights.py)"""
        if not gfx11:
            return []
        out_ = subprocess.run([sys.executable, str(RR / 'widen_weights.py'), str(weights)], capture_output=True,
                              text=True, check=True).stdout.strip()
        return [f'-DW_WIDE={out_}u']

    def glsl(spv, source, *includes, extra=()):
        run(glslang, '--target-env', 'vulkan1.3', '-V', *defs, *extra, *[f'-I{d}' for d in includes], '-o', spv, source,
            log=work / 'glsl.log')
        if relaxed < 2:   # from level 2 on the conversions are left to the driver
            python(RR / 'spv_exact_convert.py', spv)

    def native(stage, weights):
        """point rr.txt's stage line at STAGE_m.spv and its native weights"""
        text = rr_txt.read_text()
        pattern = re.compile(rf'^stage {stage} {stage}\.spv 32 weights-{stage}\.bin (.*?)( noopt)?$', re.M)
        text, n = pattern.subn(lambda m: f'stage {stage} {stage}_m.spv 32 {weights} {m[1]}', text)
        if n != 1:
            raise SystemExit(f'rr.txt: no stage line for {stage}')
        rr_txt.write_text(text, newline='\n')   # the runtimes read LF (Windows too)

    for s in SWIN:
        d = work / s
        d.mkdir(exist_ok=True)
        shutil.copyfile(work / 'dll' / f'{s}.npz', d / 'weights.npz')
        python(RR / 'swin_pack.py', s, d / 'weights.npz', out / f'weights-{s}m.bin', d / 'swin_stage.h')
        glsl(out / f'{s}_m.spv', RR / 'swin_m.comp', d, RR, extra=widen(out / f'weights-{s}m.bin'))
        native(s, f'weights-{s}m.bin')

    d = work / 'dec0'
    d.mkdir(exist_ok=True)
    shutil.copyfile(work / 'dll' / 'dec0.npz', d / 'weights.npz')
    python(RR / 'dec0_pack.py', d / 'weights.npz', out / 'weights-dec0m.bin', d / 'dec0_m_w.h')
    glsl(out / 'dec0_m.spv', RR / 'dec0_m.comp', d, RR, extra=widen(out / 'weights-dec0m.bin'))
    native('dec0', 'weights-dec0m.bin')

    # encoder 0: the translated feature gather + the native rest; preset E's gather dithers with the noise table
    # that starts NGX's weight image
    d = work / 'enc0'
    d.mkdir(exist_ok=True)
    shutil.copyfile(work / 'dll' / 'enc0.npz', d / 'weights.npz')
    base = 0 if preset == '4' else 4096
    python(RR / 'enc0_pack.py', d / 'weights.npz', d / 'enc0_m.bin', d / 'enc0_m_w.h', base)
    blob = (d / 'enc0_m.bin').read_bytes()
    if base:
        noise = work / 'dll' / 'noise.bin'
        if not noise.is_file() or noise.stat().st_size != 4096:
            raise SystemExit(f'no noise table found in {dll}')
        blob = noise.read_bytes() + blob
    (out / 'weights-enc0m.bin').write_bytes(blob)
    python(RR / 'gen_enc0_m.py', out / 'enc0.generated.comp', d / 'enc0_m.comp')
    glsl(out / 'enc0_m.spv', d / 'enc0_m.comp', d, RR, extra=widen(out / 'weights-enc0m.bin'))
    native('enc0', 'weights-enc0m.bin')

    # the translated output stage has no subgroup operations: 64-wide waves
    if json.loads((out / 'model.json').read_text())['stages']['hkpn'].get('shader') == 'generated from the DLL':
        rr_txt.write_text(re.sub(r'^stage hkpn hkpn\.spv 32 ', 'stage hkpn hkpn.spv 64 ', rr_txt.read_text(),
                                 flags=re.M), newline='\n')
    for s in ('enc0', *SWIN, 'dec0'):
        (out / f'weights-{s}.bin').unlink(missing_ok=True)
    # a binding sidecar for every used shader, so an absent optional input never shifts the descriptor bindings
    for line in rr_txt.read_text().splitlines():
        f = line.split()
        if not f or f[0] != 'stage':
            continue
        textures = [] if f[5] == '-' else list(map(int, f[5].split(',')))
        surfaces = [] if f[6] == '-' else [[int(s.split(':')[0]), s.split(':')[1]] for s in f[6].split(',')]
        (out / f[2]).with_suffix('.json').write_text(json.dumps({'textures': textures, 'surfaces': surfaces}) + '\n')
    print(f'Native Ray Reconstruction model (preset {a.preset}) in {out}')


if __name__ == '__main__':
    main()
