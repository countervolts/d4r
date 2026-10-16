#!/usr/bin/env python3
"""Build the engine's Ray Reconstruction model directory (preset D or E) from a capture of the denoiser.

usage: compile_rr.py CAPTURE_DIR OUT_DIR [--ptx-dir DIR] [--frame N] [--network-bindings-only]
       compile_rr.py --dll NVNGX_DLSSD_DLL --preset D|E OUT_DIR [--ptx-dir DIR] --network-bindings-only

CAPTURE_DIR is a bridge capture of a few denoiser frames (engine/rr/rr_capture.py: replay/ with the launches' weight
allocations, dump/, run/run.log), taken with the preset the model is for. Writes:
  weights-<stage>.bin   each stage's weight image as NGX uploads it (argument 40)
  enc0.args, hkpn.args  the two argument-block templates whose constants the runtime keeps (engine/runtime_rr.inc)
  <stage>.spv           one shader per stage, exposure_rr0.spv / exposure_rr1.spv
  rr.txt                the manifest RModel reads
A stage's shader is d4r's own engine/rr/<stage>.comp (or <stage>_d.comp for preset D) when it exists; otherwise it is
generated from the kernel's PTX in the user's own nvngx_dlssd.dll by engine/rr/ptx2rr.py (--ptx-dir: the modules
kernels/tools/extract_dlss_ptx.py wrote, named NNNN-<first entry>.ptx; default ~/.cache/d4r-rr-corpus).
The weights, the templates and any generated shader are derived from NVIDIA's DLL: they stay on the user's machine
and are never packaged (scripts/check_redistributable.py).

With --dll instead of a capture nothing is taken from a capture: the templates come from engine/rr/rr_templates.py,
the output stage's weight image is NGX's 16 zero bytes, and the network stages get no weight image, so the model is
only usable once engine/rr/build_native_model.sh has replaced them with the native stages (weights read from the DLL).
--network-bindings-only translates enc0 ... dec0 for their bindings (and enc0's gather, engine/rr/gen_enc0_m.py) but
does not compile them: build_native_model.sh replaces them, and the translations' pipeline compiles take tens of GB.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'rr'))
from rr_capture import Capture  # noqa: E402
from test_layer import stage_params  # noqa: E402

STAGES = ['enc0', 'enc1', 'enc2', 'enc3', 'enc4', 'enc5', 'dec4', 'dec3', 'dec2', 'dec1', 'dec0', 'hkpn']
KERNEL = {s: f'cuda_dldn_engine_swin_{s}_kernel' for s in STAGES[:-1]}
KERNEL['hkpn'] = 'cuda_dldn_engine_hkpn_output_kernel_transformer'
# preset D runs its own enc0, dec0 and output kernels
D_KERNEL = {'enc0': 'cuda_dldn_engine_swin_enc0_diamond_wallaby_kernel',
            'dec0': 'cuda_dldn_engine_swin_dec0_diamond_wallaby_kernel',
            'hkpn': 'cuda_dldn_engine_hkpn_output_kernel_transformer_diamond_wallaby'}


# d4r's own stage shaders: their texture and surface bindings (argument offsets, ascending)
OWN_BINDINGS = {'hkpn': ([176, 192, 200, 208, 216, 280, 288, 296, 448], [(224, 'rgba16f'), (240, 'rgba16f')])}
# own shaders that implement the reset path only: engine/rr/hkpn.comp upsamples the current frame and has no history
# reprojection, so a game (Reset = 0) gets a blocky, jittering image from it (RE Requiem, 2026-10-12). The translated
# kernel is used unless --reset-only asks for the own shader (captures that reset every frame).
RESET_ONLY = {'hkpn'}
# stages compiled without driver optimizations (engine/runtime_rr.inc, "noopt"): measured per stage against capE3 on
# RADV/Mesa 26.2 gfx1201, only enc1 is better for it (its merged output 43 -> 64 dB); every other stage loses 2-6 dB
NOOPT = {'enc1'}
# formatted stores (sust.p) whose surface is not the store's own guess: dec0's recurrent feature is NGX's SNORM_INT8X4
SURFACE_FORMATS = {'dec0': {144: 'rgba8_snorm'}}


# D4R_RR_GFX11=1: shaders for RDNA3, which has no FP8 matrices or VK_EXT_shader_float8 (engine/rr/rr_gfx11.glsl)
GFX11 = ['-DRR_GFX11'] if os.environ.get('D4R_RR_GFX11') == '1' else []


def glsl(source, output, *defines):
    subprocess.run(['glslangValidator', '-V', '--target-env', 'vulkan1.3', *GFX11, *defines, str(source), '-o', str(output)],
                   check=True, stdout=subprocess.DEVNULL)


def find_module(ptx_dir, kernel):
    for path in sorted(Path(ptx_dir).glob('*.ptx')):
        if f'.entry {kernel}(' in path.read_text(errors='replace'):
            return path
    raise FileNotFoundError(f'no PTX module with the entry {kernel} in {ptx_dir}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('capture', nargs='?')
    ap.add_argument('output', type=Path)
    ap.add_argument('--dll', help="build without a capture, from the user's nvngx_dlssd.dll (with --preset)")
    ap.add_argument('--preset', choices=['D', 'E'])
    ap.add_argument('--network-bindings-only', action='store_true',
                    help='translate the network stages for their bindings only, without compiling them')
    ap.add_argument('--ptx-dir', default=os.path.expanduser('~/.cache/d4r-rr-corpus'))
    ap.add_argument('--frame', type=int, default=2, help='which captured frame supplies the templates (1-based)')
    ap.add_argument('--reset-only', action='store_true', help='use the own output shader that has no history path')
    a = ap.parse_args()
    if a.dll:
        if a.capture or not a.preset or not a.network_bindings_only:
            ap.error('--dll takes --preset and --network-bindings-only, and no capture')
        cap = None
        preset_d = a.preset == 'D'
        from rr_templates import templates
        dll_templates = dict(zip(('enc0', 'hkpn'), templates(Path(a.dll).read_bytes(), a.preset)))
    else:
        if not a.capture:
            ap.error('a capture directory, or --dll')
        cap = Capture(a.capture)
        preset_d = any('diamond_wallaby' in l.kernel for l in cap.launches)
    kernels = dict(KERNEL, **(D_KERNEL if preset_d else {}))
    a.output.mkdir(parents=True, exist_ok=True)
    lines = ['D4RR0001', f'preset {4 if preset_d else 5}']
    info = {'preset': 'D' if preset_d else 'E', 'stages': {}}
    for stage in STAGES:
        network = stage != 'hkpn'
        if cap is None:
            # the output kernel's weight argument: NGX uploads 16 zero bytes for it (every capture of 310.7, D and E)
            weights = bytes(16) if not network else None
            if stage in dll_templates:
                (a.output / f'{stage}.args').write_bytes(dll_templates[stage])
        else:
            launches = [l for l in cap.launches if l.kernel == kernels[stage]]
            if len(launches) < a.frame:
                raise SystemExit(f'the capture has {len(launches)} launches of {kernels[stage]}; frame {a.frame} needed')
            launch = launches[a.frame - 1]
            weights = launch.weights
            if stage in ('enc0', 'hkpn'):
                (a.output / f'{stage}.args').write_bytes(stage_params(launch))
        if weights is not None:
            (a.output / f'weights-{stage}.bin').write_bytes(weights)
        own = HERE / 'rr' / (f'{stage}_d.comp' if preset_d and stage in D_KERNEL else f'{stage}.comp')
        spv = a.output / f'{stage}.spv'
        if own.exists() and (stage not in RESET_ONLY or a.reset_only):
            glsl(own, spv)
            origin = str(own.relative_to(HERE.parent))
            textures, surfaces = OWN_BINDINGS[own.stem]
        else:
            generated = a.output / f'{stage}.generated.comp'
            formats = [f'--surface={off}={fmt}' for off, fmt in SURFACE_FORMATS.get(stage, {}).items()]
            subprocess.run([sys.executable, str(HERE / 'rr' / 'ptx2rr.py'), str(find_module(a.ptx_dir, kernels[stage])),
                            kernels[stage], str(generated), *formats], check=True)
            if not (network and a.network_bindings_only):
                glsl(generated, spv)
            origin = 'generated from the DLL'
            binding = json.loads(generated.with_suffix('.json').read_text())
            textures, surfaces = binding['textures'], [tuple(x) for x in binding['surfaces']]
        tex = ','.join(map(str, textures)) or '-'
        surf = ','.join(f'{off}:{fmt}' for off, fmt in surfaces) or '-'
        lines.append(f'stage {stage} {stage}.spv 32 weights-{stage}.bin {tex} {surf}' + (' noopt' if stage in NOOPT else ''))
        n = len(weights) if weights is not None else 0
        info['stages'][stage] = {'kernel': kernels[stage], 'shader': origin, 'weights': n}
        print(f'{stage}: {kernels[stage]} ({origin}), {n} weight bytes')
    lines += ['template enc0 enc0.args', 'template hkpn hkpn.args']
    for p in (0, 1):
        glsl(HERE / 'exposure_k.comp', a.output / f'exposure_rr{p}.spv', '-DMODE=2', f'-DPASS={p}')
    (a.output / 'rr.txt').write_text('\n'.join(lines) + '\n')
    (a.output / 'model.json').write_text(json.dumps(info, indent=1) + '\n')
    print(f'Ray Reconstruction preset {info["preset"]} model in {a.output}')


if __name__ == '__main__':
    main()
