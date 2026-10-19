#!/usr/bin/env python3
"""Build the engine's Ray Reconstruction model directory for preset F (DLSS 4.5, the rr2_* kernels of
nvngx_dlssd.dll 310.9) from a capture of the denoiser.

usage: compile_rr2.py CAPTURE_DIR OUT_DIR [--ptx-dir DIR] [--frame N] [--native NVNGX_DLSSD_DLL]
       compile_rr2.py --dll NVNGX_DLSSD_DLL OUT_DIR --ptx-dir DIR

CAPTURE_DIR is a bridge capture of a few preset F frames (engine/rr/rr_capture.py, taken with
D4R_CUDA_REPLAY_DUMP_FILTER=rr2 D4R_CUDA_LAUNCH_STATS_FILTER=rr2 and preset hint 6). Writes:
  <stage>.spv          one shader per kernel, generated from its PTX by engine/rr/ptx2rr.py
  weights-<stage>.bin  the stage's weight image as NGX uploads it (argument 24); the output filter's is the constant
                       table of its PTX module; enc0-aux.bin is encoder 0's second table (argument 104)
  <stage>.args         the argument block of the capture's frame, pointers and handles cleared: the runtime keeps its
                       constants and fills what depends on the frame or the geometry (engine/runtime_rr2.inc)
  exposure_f0.spv, exposure_f1.spv   engine/rr/exposure_f.comp (exposure_f2.spv: both in one, 64-bit atomics)
  rr.txt               the manifest RModel reads ("D4RR0002")
With --native, the stages engine/rr/rr2_m.comp implements (enc1 ... enc5, dec4 ... dec1, dec0's first kernel) are that
shader with weights read from the DLL itself (engine/rr/rr2_pack.py) instead of the translation: bit-identical to
NVIDIA's kernels on the reference captures and about three times faster. D4R_RR_MIX=1 builds the one-rounding variant.
--dll builds the same model without a capture: the native stages from the DLL's weights (as --native), the argument
templates, encoder 0's second table and the arena regions from engine/rr/rr2_templates.py, and the particle variant of
the downsample (downp) always. This is what install.sh runs.
Everything but the exposure shader is derived from NVIDIA's DLL: it stays on the user's machine and is never packaged.
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

# D4R_RR_FUSE=1: the fused multiply-adds of an NVIDIA card (rr2_m.comp RR_FUSE, enc1_ref.FUSE); give --ptx-dir the
# modules rewritten the same way (build/rr/f/nv/mkptx.sh) so the translated parts match
FUSE_DEFS = ['-DRR_FUSE'] if os.environ.get('D4R_RR_FUSE') == '1' else []
# D4R_RR_RELAXED=1|2|3: this GPU's own arithmetic in the native stages instead of the reference's roundings
# (engine/rr/rr_exact.glsl; 3 = also the residual steps in f32, rr2_m.comp RR_NATIVE): faster, ~40 dB from the
# reference, which is also how far two NVIDIA cards are from each other (engine/rr/README.md)
# level 3 (this GPU's own arithmetic, one wave per window where the shader has it) is the default: the fastest build,
# as far from an NVIDIA card as two NVIDIA setups are from each other. D4R_RR_RELAXED=0 builds NVIDIA's roundings
# exactly (bit-identical to the CUDA path through ZLUDA)
RELAXED = int(os.environ.get('D4R_RR_RELAXED', '3') or 0)
# the stages that have a one-wave shader for level 3 (engine/rr/rr2_w.comp: byte-identical to rr2_m.comp there, faster)
ONE_WAVE = {'dec0_0', 'enc0'}
RELAX_DEFS = ([f'-DRR_RELAXED={min(RELAXED, 2)}'] if RELAXED else []) + (['-DRR_NATIVE'] if RELAXED >= 3 else [])
# ... and the normals' filtered fetch of encoder 0 (argument 528) rounded to 16-bit steps, as an NVIDIA card returns it
SNORM16 = {'enc0': [528]} if FUSE_DEFS else {}
NETWORK = ['enc0', 'enc1', 'enc2', 'enc3', 'enc4', 'enc5'] + [f'dec{n}_{k}' for n in (4, 3, 2, 1, 0) for k in (0, 1)]
STAGES = NETWORK + ['post', 'down', 'pooled']
KERNEL = {s: f'rr2_{s}_kernel' for s in NETWORK[:6]}
KERNEL.update({f'dec{n}_{k}': f'rr2_dec{n}_kernel_{k}' for n in range(5) for k in (0, 1)})
KERNEL.update(post='rr2_post_kernel', down='rr2_downsample_kernel_static_hdr_no_transparency',
              pooled='rr2_pooled_history2_kernel')
BLOCK = {'enc0': '32,1,1', 'enc1': '32,1,1', 'enc2': '32,1,4', 'enc3': '32,1,4', 'enc4': '32,1,8', 'enc5': '32,1,8',
         'post': '16,16,1', 'down': '16,16,1', 'pooled': '16,16,1'}
for n, b in ((4, '32,1,8'), (3, '32,1,4'), (2, '32,1,4'), (1, '32,1,2'), (0, '32,1,1')):
    BLOCK[f'dec{n}_0'] = BLOCK[f'dec{n}_1'] = b
# pointer arguments that are neither the weights (24) nor the plane arena: the exposure values (16 bytes), encoder
# 0's second table, and the output passes' progress counters (the runtime leaves those pointers null)
BUFFERS = {'enc0': [104, 256], 'post': [16, 256, 264], 'down': [72, 168, 176]}
# surfaces whose image format is not the store's own guess. The two pooled histories are UNORM 10:10:10:2 arrays for
# NVIDIA; the reference this engine is checked against (ZLUDA) backs them with RGBA32F, and so does the engine.
NATIVE = ['enc1', 'enc2', 'enc3', 'enc4', 'enc5', 'dec4_0', 'dec4_1', 'dec3_0', 'dec3_1', 'dec2_0', 'dec2_1', 'dec1_0',
          'dec1_1', 'dec0_0']
# --particles: the downsample kernel with the particle layer
KERNEL['downp'] = 'rr2_downsample_kernel_static_hdr'
BLOCK['downp'] = '16,16,1'
BUFFERS['downp'] = [72, 168, 176]
# with --native: the translation of the per-pixel code around the native block (engine/rr/gen_rr2_enc0.py)
HYBRID = {'enc0': [], 'dec0_1': ['--dec0_1', '--tail-waves', os.environ.get('D4R_RR2_TAIL_WAVES', '2')]}
# windows per workgroup of the hybrid stages (gen_rr2_enc0.py --group; D4R_RR2_GROUP to experiment)
GROUP = int(os.environ.get('D4R_RR2_GROUP', '1'))
# The pooled histories as RGBA32F holding q / 1023, as ZLUDA backs them (16 bytes a pixel). D4R_RR2_POOLED=rgb10_a2
# makes them real UNORM 10:10:10:2 images (0.1 ms faster at 4K, 70 MB less), but the hardware's bilinear filter then
# rounds differently: identical at 3x and 2x scaling, 44 - 51 dB on temporal frames at 1.5x (capFQ, 2026-10-13); on
# random-motion noise also at 2x (29 dB on the history frames, engine/rr/README.md)
POOLED = os.environ.get('D4R_RR2_POOLED', 'rgba32f')
SURFACES = {'dec0_1': {176: 'rgba8_snorm', 184: 'rgba8', 192: 'rgba8'}, 'post': {120: POOLED}, 'pooled': {8: POOLED}}
# D4R_RR2_UNGUARDED=1 (opt-in, D4R_RR_RELAXED=3 only): the translated code (encoder 0's and decoder 0's per-pixel parts,
# the output passes) without its exactness guards: no `precise` registers, no opaque zero OR-ed into binary16 stores and
# conversions (ptx2rr.py --unguarded) and no NoContraction pass (spv_exact_convert.py) on those modules, so the driver may
# contract, fold and fuse there as it does in the relaxed native stages (engine/rr/README.md)
UNGUARDED = os.environ.get('D4R_RR2_UNGUARDED') == '1'
if UNGUARDED and RELAXED < 3:
    raise SystemExit('D4R_RR2_UNGUARDED=1 is for D4R_RR_RELAXED=3')
# decoder 0's second kernel on one wave (rr2_w.comp KIND 5, the translated tail on the same wave), which the
# D4R_RR2_UNGUARDED=1 build uses: dec0_1 393 -> 374 us at 1280x720 -> 1440p, the same images. In the guarded build its
# tail's channel kc[2] comes out as a NaN of the other sign on the stress inputs (a product of two NaNs, whose sign
# follows the compiler's operand order), so that build keeps the four-wave block. D4R_RR2_ONE_WAVE_DEC0_1=1 / 0 forces it.
if os.environ.get('D4R_RR2_ONE_WAVE_DEC0_1', '1' if UNGUARDED else '0') != '0' and RELAXED >= 3:
    ONE_WAVE.add('dec0_1')
# encoder 0's gather as a pass of its own, which pays with the four-wave block (one shader would run the gather on one
# wave of the four). With the one-wave block (level 3) the gather runs in the block's own shader and hands its
# features over in shared memory instead (rr2_w.comp RR2_XF_SHARED): no record in the arena, one dispatch less. Without
# the guards (D4R_RR2_UNGUARDED=1) the gather pass is faster again: encoder 0 650 -> 623 us at 1280x720 -> 1440p
# (2026-10-20; the guarded build is the same either way), so that build splits.
# D4R_RR2_SPLIT=1 / 0 forces the gather pass / the single shader.
SPLIT = os.environ.get('D4R_RR2_SPLIT', '0' if RELAXED >= 3 and 'enc0' in ONE_WAVE and not UNGUARDED else '1') != '0'


def find_module(ptx_dir, kernel):
    for path in sorted(Path(ptx_dir).glob('*.ptx')):
        if f'.entry {kernel}(' in path.read_text(errors='replace'):
            return path
    raise FileNotFoundError(f'no PTX module with the entry {kernel} in {ptx_dir}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('capture', nargs='?')
    ap.add_argument('output', type=Path)
    ap.add_argument('--dll', help="build without a capture, from the user's nvngx_dlssd.dll 310.9")
    ap.add_argument('--ptx-dir', default=os.path.expanduser('~/.cache/d4r-rr-corpus-f'))
    ap.add_argument('--frame', type=int, default=2, help='which captured frame supplies the templates (1-based)')
    ap.add_argument('--particles', metavar='CAPTURE_DIR',
                    help='a capture taken with DLSSD.ColorBeforeParticles: adds the downsample kernel NGX runs then (downp)')
    ap.add_argument('--native', metavar='DLL', help='nvngx_dlssd.dll 310.9: build the native stages from its weights')
    a = ap.parse_args()
    if a.dll:
        if a.capture or a.particles:
            ap.error('--dll takes no capture')
        from rr2_templates import ARENA, aux_table, templates
        dll = Path(a.dll).read_bytes()
        blocks, aux = templates(dll), aux_table(dll)
        a.native = a.native or a.dll
        cap = None
    elif not a.capture:
        ap.error('a capture directory, or --dll')
    else:
        cap = Capture(a.capture)
    out = a.output
    out.mkdir(parents=True, exist_ok=True)
    capped = ['systemd-run', '--user', '--scope', '-q', '-p', 'MemoryMax=24G', '-p', 'MemorySwapMax=0']
    # each compiler job in a memory-capped scope where systemd provides one (D4R_NO_SCOPE=1: none)
    if os.environ.get('D4R_NO_SCOPE') == '1' or subprocess.run(capped[:4] + ['true'], stderr=subprocess.DEVNULL,
                                                                 stdout=subprocess.DEVNULL).returncode:
        capped = []
    lines = ['D4RR0002', 'preset 6']
    pcap = Capture(a.particles) if a.particles else None
    for stage in STAGES + (['downp'] if pcap or a.dll else []):
        kernel = KERNEL[stage]
        network = stage in NETWORK
        if a.dll:
            launch = None
            if network and stage not in NATIVE and stage not in HYBRID:
                raise SystemExit(f'{stage}: no native stage, and its weights come from a capture')
        else:
            if stage == 'downp':
                cap = pcap
            launches = [l for l in cap.launches if l.kernel == kernel]
            per_frame = len(launches) // len([l for l in cap.launches if l.kernel == KERNEL['enc0']])
            launch = launches[(a.frame - 1) * per_frame]
        if a.native and stage in NATIVE:
            work = out / 'native-work' / stage
            work.mkdir(parents=True, exist_ok=True)
            weights = f'weights-{stage}m.bin'
            subprocess.run([sys.executable, str(HERE / 'rr' / 'rr2_pack.py'), stage, a.native, str(out / weights),
                            str(work / 'rr2_stage.h')], check=True, stdout=subprocess.DEVNULL)
            defs = ['-DRR_EXACT_CONVERT'] + (['-DRR_MIX'] if os.environ.get('D4R_RR_MIX') == '1' else []) + FUSE_DEFS + RELAX_DEFS
            subprocess.run(capped + ['glslangValidator', '-V', '--target-env', 'vulkan1.3', *defs, f'-I{work}',
                                     f'-I{HERE / "rr"}', str(HERE / 'rr' / ('rr2_w.comp' if RELAXED >= 3 and stage in ONE_WAVE else 'rr2_m.comp')), '-o', str(out / f'{stage}_m.spv')],
                           check=True, stdout=subprocess.DEVNULL)
            if RELAXED < 2:   # (the relaxed stages leave their conversions to the driver)
                subprocess.run([sys.executable, str(HERE / 'rr' / 'spv_exact_convert.py'), str(out / f'{stage}_m.spv')],
                               check=True, stdout=subprocess.DEVNULL)
            if launch is None:
                (out / f'{stage}.args').write_bytes(blocks[stage])
                ptrs = [ARENA[stage]]
            else:
                (out / f'{stage}.args').write_bytes(stage_params(launch, 24)[:len(launch.args)])
                arena = launch.arena_alloc
                unit = launch.allocs[arena][1] / 3152
                ptrs = [f'{p.offset}:{p.delta / unit:g}' for p in launch.pointers if p.alloc == arena and p.offset != 24]
            lines.append(f'stage {stage} {stage}_m.spv 32 {weights} {stage}.args - - - {",".join(ptrs)}')
            print(f'{stage}: native (engine/rr/rr2_m.comp)')
            continue
        generated = out / f'{stage}.generated.comp'
        cmd = [sys.executable, str(HERE / 'rr' / 'ptx2rr.py'), str(find_module(a.ptx_dir, kernel)), kernel, str(generated),
               '--block', BLOCK[stage], '--weights-arg', '24' if network else '9999']
        cmd += [f'--buffer={off}' for off in BUFFERS.get(stage, [])]
        cmd += [f'--snorm16={off}' for off in SNORM16.get(stage, [])]
        cmd += [f'--surface={off}={fmt}' for off, fmt in SURFACES.get(stage, {}).items()]
        # the hybrid stages' and the output passes' modules go through spv_exact_convert.py, so their conversions need
        # only the lighter guard (ptx2rr.py --exact-convert)
        exact = ((bool(a.native) and stage in HYBRID) or not network) and not UNGUARDED
        cmd += ['--exact-convert'] if exact else ['--unguarded'] if UNGUARDED else []
        subprocess.run(cmd, check=True, stderr=subprocess.DEVNULL)
        info = json.loads(generated.with_suffix('.json').read_text())
        weights = '-'
        hybrid = a.native and stage in HYBRID
        if hybrid:
            work = out / 'native-work' / stage
            work.mkdir(parents=True, exist_ok=True)
            weights = f'weights-{stage}m.bin'
            subprocess.run([sys.executable, str(HERE / 'rr' / 'rr2_pack.py'), stage, a.native, str(out / weights),
                            str(work / 'rr2_stage.h')], check=True, stdout=subprocess.DEVNULL)
            defs = ['-DRR_EXACT_CONVERT'] + (['-DRR_MIX'] if os.environ.get('D4R_RR_MIX') == '1' else []) + FUSE_DEFS + RELAX_DEFS
            # encoder 0 in two passes (gen_rr2_enc0.py --split): its gather as the stage enc0g, whose records' place
            # (three ranges) is six words appended to the argument block
            split = stage == 'enc0' and SPLIT and GROUP == 1
            n_args = len(blocks[stage]) if launch is None else len(launch.args)
            how = ['--split', 'block', '--xf-word', str((n_args + 7) // 8 * 2)] if split else ['--group', str(GROUP)]
            if split:
                subprocess.run([sys.executable, str(HERE / 'rr' / 'gen_rr2_enc0.py'), str(find_module(a.ptx_dir, kernel)),
                                str(generated), str(work / 'enc0_g.comp'), '--split', 'gather', *how[2:]],
                               check=True, stdout=subprocess.DEVNULL)
                subprocess.run(capped + ['glslangValidator', '-V', '--target-env', 'vulkan1.3', *defs, f'-I{work}',
                                         f'-I{HERE / "rr"}', str(work / 'enc0_g.comp'), '-o', str(out / 'enc0g.spv')],
                               check=True, stdout=subprocess.DEVNULL)
                if not UNGUARDED:
                    subprocess.run([sys.executable, str(HERE / 'rr' / 'spv_exact_convert.py'), str(out / 'enc0g.spv')],
                                   check=True, stdout=subprocess.DEVNULL)
                record = int((work / 'enc0_g.record').read_text())
            hybrid_args = HYBRID[stage]
            if RELAXED >= 3 and stage in ONE_WAVE and (split or GROUP == 1):
                how = how + ['--one-wave']
                if stage == 'dec0_1':   # the tail runs on the block's one wave
                    hybrid_args = ['--dec0_1', '--tail-waves', '1']
            subprocess.run([sys.executable, str(HERE / 'rr' / 'gen_rr2_enc0.py'), str(find_module(a.ptx_dir, kernel)),
                            str(generated), str(work / f'{stage}_m.comp'), *hybrid_args, *how],
                           check=True, stdout=subprocess.DEVNULL)
            subprocess.run(capped + ['glslangValidator', '-V', '--target-env', 'vulkan1.3', *defs, f'-I{work}',
                                     f'-I{HERE / "rr"}', str(work / f'{stage}_m.comp'), '-o', str(out / f'{stage}.spv')],
                           check=True, stdout=subprocess.DEVNULL)
            if not UNGUARDED:
                subprocess.run([sys.executable, str(HERE / 'rr' / 'spv_exact_convert.py'), str(out / f'{stage}.spv')],
                               check=True, stdout=subprocess.DEVNULL)
        else:
            subprocess.run(capped + ['glslangValidator', '-V', '--target-env', 'vulkan1.3', str(generated), '-o',
                                     str(out / f'{stage}.spv')], check=True, stdout=subprocess.DEVNULL)
            if exact:
                subprocess.run([sys.executable, str(HERE / 'rr' / 'spv_exact_convert.py'), str(out / f'{stage}.spv')],
                               check=True, stdout=subprocess.DEVNULL)
        if hybrid:
            pass
        elif network:
            weights = f'weights-{stage}.bin'
            (out / weights).write_bytes(launch.weights_at(24))
        elif generated.with_suffix('.globals.bin').exists():
            weights = f'weights-{stage}.bin'
            (out / weights).write_bytes(generated.with_suffix('.globals.bin').read_bytes())
        if stage == 'enc0':
            (out / 'enc0-aux.bin').write_bytes(aux if launch is None else launch.pre(launch.pointer(104).alloc))
        args = blocks[stage] if launch is None else stage_params(launch, 24 if network else 9999)[:len(launch.args)]
        if hybrid and split:
            args = args + bytes(-len(args) % 8) + bytes(24)   # the gather records' three ranges (the runtime's)
        (out / f'{stage}.args').write_bytes(args)
        # the stage's arena regions, in units of 1/3152 of NGX's arena (W0 * H0 / 32 bytes: the regions scale with it)
        if launch is None:
            ptrs = [ARENA[stage]] if network else []
        else:
            arena = launch.arena_alloc
            unit = launch.allocs[arena][1] / 3152 if arena is not None else 1
            ptrs = [f'{p.offset}:{p.delta / unit:g}' for p in launch.pointers
                    if p.alloc == arena and p.offset not in BUFFERS.get(stage, []) and not (network and p.offset == 24)]
        if not network:
            ptrs = []   # the output passes' regions are the runtime's own
        tex = ','.join(map(str, info['textures'])) or '-'
        surf = ','.join(f'{off}:{fmt}' for off, fmt in info['surfaces']) or '-'
        buf = ','.join(map(str, info['buffers'])) or '-'
        # the output passes have no 32-lane matrix products: 64-wide waves (their warp-level shuffles address the
        # lane's own half, engine/rr/ptx2rr.py)
        wave = 32 if network else 64
        lines.append(f'stage {stage} {stage}.spv {wave} {weights} {stage}.args {tex} {surf} {buf} {",".join(ptrs) or "-"}' +
                     (f' x{GROUP}' if hybrid and GROUP > 1 else ''))   # windows per workgroup (engine/rr/gen_rr2_enc0.py)
        if hybrid and split:   # the same block, bindings and regions; "rN": N words per window record, "s3": in three ranges
            lines.append(f'stage enc0g enc0g.spv 32 - {stage}.args {tex} {surf} {buf} {",".join(ptrs) or "-"} r{record} s3')
        print(f'{stage}: {kernel}, {len(args)} argument bytes' + (' (native block)' if hybrid else ''))
    for p in (0, 1, 2):
        subprocess.run(['glslangValidator', '-V', '--target-env', 'vulkan1.3', f'-DPASS={p}',
                        str(HERE / 'rr' / 'exposure_f.comp'), '-o', str(out / f'exposure_f{p}.spv')],
                       check=True, stdout=subprocess.DEVNULL)
        # the driver folds float(float16_t(x)) away unless the conversions are marked (engine/rr/spv_exact_convert.py):
        # the kernel truncates the exposure to binary16
        subprocess.run([sys.executable, str(HERE / 'rr' / 'spv_exact_convert.py'), str(out / f'exposure_f{p}.spv')],
                       check=True, stdout=subprocess.DEVNULL)
    (out / 'rr.txt').write_text('\n'.join(lines) + '\n')
    print(f'Ray Reconstruction preset F model in {out}')


if __name__ == '__main__':
    main()
