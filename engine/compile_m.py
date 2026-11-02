#!/usr/bin/env python3
"""Build the engine's preset M or preset L model directory from a capture of one frame.

usage: compile_m.py CAPTURE_DIR OUT_DIR [--frame N]
  CAPTURE_DIR/replay must hold the frame's rrlite_enc0 and rrlite_dec0 launches and its ten Swin layer launches,
  with their weight allocations (D4R_CUDA_REPLAY_DUMP_DIR, filter `rrlite_`). A capture of preset M has the `_folded`
  input and expansion kernels; one of preset L (DLSS preset 12) has the unfolded ones, and gets an L model directory
  (offsets.bin `D4RL0002`, L_NOTES.md): the same network, layer_m.comp's L input and expansion stages, post_m.comp's
  L reconstruction.
Writes weights8.bin (the layers' FP8 matrices in the blocked fixed-16 tile layout of test_m.pack8, with the front-end
enc0/dec0 matrices plain and NGX-ordered as their own stages read them), weights16.bin (f16 vectors and the dither
table), offsets.bin and the compiled shaders. The weights are values read from NVIDIA's DLSS and are not redistributable
with d4r; the shaders are d4r's own.
"""
# D4RM0007 makes the dot2 expansion stage's and decoders' expansion products int8 (dec0_m.comp INT8, layer_m_dot2.comp
# EXPAND_INT8; runtime_m.inc widenM packs them so).
# D4RM0006 makes the dot2 layers' Q and V int8 (runtime_m.inc widenM packs them so; older packages keep binary16).
# D4RM0005 adds display-resolution motion to M (enc0_m.comp displayMv, post_m.comp binding 12; NGX's *_mvhi_* kernels).
# D4RM0004 / D4RL0002 add active-rectangle sampling for dynamic input sizes.
# D4RM0003 coupled the blocked FP8 operands, RGBA32F half-history backing and 8x64 downsample dispatch;
# the previous package's shader workgroup dimensions are not interchangeable with this runtime.
import argparse, glob, hashlib, json, os, shutil, struct, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import m_model as mm
import l_model as lm
import model_enc3 as me
import swin_model as sm
import m_d3d12_build
from test_m import pack_layer, defines, dot2_defines, dot2_merge_defines
import hlsl_build

# The network is built with this GPU's own arithmetic and two windows per wave by default (layer_m.comp M_RELAXED, WPW;
# engine/README.md): the fastest build, about 1 dB further from an NVIDIA card than NVIDIA's roundings reproduced
# exactly. M_EXACT=1 builds those roundings instead (bit-identical to the CUDA path); M_WPW / M_DEFS still apply.
EXACT = os.environ.get('M_EXACT') == '1'
FAST = [] if EXACT or '-DM_RELAXED' in os.environ.get('M_DEFS', '').split() else ['-DM_RELAXED']
FAST_WPW = [] if EXACT or os.environ.get('M_WPW') or 'WPW=' in os.environ.get('M_DEFS', '') else ['-DWPW=2']
# preset L's expansion stage (dec0l) two windows per wave as well, and its input stage (enc0l) since that keeps its FP8
# buffers in its binary16 one (7.5 KiB of shared memory, layer_m.comp): 445 -> 415 us at 1280x720, the same values
L_WPW = [] if EXACT or 'WPW=' in os.environ.get('M_DEFS', '') + os.environ.get('M_DEFS_DEC0L', '') else ['-DWPW=2']
L_WPW_ENC0 = [] if EXACT or 'WPW=' in os.environ.get('M_DEFS', '') + os.environ.get('M_DEFS_ENC0L', '') else ['-DWPW=2']
# M_RESIDUAL_F32 (on by default in the relaxed build; M_RESIDUAL_F32=0 turns it off, and M_EXACT=1 implies 0): every
# layer_m.comp stage keeps the residual stream f32 across the MLP's 32-channel chunks, one binary16 rounding after them
# instead of one per chunk (engine/README.md)
RESF32 = ['-DM_RESIDUAL_F32'] if os.environ.get('M_RESIDUAL_F32', '0' if EXACT else '1') == '1' else []
if RESF32 and EXACT:
    raise SystemExit('M_RESIDUAL_F32 is part of the relaxed arithmetic: not with M_EXACT=1')
# preset L's input and expansion stages (enc0_l, dec0_l) in the relaxed build: their vector work trimmed for this GPU
# (layer_m.comp M_F32ACT, M_RES32, M_PK16, M_BIAS32, M_LDAU; docs/amd-vector-issue-limit.md): 4.24 -> 4.03 ms at
# 1920x1080 -> 3840x2160 on an RX 9070 XT. M_L_FAST=0 builds them as before.
L_FAST = [] if EXACT or os.environ.get('M_L_FAST', '1') == '0' else \
    ['-DM_F32ACT', '-DM_PK16', '-DM_BIAS32', '-DM_LDAU'] + (['-DM_RES32'] if RESF32 else [])
SHADERS = {'enc1': 'm_enc1', 'enc2': 'm_enc2', 'enc3_tube': 'm_tube', 'dec2': 'm_dec2', 'dec1': 'm_dec1'}


def build_dxil(output, ldr, preset_l=False, target='windows'):
    """the D3D12 backend's shaders (d3d12/, AMD's driver): the network layers from d3d12/layer_m.hlsl (AGS WMMA, the
    relaxed arithmetic; M_EXACT=1 selects NVIDIA's roundings), the other stages from their GLSL
    (hlsl_build.py). M_D3D12_DEFS adds layer_m.hlsl defines. Preset L: its input stage as two shaders, the feature
    slots (enc0_m.comp L_FEATURES, enc0_lf) and the embedding, block and patch merge (layer_m.hlsl KIND 3, enc0_l), its
    expansion (KIND 4, dec0_l) and the reconstruction's L tiers (post_l*).
    target 'windows' is AMD's D3D12 driver. target 'vkd3d' (vkd3d-proton with patches/vkd3d-proton, on RADV) reaches
    the AGS matrices' scalar registers through AGS element operations (ags_wmma.hlsli D4R_AGS_SCALAR_REGISTERS): the
    shaders that use AGS WMMA (the layers, enc0/dec0, enc0_lf) keep the same arithmetic. Relaxed vkd3d network builds
    use O1 and two windows per wave in the encoders, decoders and L input/expansion; tubes keep one window. Exact and
    Windows targets retain their original settings. RX 9070 XT, 2026-10-25: against
    Vulkan built from the same source, final RGB 47.6 dB (M, 854x480 -> 2560x1440) and 50.3 dB (L, 1280x720 ->
    2560x1440) over four frames; the native binary16 conversion (D4R_NATIVE_HALF) rounds toward zero on both D3D12
    implementations, and with emulated rounding M agrees at 51.9 dB, Vulkan's own relaxed against exact build."""
    if target not in ('windows', 'vkd3d'):
        raise ValueError('D3D12 target must be windows or vkd3d')
    dxc = hlsl_build.dxc_path()
    exact = ['-DM_EXACT=1'] if EXACT else []
    # D4R_VKD3D: RADV compiles the shaders, as for Vulkan, so they skip the workarounds for AMD's D3D12 compiler
    # (binary16 casts round to nearest even there; post_m.comp D4R_AMD_DXC, m_wmma.hlsli fp8Of)
    ags = ['-DD4R_AGS_SCALAR_REGISTERS=1', '-DD4R_VKD3D=1'] if target == 'vkd3d' else []
    tuned = target == 'vkd3d' and not EXACT
    # O1 avoids DXC's expensive optimization of expanded AGS element operations. RADV still optimizes
    # the resulting SPIR-V.
    # Two windows per wave, f32 accumulators outside the tube (layer_m.hlsl M_ACC32; the tube's f32 Q and V lose 1.2 dB
    # over 32 recurrent frames). The tube also as m_tube_w2 with two windows, which the runtime takes for large
    # dispatches (runtime_m_d3d12.cpp MTubeW2Groups), m_tube keeping one. RX 9070 XT, M 1280x720 -> 3840x2160:
    # network 1009 -> 874 us with the saturating FP8 conversion, Vulkan 829 us.
    layers = dict(SHADERS)
    if preset_l: layers.update({'enc0l': 'enc0_l', 'dec0l': 'dec0_l'})
    builds = [(name, out, None) for name, out in layers.items()]
    if tuned: builds.append(('enc3_tube', 'm_tube_w2', 2))
    for name, out, wpw in builds:
        C, heads, NPM, CIN = mm.LAYERS[name]
        kind = {'enc0l': 3, 'dec0l': 4}.get(name, 1 if NPM else 2 if CIN else 0)
        schedule = ([f'-DWPW={wpw or (1 if name == "enc3_tube" else 2)}'] + ([] if name == 'enc3_tube' else ['-DM_ACC32=1'])) if tuned else []
        args = [dxc, '-nologo', '-T', 'cs_6_6', '-E', 'main', '-HV', '2021', '-enable-16bit-types',
                '-O1' if tuned else '-O3',
                '-I', str(HERE / 'd3d12'), f'-DC={C}', f'-DNH={heads}', f'-DKIND={kind}', f'-DXC={NPM or CIN or 16}',
                *exact, *ags, *schedule, *os.environ.get('M_D3D12_DEFS', '').split(), str(HERE / 'd3d12' / 'layer_m.hlsl')]
        subprocess.run(args + ['-Fo', str(output / (out + '.dxil'))], check=True)
        # L's input and expansion stages again with the token buffers in NGX's plane layout, for the native HIP network
        # (layer_m.hlsl PLANES; the same flags as the stage above, as the SPIR-V plane builds are layer_m.comp's)
        if preset_l and name in ('enc0l', 'dec0l'):
            subprocess.run(args + ['-DPLANES=1', '-Fo', str(output / (out + '_planes.dxil'))], check=True)
    ldr_define = ['-DLDR=1'] if ldr else []
    if preset_l and tuned:
        # vkd3d-proton: L's input stage also in one dispatch (enc0_lx, layer_m.hlsl KIND 3 with L_FUSED and
        # d3d12/enc0_l_fused.hlsli): the layer gathers its block's feature slots itself, as the Vulkan build's
        # layer_m.comp does, instead of the two-shader stage's buffer round trip (32 bytes a pixel each way)
        C, heads, NPM, CIN = mm.LAYERS['enc0l']
        with tempfile.TemporaryDirectory(prefix='d4r-enc0lx.') as tmp:
            features = HERE / 'enc0_features.glsl'
            lines, _ = hlsl_build.translate(features.read_text(), False, HERE, features.as_posix())
            (Path(tmp) / 'enc0_features_hlsl.hlsli').write_text('\n'.join(lines) + '\n')
            subprocess.run([dxc, '-nologo', '-T', 'cs_6_6', '-E', 'main', '-HV', '2021', '-enable-16bit-types', '-O1',
                            '-I', str(HERE / 'd3d12'), '-I', str(HERE), '-I', tmp, f'-DC={C}', f'-DNH={heads}', '-DKIND=3',
                            f'-DXC={NPM or CIN or 16}', '-DD4R_WAVE_SIZE=32', '-DL_FUSED=1', '-DWPW=2', '-DM_ACC32=1',
                            *ags, *ldr_define, *os.environ.get('M_D3D12_DEFS', '').split(),
                            str(HERE / 'd3d12' / 'layer_m.hlsl'), '-Fo', str(output / 'enc0_lx.dxil')], check=True)
    # the reconstruction and the downsample convert to binary16 on the hardware (round to nearest even through f32tof16,
    # dialect_hlsl.h D4R_NATIVE_HALF) instead of the emulated rounding: RX 9070 XT, 854x480 -> 2560x1440, 326 -> 304 us
    # and 107 -> 91 us, the same PSNR against the ZLUDA capture. The input and expansion stages keep the emulation (0.1 dB
    # each for 2-4 us: their values become FP8 network inputs), as does M_EXACT
    # vkd3d: the plain casts, which round to nearest even there (RX 9070 XT, M reconstruction 657 -> 597 us, Vulkan 565)
    native = ['-DD4R_VKD3D=1'] if target == 'vkd3d' else [] if EXACT else ['-DD4R_NATIVE_HALF=1']
    if preset_l:
        hlsl_build.build(HERE / 'enc0_m.comp', output / 'enc0_lf.dxil', ['-DL_FEATURES=1'] + ldr_define + ags, wave=32)
        # the reconstruction one render texel wider per tier than M's (compile_l)
        tiers = (('post_l', 11), ('post_l_q', 16), ('post_l_c', 12))
        for out, tile in tiers:
            hlsl_build.build(HERE / 'post_m.comp', output / (out + '.dxil'),
                             ['-DPRESET_L=1', f'-DTILE_W={tile}', f'-DTILE_H={tile}'] + ldr_define + exact + native, wave=64)
    else:
        hlsl_build.build(HERE / 'enc0_m.comp', output / 'enc0_m.dxil', ldr_define + ags, wave=32)
        hlsl_build.build(HERE / 'dec0_m.comp', output / 'dec0_m.dxil', ags, wave=32)
        # the same stages with the token buffers in NGX's plane layout, for the native HIP network (runtime_m_d3d12.cpp)
        hlsl_build.build(HERE / 'enc0_m.comp', output / 'enc0_m_planes.dxil', ['-DPLANES=1'] + ldr_define + ags, wave=32)
        hlsl_build.build(HERE / 'dec0_m.comp', output / 'dec0_m_planes.dxil', ['-DPLANES=1'] + ags, wave=32)
        for out, flags in (('post_m', []), ('post_m_q', ['-DTILE_W=15', '-DTILE_H=15']), ('post_m_c', ['-DTILE_W=11', '-DTILE_H=11'])):
            hlsl_build.build(HERE / 'post_m.comp', output / (out + '.dxil'), flags + ldr_define + exact + native, wave=64)
    for out, flags in (('down_m', []), ('down_m_rgb10', ['-DOUTPUT_RGB10=1'])):
        hlsl_build.build(HERE / 'down_m.comp', output / (out + '.dxil'), flags + ldr_define + native, wave=64)
    for p in '01':
        hlsl_build.build(HERE / 'exposure_k.comp', output / f'exposure_m{p}.dxil', ['-DMODE=1', '-DPASS=' + p], wave=64)


def blob_of(directory, ptr_offset=0):
    args = open(f'{directory}/args.bin', 'rb').read()
    ptr = struct.unpack_from('<Q', args, ptr_offset)[0]
    for line in open(f'{directory}/manifest.txt'):
        q = line.split()
        if q and q[0] == 'alloc':
            base = int(q[2], 16); size = os.path.getsize(f'{directory}/alloc-{q[1]}.bin')
            if base <= ptr < base + size:
                return np.fromfile(f'{directory}/alloc-{q[1]}.bin', np.uint8)[ptr - base:]
    raise ValueError(f'{directory}: the weights allocation is missing')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('capture'); ap.add_argument('output', type=Path); ap.add_argument('--frame', type=int, default=0)
    ap.add_argument('--ldr', action='store_true', help="NGX's *_ldr_* stages, for games without the HDR flag (the weights may come from an HDR capture)")
    ap.add_argument('--hip', type=Path, metavar='KERNEL_DIR', help='also package the network for the native HIP layers: the layer '
                    'weights as NGX uploads them (hipnet.bin) and the rrlite_*_4x4.hsaco kernels of KERNEL_DIR (kernels/build.sh output)')
    ap.add_argument('--dxil', choices=('auto', 'yes', 'no'), default='auto', help='also build the D3D12 backend DXIL '
                    '(build_dxil, needs dxc): auto builds it on Windows when dxc is found')
    ap.add_argument('--rdna3', choices=('auto', 'yes', 'no'), default='auto', help='also package the RDNA 3 D3D12 backend (FP16 WMMA with '
                    'software FP8 storage, d3d12_ml.bin; m_d3d12_build.py): auto builds it on Windows when dxc is found')
    ap.add_argument('--rdna3-math', choices=('fast', 'fp8'), default='fast', help='RDNA 3 network arithmetic; fast follows the Linux HIP '
                    'fast switches (no intermediate FP8 rounding, f32 accumulation)')
    ap.add_argument('--d3d12-target', choices=('windows', 'vkd3d'), default='windows', help='the D3D12 implementation '
                    'the DXIL is built for: AMD\'s Windows driver, or vkd3d-proton with patches/vkd3d-proton (see build_dxil)')
    a = ap.parse_args()
    dxil = a.dxil == 'yes'
    if a.dxil == 'auto' and os.name == 'nt':
        try:
            hlsl_build.dxc_path()
            dxil = True
        except FileNotFoundError:
            print('note: dxc not found; the package has no D3D12 shaders', file=sys.stderr)
    a.dxil = dxil
    a.rdna3_enabled = a.rdna3 == 'yes' or (a.rdna3 == 'auto' and dxil)
    a.output.mkdir(parents=True, exist_ok=True)
    def one(kernel):
        d = sorted(glob.glob(f'{a.capture}/replay/replay-*-{kernel}'))
        if len(d) <= a.frame: raise ValueError(f'no launch of {kernel} for frame {a.frame}')
        return d[a.frame]
    R = 'ldr' if a.ldr else 'hdr'
    # NGX's HDR and LDR stages read the same weights and constants (RE Requiem's m / m-ldr and l / l-ldr packages,
    # built from separate captures, are identical but for the shaders): an LDR model can come from an HDR capture
    if a.ldr and not glob.glob(f'{a.capture}/replay/replay-*-rrlite_enc0_4x4_mvlo_ldr*'):
        R = 'hdr'
    preset_l = bool(glob.glob(f'{a.capture}/replay/replay-*-rrlite_enc0_4x4_mvlo_{R}'))
    if preset_l == bool(glob.glob(f'{a.capture}/replay/replay-*-rrlite_enc0_4x4_mvlo_{R}_folded')):
        raise ValueError(f'the capture must hold the input stage of exactly one of preset M or L ({R})')
    if preset_l:
        return compile_l(a, one, R)
    w8, w16 = [], []
    n8 = n16 = 0
    def add8(x):
        nonlocal n8
        x = np.ascontiguousarray(x, np.uint8).ravel(); pad = (-x.size) % 64
        off = n8; w8.append(x); w8.append(np.zeros(pad + 64, np.uint8)); n8 += x.size + pad + 64
        return off
    def add16(x):
        nonlocal n16
        x = np.ascontiguousarray(x).view(np.uint8).ravel(); pad = (-x.size) % 64
        off = n16; w16.append(x); w16.append(np.zeros(pad + 64, np.uint8)); n16 += x.size + pad + 64
        return off
    table = []      # (w8 offset, w16 offset) per stage, then the layers' offset blocks
    # input stage: embedding [128][64] in natural output order, bias tile, dither table
    wb = blob_of(one(f'rrlite_enc0_4x4_mvlo_{R}_folded'))
    g = sm.gp(np.arange(64))
    enc0 = (add8(wb[sm.woff_table(4096, 512, 2048, 128, 64)][:, g]), add16(np.tile(wb[12288:12288 + 128].view(np.float16)[g][None, :], (16, 1))), add16(wb[:4096]))
    # expansion: the 20 columns per sub-position the stage uses
    wb = blob_of(one('rrlite_dec0_4x4_folded'))
    cols = (np.arange(4)[:, None] * 32 + np.arange(20)[None, :]).ravel()
    dec0 = (add8(wb[sm.woff_table(0, 512, 1024, 64, 128)][:, cols]), add16(np.tile(wb[8192:8192 + 256].view(np.float16)[cols][None, :], (16, 1))))
    Ps, dirs = mm.launches(a.capture, a.frame)
    layers = []
    for P, name in zip(Ps, mm.ORDER):
        o, b8, b16 = pack_layer(mm.plain_layer(P, name), name)
        layers.append((add8(b8), add16(b16), o))
    blob8, blob16 = np.concatenate(w8), np.concatenate(w16)
    blob8.tofile(a.output / 'weights8.bin'); blob16.tofile(a.output / 'weights16.bin')
    head = struct.pack('<8sII', b'D4RM0007', blob8.size, blob16.size) + struct.pack('<5I', *enc0, *dec0)
    for o8, o16, o in layers: head += struct.pack('<2I72I', o8, o16, *o)
    (a.output / 'offsets.bin').write_bytes(head)
    def glsl(src, out, flags=()):
        subprocess.run(['glslangValidator', '-V', '--target-env', 'vulkan1.3', *flags, str(HERE / src), '-o', str(a.output / out)], check=True, stdout=subprocess.DEVNULL)
    # the network in the arithmetic of ZLUDA's lowering of NVIDIA's PTX (TEXL): byte-identical to the CUDA backend
    # with NVIDIA's translated kernels in the harness (2026-10-11), where the native accuracy kernels' K32_MIX order
    # drifted to 36-41 dB through the FP8 tube
    for name, out in SHADERS.items(): glsl('layer_m.comp', out + '.spv', defines(name) + ['-DTEXL=1'] + FAST + FAST_WPW + RESF32)
    for s in ('enc0_m', 'dec0_m', 'post_m', 'down_m'): glsl(s + '.comp', s + '.spv', ['-DLDR=1'] if a.ldr and s != 'dec0_m' else [])
    glsl('down_m.comp', 'down_m_rgb10.spv', ['-DOUTPUT_RGB10=1'] + (['-DLDR=1'] if a.ldr else []))
    # the final store into an R11G11B10F output (Indiana Jones on its native Vulkan route)
    glsl('down_m.comp', 'down_m_r11g11b10.spv', ['-DOUTPUT_R11G11B10=1'] + (['-DLDR=1'] if a.ldr else []))
    glsl('post_m.comp', 'post_m_q.spv', ['-DTILE_W=15', '-DTILE_H=15'] + (['-DLDR=1'] if a.ldr else []))
    glsl('post_m.comp', 'post_m_c.spv', ['-DTILE_W=11', '-DTILE_H=11'] + (['-DLDR=1'] if a.ldr else []))
    # the same stages with the token buffers in NGX's plane layout, for a network that runs outside the engine
    for s in ('enc0_m', 'dec0_m'): glsl(s + '.comp', s + '_planes.spv', ['-DPLANES=1'] + (['-DLDR=1'] if a.ldr and s != 'dec0_m' else []))
    # the network, input and expansion stages for GPUs without cooperative matrices or FP8 (RDNA2): v_dot2 on the
    # matrices widened to binary16 at load (layer_m_dot2.comp, runtime_m.inc widenM), the relaxed build's arithmetic
    for name, out in SHADERS.items(): glsl('layer_m_dot2.comp', out + '_dot2.spv', dot2_defines(name))
    for name, out in (('enc1', 'm_merge1'), ('enc2', 'm_merge2')): glsl('merge_m_dot2.comp', out + '_dot2.spv', dot2_merge_defines(name))
    glsl('enc0_m.comp', 'enc0_m_dot2.spv', ['-DDOT2=1'] + (['-DLDR=1'] if a.ldr else []))
    glsl('dec0_m.comp', 'dec0_m_dot2.spv', ['-DDOT2=1', '-DINT8=1'])
    # the reconstruction with its half-resolution mean in RGB10A2 (post_m.comp HALF_RGB10)
    for out, flags in (('post_m_dot2', []), ('post_m_q_dot2', ['-DTILE_W=15', '-DTILE_H=15']), ('post_m_c_dot2', ['-DTILE_W=11', '-DTILE_H=11'])):
        glsl('post_m.comp', out + '.spv', ['-DHALF_RGB10=1'] + flags + (['-DLDR=1'] if a.ldr else []))
    for out, flags in (('down_m_dot2', []), ('down_m_rgb10_dot2', ['-DOUTPUT_RGB10=1']),
                       ('down_m_r11g11b10_dot2', ['-DOUTPUT_R11G11B10=1'])):
        glsl('down_m.comp', out + '.spv', ['-DSEG=2'] + flags + (['-DLDR=1'] if a.ldr else []))
    if a.hip:
        package_hip(a, Ps)
    marker = a.output / 'ldr'      # tells the runtime to skip exposure
    if a.ldr: marker.write_text('1\n')
    elif marker.exists(): marker.unlink()
    for p in '01': glsl('exposure_k.comp', f'exposure_m{p}.spv', ['-DMODE=1', '-DPASS=' + p])
    if a.dxil: build_dxil(a.output, a.ldr, target=a.d3d12_target)
    package_rdna3(a)
    files = (sorted(a.output.glob('*.bin')) + sorted(a.output.glob('*.spv')) + sorted(a.output.glob('*.dxil')) + sorted(a.output.glob('rdna3/*.dxil')) + sorted(a.output.glob('rdna3/*.bin')) +
             sorted(a.output.glob('hip/*.hsaco')))
    (a.output / 'model.json').write_text(json.dumps({'format': 'D4RM0007', 'preset': 'M', 'files': {str(f.relative_to(a.output)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}}, indent=1))
    print(f'Compiled native preset M model in {a.output} ({blob8.size} FP8 bytes, {blob16.size} bytes of f16 data)')


def compile_l(a, one, R):
    """preset L: the network's layers plus the input (KIND 3) and expansion (KIND 4) stages, each packed as a layer"""
    w8, w16 = [], []
    n8 = n16 = 0
    def add(lst, x, n):
        x = np.ascontiguousarray(x).view(np.uint8).ravel(); pad = (-x.size) % 64
        lst.append(x); lst.append(np.zeros(pad + 64, np.uint8)); return x.size + pad + 64
    blocks = []
    def layer(t, name):
        nonlocal n8, n16
        o, b8, b16 = pack_layer(t, name)
        blocks.append((n8, n16, o)); n8 += add(w8, b8, n8); n16 += add(w16, b16, n16)
    wb = blob_of(one(f'rrlite_enc0_4x4_mvlo_{R}'))
    dither = n16; n16 += add(w16, wb[:4096], n16)
    layer(lm.enc0_layer(wb), 'enc0l')
    Ps, dirs = mm.launches(a.capture, a.frame)
    for P, name in zip(Ps, mm.ORDER): layer(mm.plain_layer(P, name), name)
    layer(lm.dec0_layer(blob_of(one('rrlite_dec0_4x4'))), 'dec0l')
    blob8, blob16 = np.concatenate(w8), np.concatenate(w16)
    blob8.tofile(a.output / 'weights8.bin'); blob16.tofile(a.output / 'weights16.bin')
    # the stages' constants, as NGX passes them (they differ from preset M's): the input stage's motion-change scale and
    # gain, the expansion's covariance sigma and rho, the reconstruction's first-gate gain
    arg = lambda kernel, off: struct.unpack_from('<f', open(f'{one(kernel)}/args.bin', 'rb').read(), off)[0]
    consts = (arg(f'rrlite_enc0_4x4_mvlo_{R}', 268), arg(f'rrlite_enc0_4x4_mvlo_{R}', 272), arg('rrlite_dec0_4x4', 160),
              arg('rrlite_dec0_4x4', 164), arg(next(k for k in (f'rrlite_post_3_1_mvlo_{R}', f'rrlite_post_3_2_mvlo_{R}')
                                                       if glob.glob(f'{a.capture}/replay/replay-*-{k}')), 120))
    head = struct.pack('<8sIII5f', b'D4RL0002', blob8.size, blob16.size, dither, *consts)
    for o8, o16, o in blocks: head += struct.pack('<2I72I', o8, o16, *o)
    (a.output / 'offsets.bin').write_bytes(head)
    def glsl(src, out, flags=()):
        subprocess.run(['glslangValidator', '-V', '--target-env', 'vulkan1.3', *flags, str(HERE / src), '-o', str(a.output / out)], check=True, stdout=subprocess.DEVNULL)
    ldr = ['-DLDR=1'] if a.ldr else []
    # the network in the arithmetic of ZLUDA's lowering of NVIDIA's PTX (TEXL), as for preset M (L_NOTES.md)
    for name, out in SHADERS.items(): glsl('layer_m.comp', out + '.spv', defines(name) + ['-DTEXL=1'] + FAST + FAST_WPW + RESF32)
    for planes, suffix in ((0, ''), (1, '_planes')):
        glsl('layer_m.comp', f'enc0_l{suffix}.spv', defines('enc0l') + ldr + [f'-DPLANES={planes}'] + FAST + L_WPW_ENC0 + RESF32 + L_FAST)
        glsl('layer_m.comp', f'dec0_l{suffix}.spv', defines('dec0l') + [f'-DPLANES={planes}'] + FAST + L_WPW + RESF32 + L_FAST)
    # the reconstruction one render texel wider per tier than M's (4x4 taps): 11 (reconstruction/render >= 3),
    # 16 (the whole post_3_2 range), 12 (the compact tier: render/reconstruction <= 0.45, jitter within half a texel)
    for out, tile in (('post_l', 11), ('post_l_q', 16), ('post_l_c', 12)):
        glsl('post_m.comp', out + '.spv', ['-DPRESET_L=1', f'-DTILE_W={tile}', f'-DTILE_H={tile}'] + ldr)
    glsl('down_m.comp', 'down_m.spv', ldr)
    glsl('down_m.comp', 'down_m_rgb10.spv', ['-DOUTPUT_RGB10=1'] + ldr)
    glsl('down_m.comp', 'down_m_r11g11b10.spv', ['-DOUTPUT_R11G11B10=1'] + ldr)
    # the stages for GPUs without cooperative matrices or FP8 (RDNA2): preset M's network, the input stage
    # (layer_l_dot2.comp KIND 3, its patch merge merge_l_dot2.comp) and the expansion stage (KIND 4) on v_dot2 / v_dot4
    # (runtime_m.inc widenM: int8 as preset M's D4RM0007), the reconstruction's half-resolution mean in RGB10A2; these
    # take display-resolution motion as M's do (post_m.comp DISPLAY_MV, enc0_features.glsl)
    for name, out in SHADERS.items(): glsl('layer_m_dot2.comp', out + '_dot2.spv', dot2_defines(name))
    for name, out in (('enc1', 'm_merge1'), ('enc2', 'm_merge2')): glsl('merge_m_dot2.comp', out + '_dot2.spv', dot2_merge_defines(name))
    glsl('layer_l_dot2.comp', 'enc0_l_dot2.spv', ['-DKIND=3'] + ldr)
    glsl('merge_l_dot2.comp', 'merge_l_dot2.spv')
    glsl('layer_l_dot2.comp', 'dec0_l_dot2.spv', ['-DKIND=4'])
    # the same stages with the token buffers in NGX's plane layout, for the native HIP network (vkhip/HIP, runtime_m.inc):
    # AMD's Windows driver (LLPC) crashes compiling the FP8 cooperative-matrix plane stages (enc0_l_planes.spv) and rejects
    # SPV_VALVE_mixed_float_dot_product (capability 6912) when a shader module is created, so preset L takes these there
    # (runtime_m.inc: the driver gate) and they are built with DOT2_SOFT (the f16 pair products as f32 FMAs of exact f16
    # products; the int8 dot4 and the ballot swizzle are the intrinsics). The merge is the dot2 route's writer of the
    # network's tokens.
    glsl('layer_l_dot2.comp', 'enc0_l_planes_dot2.spv', ['-DKIND=3', '-DPLANES=1', '-DDOT2_SOFT=1'] + ldr)
    glsl('merge_l_dot2.comp', 'merge_l_dot2_planes.spv', ['-DPLANES=1'])
    glsl('layer_l_dot2.comp', 'dec0_l_planes_dot2.spv', ['-DKIND=4', '-DPLANES=1', '-DDOT2_SOFT=1'])
    for out, tile in (('post_l_dot2', 11), ('post_l_q_dot2', 16), ('post_l_c_dot2', 12)):
        glsl('post_m.comp', out + '.spv', ['-DPRESET_L=1', '-DHALF_RGB10=1', '-DDISPLAY_MV=1', f'-DTILE_W={tile}', f'-DTILE_H={tile}'] + ldr)
    for out, flags in (('down_m_dot2', []), ('down_m_rgb10_dot2', ['-DOUTPUT_RGB10=1']),
                       ('down_m_r11g11b10_dot2', ['-DOUTPUT_R11G11B10=1'])):
        glsl('down_m.comp', out + '.spv', ['-DSEG=2'] + flags + ldr)
    if a.hip:
        package_hip(a, Ps)
    marker = a.output / 'ldr'
    if a.ldr: marker.write_text('1\n')
    elif marker.exists(): marker.unlink()
    for p in '01': glsl('exposure_k.comp', f'exposure_m{p}.spv', ['-DMODE=1', '-DPASS=' + p])
    if a.dxil: build_dxil(a.output, a.ldr, preset_l=True, target=a.d3d12_target)
    package_rdna3(a)
    for stale in ('enc0_m.spv', 'dec0_m.spv', 'post_m.spv', 'post_m_q.spv', 'post_m_c.spv', 'enc0_m_planes.spv', 'dec0_m_planes.spv',
                  'enc0_m_dot2.spv', 'dec0_m_dot2.spv', 'post_m_dot2.spv', 'post_m_q_dot2.spv', 'post_m_c_dot2.spv',
                  'enc0_m.dxil', 'dec0_m.dxil', 'post_m.dxil', 'post_m_q.dxil', 'post_m_c.dxil'):
        (a.output / stale).unlink(missing_ok=True)
    files = (sorted(a.output.glob('*.bin')) + sorted(a.output.glob('*.spv')) + sorted(a.output.glob('*.dxil')) + sorted(a.output.glob('rdna3/*.dxil')) + sorted(a.output.glob('rdna3/*.bin')) +
             sorted(a.output.glob('hip/*.hsaco')))
    (a.output / 'model.json').write_text(json.dumps({'format': 'D4RL0002', 'preset': 'L', 'files': {str(f.relative_to(a.output)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}}, indent=1))
    print(f'Compiled native preset L model in {a.output} ({blob8.size} FP8 bytes, {blob16.size} bytes of f16 data)')


def package_rdna3(a):
    if a.rdna3_enabled:
        m_d3d12_build.build(a.output, a.rdna3_math)
    else:
        # a rebuild without it must not leave a stale RDNA 3 package behind
        (a.output / 'd3d12_ml.bin').unlink(missing_ok=True)


def package_hip(a, Ps):
    """hipnet.bin: "D4RMHIP1", layer count, then per layer {kernel name[32], shift x, y, token grid divisor, weight
    offset, weight bytes} and the weight allocations as NGX uploads them (the native kernels read that layout)"""
    scale = {'enc1': 1, 'enc2': 2, 'enc3_tube': 4, 'dec2': 2, 'dec1': 1}
    blobs, head, at = [], struct.pack('<8sI4x', b'D4RMHIP1', len(Ps)), 16 + 64 * len(Ps)
    for P, name in zip(Ps, mm.ORDER):
        w = np.ascontiguousarray(P.wbuf); pad = (-w.size) % 64
        head += struct.pack('<32siiIIQQ', f'rrlite_{name}_4x4'.encode(), P.sx, P.sy, scale[name], 0, at, w.size)
        blobs += [w, np.zeros(pad, np.uint8)]; at += w.size + pad
    (a.output / 'hipnet.bin').write_bytes(head + np.concatenate(blobs).tobytes())
    (a.output / 'hip').mkdir(exist_ok=True)
    for name in sorted(set(mm.ORDER)):
        src = a.hip / f'rrlite_{name}_4x4.hsaco'
        if not src.exists(): raise ValueError(f'{src} is missing')
        (a.output / 'hip' / src.name).write_bytes(src.read_bytes())


if __name__ == '__main__':
    main()
