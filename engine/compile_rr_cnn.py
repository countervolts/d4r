#!/usr/bin/env python3
"""Build the engine's Ray Reconstruction model for presets A, B and C (the CNN of nvngx_dlssd.dll 310.3.0).

usage: compile_rr_cnn.py NVNGX_DLSSD_DLL A|B|C OUT_DIR

NGX 310.3 runs a convolutional network for the render presets A - C (hints 1 - 3) instead of the Swin transformer of
D and E: the exposure, an input kernel (cuda_dldn_engine_hkpn_input_kernel, B and C: ..._preset_1), five encoder
levels (Enc0 - Enc4), five decoder levels (Dec4 - Dec0), the last one (PDL) and the output kernel
(cuda_dldn_engine_hkpn_output_kernel). C runs B's model. Writes OUT_DIR:
  rr.txt                "D4RR0003", the preset (1 - 3) and one line per stage (engine/runtime_rr_cnn.inc)
  input.spv, output.spv the input and output kernels, translated from the DLL's PTX (engine/rr/ptx2rr.py); every
                        pointer an offset in the engine's arena, input.globals.bin their constant tables
  <stage>.spv           the eleven network stages, engine/rr/cnn_enc.comp and cnn_dec.comp
  weights-cnn.bin       the network's weights, packed for those shaders (engine/rr/cnn_pack.py)
  exposure_rr0.spv, exposure_rr1.spv  the exposure (engine/exposure_k.comp MODE 2, as presets D and E)

Everything is read from the user's DLL (the weights at engine/rr/cnn_dll.py's offsets; another file is refused) and
stays on the user's machine.
"""
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'rr'))
import cnn_dll  # noqa: E402
import cnn_pack  # noqa: E402

# network stage -> (shader, defines); the input channels and channels are cnn_dll.CHANNELS. TS: n-tiles per store of
# the 8-bit activations (NVIDIA's warps hold 32 output channels at the 64-channel levels); TH = 4 where the 160-channel
# input tile does not fit the LDS with 8 rows
NETWORK = {
    'Enc0': ('enc', []), 'Enc1': ('enc', ['IN8', 'TS=8']), 'Enc2': ('enc', ['IN8', 'SPLITK']),
    'Enc3': ('enc', ['IN8', 'SPLITK']), 'Enc4': ('enc', ['IN8', 'SPLITK']),
    'Dec4': ('dec', ['SPLITK', 'TH=4']), 'Dec3': ('dec', ['UPS', 'SPLITK', 'TH=4']), 'Dec2': ('dec', ['UPS', 'SPLITK']),
    'Dec1': ('dec', ['UPS', 'TS=8']), 'Dec0': ('dec', ['UPS']), 'PDL': ('dec', ['UPS', 'PDL']),
}
# the 1x1 convs NVIDIA stores in 32-output blocks ([out-block][K-chunk][32][16], cnn_pack.pack1x1)
BLOCKED = {'Enc2.w2', 'Enc3.w2', 'Enc4.w2', 'Dec4.w2', 'Dec4.w3', 'Dec3.w2', 'Dec3.w3', 'Dec2.w2', 'Dec2.w3'}
# the weight pointers' argument offsets
ENC_ARGS = {'w1': 24, 'w2': 40, 'w3': 56, 'b1': 72, 'b2': 88, 'b3': 104}
DEC_ARGS = {'w1': 64, 'w2': 80, 'w3': 96, 'w4': 112, 'b1': 128, 'b2': 144, 'b3': 160, 'b4': 176}
INPUT_KERNEL = {'A': 'cuda_dldn_engine_hkpn_input_kernel', 'B': 'cuda_dldn_engine_hkpn_input_kernel_preset_1'}
OUTPUT_KERNEL = 'cuda_dldn_engine_hkpn_output_kernel'
# the pointer arguments of the two translated kernels (all into the arena)
INPUT_BUFFERS, OUTPUT_BUFFERS = (296, 304, 312), (224, 232, 240)


def glsl(source, output, *defines):
    r = subprocess.run(['glslangValidator', '-V', '--target-env', 'vulkan1.3', *defines, str(source), '-o', str(output)],
                       capture_output=True, text=True)
    if r.returncode:
        sys.exit(f'{source}: {r.stdout}{r.stderr}')


def find_module(ptx_dir, kernel):
    """the module with the entry `kernel`, its sm_89 build where the DLL has several (the one the stages were checked
    against)"""
    found = [p for p in sorted(Path(ptx_dir).glob('*.ptx')) if f'.entry {kernel}(' in p.read_text(errors='replace')]
    if not found:
        raise FileNotFoundError(f'no PTX module with the entry {kernel} in {ptx_dir}')
    return next((p for p in found if '.target sm_89' in p.read_text(errors='replace')), found[-1])


def translate(ptx_dir, kernel, buffers, out, name):
    """ptx2rr.py: the kernel as a stage shader whose pointers are arena offsets; returns its textures and surfaces"""
    comp = out / f'{name}.comp'
    subprocess.run([sys.executable, str(HERE / 'rr' / 'ptx2rr.py'), str(find_module(ptx_dir, kernel)), kernel, str(comp),
                    '--block', '16,16,1', '--weights-arg', '999', '--unified-arena', *[f'--buffer={b}' for b in buffers]],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    glsl(comp, out / f'{name}.spv')
    info = json.loads(comp.with_suffix('.json').read_text())
    comp.unlink()
    comp.with_suffix('.json').unlink()
    return info['textures'], info['surfaces']


def pack(stage, tensors, dll):
    """the stage's packed tensors: argument offset -> bytes"""
    cin, c = cnn_dll.CHANNELS[stage]
    args = ENC_ARGS if stage.startswith('Enc') else DEC_ARGS
    out = {}
    for name, off in args.items():
        key = f'{stage}.{name}'
        if key not in tensors:
            continue
        at, size = tensors[key]
        raw = np.frombuffer(dll[at:at + size], np.float16)
        if name.startswith('b'):
            data = raw
        elif name == 'w1':
            data = cnn_pack.pack3x3(raw, cin, c)
        else:
            # the head: 16 outputs (PDL has none: its third conv, w3, has C outputs and is the head)
            head = (stage.startswith('Enc') and name == 'w3') or name == 'w4'
            data = cnn_pack.pack1x1(raw, c, 16 if head else c, 32 if key in BLOCKED else None)
        out[off] = np.ascontiguousarray(data, np.float16).tobytes()
    return out


def main():
    if len(sys.argv) != 4 or sys.argv[2] not in ('A', 'B', 'C'):
        sys.exit(__doc__)
    dll_path, preset, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    dll = dll_path.read_bytes()
    if hashlib.sha256(dll).hexdigest() != cnn_dll.DLL_SHA256:
        sys.exit(f'{dll_path}: not nvngx_dlssd.dll 310.3.0, whose CNN model the stages were written against')
    model = 'A' if preset == 'A' else 'B'
    out.mkdir(parents=True, exist_ok=True)
    lines = ['D4RR0003', f'preset {"ABC".index(preset) + 1}']
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, str(HERE.parent / 'kernels' / 'tools' / 'extract_dlss_ptx.py'), str(dll_path), tmp],
                       check=True, stdout=subprocess.DEVNULL)
        for name, kernel, buffers in (('input', INPUT_KERNEL[model], INPUT_BUFFERS), ('output', OUTPUT_KERNEL, OUTPUT_BUFFERS)):
            textures, surfaces = translate(tmp, kernel, buffers, out, name)
            consts = f'{name}.globals.bin' if (out / f'{name}.globals.bin').exists() else '-'
            lines.append(f'{name} {name}.spv {consts} {",".join(map(str, textures)) or "-"} '
                         f'{",".join(f"{o}:{f}" for o, f in surfaces) or "-"}')
    blob = bytearray()
    for stage, (kind, defines) in NETWORK.items():
        cin, c = cnn_dll.CHANNELS[stage]
        glsl(HERE / 'rr' / f'cnn_{kind}.comp', out / f'{stage.lower()}.spv', f'-DCIN={cin}', f'-DC={c}',
             *[f'-D{d}' for d in defines])
        pointers = []
        for off, data in pack(stage, cnn_dll.TENSORS[model], dll).items():
            blob += bytes(-len(blob) % 256)
            pointers.append(f'{off}:{len(blob)}')
            blob += data
        rows = next((int(d[3:]) for d in defines if d.startswith('TH=')), 8)
        lines.append(f'stage {stage} {stage.lower()}.spv {rows} {",".join(pointers)}')
    (out / 'weights-cnn.bin').write_bytes(bytes(blob))
    lines.append('weights weights-cnn.bin')
    for p in (0, 1):
        glsl(HERE / 'exposure_k.comp', out / f'exposure_rr{p}.spv', '-DMODE=2', f'-DPASS={p}')
    (out / 'rr.txt').write_text('\n'.join(lines) + '\n')
    print(f'Ray Reconstruction preset {preset} (CNN) model in {out}')


if __name__ == '__main__':
    main()
