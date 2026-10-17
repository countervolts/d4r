"""Capture and extract private K weights through the existing synthetic Windows probe.

Uses the audited ZLUDA layer-capture hook, not a game or DLL patch. Generated
weights/captures are private local assets and must never enter release archives.
This does not yet capture the reconstruction LUT or build a full K model.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('probe', 'shim', 'runtime', 'core', 'dlss', 'native', 'output'):
        parser.add_argument('--' + name, required=True, type=lambda p: Path(p).resolve())
    parser.add_argument('--output-resolution', default='512x288', help='NGX output WxH (default 512x288)')
    parser.add_argument('--input-resolution', default='256x144', help='NGX render WxH (default 256x144)')
    parser.add_argument('--include-lut', action='store_true', help='requires audited 328-byte output capture support in isolated runtime')
    args = parser.parse_args()
    if args.output.exists() or not args.output.is_relative_to((repo / 'build').resolve()):
        parser.error('use a fresh directory under ignored build/')
    if sha(args.dlss) != '3975567b8943c53acce397f2b72380092f84f162d00b0d2c7d08a1025c563983':
        parser.error('requires pinned SR 310.9.1')
    if sha(args.core) != '66767018c36b3bab46398dade3adf173daa3730fda75965689ea848c9bc4e79b':
        parser.error('requires pinned NGX core 617.14')
    args.output.mkdir(parents=True)
    capture = args.output / 'capture'
    env = {key: value for key, value in os.environ.items()
           if not key.startswith('D4R_') and key not in ('ZLUDA_LOG_DIR', 'ZLUDA_CACHE_DIR', 'ZLUDA_CUDA_LIB')}
    runtime = args.runtime
    nvapi = runtime / 'nvapi/gfx1101/nvapi64.dll'
    if not nvapi.is_file():
        nvapi = runtime / 'nvapi/nvapi64.dll'
    env.update(HIP_PATH=str(runtime / 'hip'), ZLUDA_CUDA_LIB=str(runtime / 'zluda/nvcuda.dll'),
               D4R_NVAPI_BACKEND=str(runtime / 'nvapi/backend/nvapi64.dll'),
               D4R_FORMAT_MODULE=str(runtime / 'pixel_convert_gfx1101.hsaco'),
               D4R_ZLUDA_NATIVE_DIR=str(args.native), D4R_ZLUDA_WMMA='1',
               D4R_ZLUDA_WMMA_FP8='1', D4R_ZLUDA_WMMA_F16_REFERENCE='1', D4R_QUIET_API='1',
               D4R_DIAG_DIR=str(args.output), D4R_NGX_DATA_DIR=str(args.output / 'ngx-data'),
               D4R_CUDA_REPLAY_DUMP_DIR=str(capture), D4R_CUDA_REPLAY_DUMP_FILTER='dltss_pwin_',
               D4R_CUDA_REPLAY_DUMP_LIMIT='1')
    if args.include_lut:
        env['D4R_CUDA_REPLAY_DUMP_FILTER'] = ''
    env['PATH'] = str(runtime / 'hip/bin') + os.pathsep + str(runtime / 'zluda') + os.pathsep + env['PATH']
    command = [str(args.probe), '--module', str(args.shim), '--hip-root', str(runtime / 'hip'),
               '--cuda-dll', str(runtime / 'zluda/nvcuda.dll'), '--ngx-core', str(args.core),
               '--dlss-dll', str(args.dlss), '--nvapi-dll', str(nvapi), '--ngx-abi', 'project',
               '--interop-mode', 'command-list', '--preset', '11', '--ngx-output-resolution', args.output_resolution,
               '--ngx-input-resolution', args.input_resolution, '--iterations', '1']
    report = dict(passed=False, privateLocalWeights=True, gameAccessed=False, reconstructionLutCaptured=False,
                  artifacts={str(p):sha(p) for p in (args.probe, args.shim, args.core, args.dlss, runtime / 'zluda/nvcuda.dll')})
    try:
        with (args.output / 'probe.stdout.log').open('wb') as out, (args.output / 'probe.stderr.log').open('wb') as err:
            result = subprocess.run(command, env=env, stdout=out, stderr=err, timeout=180)
        if result.returncode or 'PASS D3D12_DLSS' not in (args.output / 'probe.stdout.log').read_text(errors='replace'):
            raise RuntimeError('Synthetic capture probe failed; inspect logs')
        expected = {'enc0', 'enc1', 'enc2', 'enc3', 'enc4', 'dec5', 'dec4', 'dec3', 'dec2', 'dec1', 'dec0'}
        captures = list(capture.glob('replay-*-dltss_pwin_*_layer'))
        found = {p.name.split('-dltss_pwin_')[1].removesuffix('_layer') for p in captures}
        if found != expected or len(captures) != 11:
            raise RuntimeError('Missing/duplicate audited K layer captures')
        weights = args.output / 'weights'
        weights.mkdir()
        if args.include_lut:
            outputs = list(capture.glob('replay-*-hiluma_engine_output_depthreg_mvhi_ldr_max_v2_rel'))
            if len(outputs) != 1 or (outputs[0] / 'args.bin').stat().st_size != 328:
                raise RuntimeError('Missing or invalid audited K output capture')
            fields = [line.split() for line in (outputs[0] / 'manifest.txt').read_text().splitlines()]
            pointers = [row for row in fields if row[0] == 'pointer' and row[1] == '280']
            if len(pointers) != 1:
                raise RuntimeError('Missing/duplicate reconstruction LUT pointer')
            _, _, identifier, offset = pointers[0]
            lut = (outputs[0] / ('alloc-' + identifier + '.bin')).read_bytes()
            if len(lut) != 16 * 32768 or int(offset) % 32768 or not 0 <= int(offset) < len(lut) or not any(lut):
                raise RuntimeError('Unexpected reconstruction-table allocation; refusing guessed extraction')
            (weights / 'lut.bin').write_bytes(lut)
            report.update(reconstructionLutCaptured=True, reconstructionTables=16, capturedLutIndex=int(offset) // 32768)
        extraction_env = env.copy()
        extraction_env['PYTHONPATH'] = os.pathsep.join(str(repo / path) for path in ('.tools/python/vendor', 'engine', 'kernels/tools'))
        with (args.output / 'extract.log').open('wb') as out:
            subprocess.run([sys.executable, str(repo / 'engine/extract_k.py'), str(capture), str(weights / 'k'),
                            '--unfold-gains'], env=extraction_env, stdout=out, stderr=subprocess.STDOUT, check=True, timeout=120)
        index = json.loads((weights / 'k.json').read_text())
        if set(index) != expected or any(layer['fold_gains'] for layer in index.values()):
            raise RuntimeError('Incomplete/incorrect K weight extraction')
        report.update(passed=True, layerCaptures=11, extractedLayers=11, numerics='explicit-f16-gains',
                      products={p.name:sha(p) for p in weights.iterdir() if p.is_file()},
                      weightBytes=(weights / 'k.bin').stat().st_size,
                      standaloneModelComplete=False, qualityReference=False, published=False)
        print(json.dumps(report, indent=2))
    except Exception as error:
        report['error'] = str(error)
        raise
    finally:
        (args.output / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
