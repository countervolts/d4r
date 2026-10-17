"""Probe plain Windows SR render subrects without a game or FSR denoiser.

Retains creation-sized resources. Separate child processes compare static,
smaller-render, and reset-on-every-frame cases. Output is diagnostic evidence,
not a quality or performance gate.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('probe', 'shim', 'runtime', 'core', 'dlss', 'output'):
        parser.add_argument('--' + name, required=True, type=lambda p: Path(p).resolve())
    parser.add_argument('--creation-size', default='426x256')
    parser.add_argument('--smaller-render-size', default='384x224')
    args = parser.parse_args()
    build = Path(__file__).resolve().parents[2] / 'build'
    if not args.output.is_relative_to(build.resolve()):
        parser.error('output must stay under ignored build/')
    args.output.mkdir(parents=True, exist_ok=False)
    runtime = args.runtime
    native = runtime / 'native'
    if (native / 'gfx1101').is_dir():
        native /= 'gfx1101'
    nvapi = runtime / 'nvapi/nvapi64.dll'
    if not nvapi.is_file():
        nvapi = runtime / 'nvapi/gfx1101/nvapi64.dll'
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('D4R_', 'ZLUDA_'))}
    env.update(HIP_PATH=str(runtime / 'hip'),
               ZLUDA_CUDA_LIB=str(runtime / 'zluda/nvcuda.dll'),
               D4R_NVAPI_BACKEND=str(runtime / 'nvapi/backend/nvapi64.dll'),
               D4R_FORMAT_MODULE=str(runtime / 'pixel_convert_gfx1101.hsaco'),
               D4R_ZLUDA_NATIVE_DIR=str(native), D4R_ZLUDA_WMMA='1',
               D4R_ZLUDA_WMMA_FP8='1', D4R_ZLUDA_WMMA_F16_REFERENCE='1',
               D4R_QUIET_API='1',
               ZLUDA_CACHE_DIR=str(Path(os.environ['LOCALAPPDATA']) / 'd4r/cache'))
    env['PATH'] = str(runtime / 'hip/bin') + os.pathsep + str(runtime / 'zluda') + os.pathsep + env['PATH']
    report = {'synthetic': True, 'gameAccessed': False, 'qualityReference': False,
              'performanceEvidence': False, 'creationSize': args.creation_size,
              'outputSize': [640, 384], 'smallerRenderSize': args.smaller_render_size,
              'artifacts': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in (args.probe, args.shim, args.core, args.dlss)},
              'runs': []}
    for flags in (0, 11):
        for mode in ('static', 'smaller', 'smaller-reset'):
            output = args.output / f'flags-{flags}-{mode}'
            output.mkdir()
            child_env = dict(env, D4R_DIAG_DIR=str(output), D4R_NGX_DATA_DIR=str(output / 'ngx-data'))
            if mode != 'static':
                child_env['D4R_PROBE_RENDER_SUBRECT'] = args.smaller_render_size
            command = [str(args.probe), '--module', str(args.shim),
                       '--hip-root', str(runtime / 'hip'), '--cuda-dll', str(runtime / 'zluda/nvcuda.dll'),
                       '--ngx-core', str(args.core), '--dlss-dll', str(args.dlss), '--nvapi-dll', str(nvapi),
                       '--ngx-abi', 'project', '--interop-mode', 'command-list', '--preset', '11',
                       '--ngx-create-flags', str(flags), '--ngx-input-resolution', args.creation_size,
                       '--ngx-output-resolution', '640x384', '--iterations', '4']
            if mode == 'smaller-reset':
                command += ['--reset-every-frame', '1']
            row = {'flags': flags, 'mode': mode, 'command': command}
            with (output / 'stdout.log').open('wb') as out, (output / 'stderr.log').open('wb') as err:
                try:
                    result = subprocess.run(command, env=child_env, stdout=out, stderr=err, timeout=180)
                    row['exitCode'] = result.returncode
                except subprocess.TimeoutExpired:
                    row['timeout'] = True
            text = (output / 'stdout.log').read_text(errors='replace')
            row['finiteFrames'] = len(re.findall(r'D3D12_OUTPUT preset=11 frame=\d+ finite=1 ', text))
            expected = [args.creation_size] + [args.creation_size if mode == 'static' else args.smaller_render_size] * 3
            row['appliedRenderSizes'] = re.findall(r'D3D12_RENDER_SUBRECT frame=\d+ size=(\d+x\d+) ', text)
            row['passed'] = (row.get('exitCode') == 0 and 'PASS D3D12_DLSS' in text
                             and row['finiteFrames'] == 4 and row['appliedRenderSizes'] == expected)
            row['outputHashes'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in sorted(output.glob('output-*.rgba16f'))}
            report['runs'].append(row)
            (args.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
            print(f'flags={flags} mode={mode} exit={row.get("exitCode")} finite_frames={row["finiteFrames"]}', flush=True)
            if row.get('timeout'):
                return 1
    report['passed'] = all(row['passed'] for row in report['runs'])
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
