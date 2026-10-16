"""Compare K input batching using identical synthetic frames and alternating runs.

Uses an existing private runtime; never installs into a game. CPU interop timing
includes synchronization and NGX execution, and is not a GPU network-only time.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('probe', 'shim', 'runtime', 'core', 'dlss', 'output'):
        parser.add_argument('--' + key, type=lambda s: Path(s).resolve(), required=True)
    parser.add_argument('--resolution', default='1920x1080')
    parser.add_argument('--input-resolution', help='defaults to nominal Quality (two-thirds per axis)')
    parser.add_argument('--frames', type=int, default=16)
    parser.add_argument('--profile-stages', action='store_true',
                        help='diagnostic CPU stage logging; adds overhead to timings')
    parser.add_argument('--candidate-native', type=lambda s: Path(s).resolve(),
                        help='compare an experimental native directory instead of input batching')
    parser.add_argument('--profile-kernels', action='store_true',
                        help='diagnostic deferred GPU events, allowing legacy-stream dependencies; timings are instrumented')
    args = parser.parse_args()
    if args.frames < 8 or args.frames > 120:
        parser.error('frames must be 8..120')
    if not re.fullmatch(r'[1-9][0-9]*x[1-9][0-9]*', args.resolution):
        parser.error('resolution must be WIDTHxHEIGHT')
    ow, oh = map(int, args.resolution.split('x'))
    if not args.input_resolution:
        args.input_resolution = f'{ow * 2 // 3}x{oh * 2 // 3}'
    if not re.fullmatch(r'[1-9][0-9]*x[1-9][0-9]*', args.input_resolution):
        parser.error('input-resolution must be WIDTHxHEIGHT')
    build_root = Path(__file__).resolve().parents[2] / 'build'
    if not args.output.is_relative_to(build_root.resolve()):
        parser.error('output must stay under the repository ignored build directory')
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'passed': False, 'resolution': args.resolution, 'warmupFrames': 4,
              'inputResolution': args.input_resolution,
              'stageProfiling': args.profile_stages,
              'kernelProfiling': args.profile_kernels,
              'comparison': 'native kernels' if args.candidate_native else 'input batching',
              'timing': 'CPU interop including NGX and synchronization', 'runs': []}
    runtime = args.runtime
    native = runtime / 'native'
    if (native / 'gfx1101').is_dir():
        native = native / 'gfx1101'
    nvapi = runtime / 'nvapi/nvapi64.dll'
    if not nvapi.is_file():
        nvapi = runtime / 'nvapi/gfx1101/nvapi64.dll'
    report['artifacts'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (args.probe, args.shim, args.core, args.dlss)}
    report['nativeArtifacts'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(native.glob('*')) if p.is_file()}
    if args.candidate_native:
        report['candidateNativeArtifacts'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                             for p in sorted(args.candidate_native.glob('*')) if p.is_file()}
    env = os.environ.copy()
    # Keep profiling and arithmetic settings independent of the caller's shell.
    for key in list(env):
        if key.startswith('D4R_') or key in ('ZLUDA_LOG_DIR', 'ZLUDA_CACHE_DIR', 'ZLUDA_CUDA_LIB'):
            del env[key]
    env.update(HIP_PATH=str(runtime / 'hip'),
               ZLUDA_CUDA_LIB=str(runtime / 'zluda/nvcuda.dll'),
               D4R_NVAPI_BACKEND=str(runtime / 'nvapi/backend/nvapi64.dll'),
               D4R_FORMAT_MODULE=str(runtime / 'pixel_convert_gfx1101.hsaco'),
               D4R_ZLUDA_NATIVE_DIR=str(native),
               D4R_ZLUDA_WMMA='1', D4R_ZLUDA_WMMA_FP8='1',
               D4R_ZLUDA_WMMA_F16_REFERENCE='1', D4R_QUIET_API='1',
               ZLUDA_CACHE_DIR=str(Path(os.environ['LOCALAPPDATA']) / 'd4r/cache'),
               D4R_NGX_DATA_DIR=str(args.output / 'ngx-data'))
    env['PATH'] = str(runtime / 'hip/bin') + os.pathsep + str(runtime / 'zluda') + os.pathsep + env['PATH']
    if args.profile_stages:
        env['D4R_PROFILE_STAGES'] = '1'
    if args.profile_kernels:
        env.update(D4R_ZLUDA_PROFILE_DEFERRED='1', D4R_ZLUDA_PROFILE_ALLOW_LEGACY='1',
                   D4R_ZLUDA_PROFILE_EVERY='1')
    try:
        reference = None
        for index, batched in enumerate((False, True, True, False)):
            label = ('native-candidate' if args.candidate_native else 'batch') if batched else 'control'
            output = args.output / f'{index}-{label}'
            output.mkdir()
            env['D4R_DIAG_DIR'] = str(output)
            if batched and not args.candidate_native:
                env['D4R_BATCH_INPUT_COPIES'] = '1'
            else:
                env.pop('D4R_BATCH_INPUT_COPIES', None)
            env['D4R_ZLUDA_NATIVE_DIR'] = str(args.candidate_native if batched and args.candidate_native else native)
            command = [str(args.probe), '--module', str(args.shim), '--hip-root', str(runtime / 'hip'),
                       '--cuda-dll', str(runtime / 'zluda/nvcuda.dll'), '--ngx-core', str(args.core),
                       '--dlss-dll', str(args.dlss), '--nvapi-dll', str(nvapi),
                       '--ngx-abi', 'project', '--interop-mode', 'command-list', '--preset', '11',
                       '--ngx-output-resolution', args.resolution,
                       '--ngx-input-resolution', args.input_resolution, '--iterations', str(args.frames)]
            with (output / 'stdout.log').open('wb') as out, (output / 'stderr.log').open('wb') as err:
                result = subprocess.run(command, env=env, stdout=out, stderr=err, timeout=180)
            text = (output / 'stdout.log').read_text(errors='replace')
            if result.returncode or 'PASS D3D12_DLSS' not in text:
                raise RuntimeError(f'run {index} failed ({result.returncode}); inspect {output}')
            iw, ih = map(int, args.input_resolution.split('x'))
            expected_size = f'D3D12_NGX_DIMENSIONS input={iw}x{ih} output={ow}x{oh}'
            if expected_size not in text or f'D4R_INTEROP async=0 batch_inputs={int(batched and not args.candidate_native)}' not in text:
                raise RuntimeError(f'run {index}: requested dimensions/interop mode not applied')
            if len(re.findall(r'D3D12_OUTPUT preset=11 frame=\d+ finite=1 ', text)) != args.frames:
                raise RuntimeError(f'run {index}: missing finite-output checks')
            times = [float(x) for x in re.findall(r'D4R_FRAME .*?interop_ngx_ms=([0-9.]+)', text)]
            if len(times) != args.frames:
                raise RuntimeError(f'run {index}: incomplete frame timing')
            hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob('output-*.rgba16f')}
            if len(hashes) != args.frames:
                raise RuntimeError(f'run {index}: incomplete output readback')
            if reference is None:
                reference = hashes
            elif hashes != reference:
                raise RuntimeError(f'run {index}: output differs from control')
            report['runs'].append({'batch': batched and not bool(args.candidate_native),
                                   'candidate': batched, 'allFrameMs': times,
                                   'medianMs': statistics.median(times[4:]), 'outputHashes': hashes})
            print(f'run={index} variant={label} warmed_cpu_interop_ms={statistics.median(times[4:]):.3f}', flush=True)
        for mode in (False, True):
            values = [t for row in report['runs'] if row['candidate'] == mode for t in row['allFrameMs'][4:]]
            key = ('candidateMedianMs' if args.candidate_native else 'batchMedianMs') if mode else 'controlMedianMs'
            report[key] = statistics.median(values)
        report['passed'] = True
    except Exception as error:
        report['error'] = str(error)
        print(str(error), flush=True)
    finally:
        (args.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
