"""Run the synthetic NGX feature-13 harness with the FSR Ray Regeneration backend selected only through d4r.ini.

Stages a private player folder from a packaged runtime, adds the experimental FSR payload (vkd3d-proton core, the
amdxc64 shim) and AMD's denoiser DLL, runs d4r_rr_ngx_probe with a synthetic noisy scene (make_fsr_rr_fixture.py),
and compares the output with the scene's noise-free answer. No game, no NVIDIA denoiser; Super Sampling runs when the
packaged runtime has its dependencies. Output stays under ignored build/.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

repo = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('probe', 'shim', 'runtime', 'core', 'dlss', 'fsr-dir', 'denoiser', 'fixture', 'output'):
        parser.add_argument('--' + name, required=True, type=lambda value: Path(value).resolve())
    parser.add_argument('--timeout', type=int, default=1500)
    parser.add_argument('--device', choices=('auto', 'native', 'vkd3d'), default='auto', help='select the native D3D12 or compatibility route for this diagnostic')
    parser.add_argument('--without-sr', action='store_true', help='omit SR/core dependencies and require the same-size FSR-only fallback')
    parser.add_argument('--cache-dir', type=lambda value: Path(value).resolve(), help='isolated cache for shader/compiler experiments')
    parser.add_argument('--vkd3d-config', help='diagnostic per-process vkd3d configuration')
    parser.add_argument('--vkd3d-debug', choices=('none', 'err', 'warn', 'fixme', 'info', 'trace'), help='diagnostic vkd3d logging level')
    parser.add_argument('--shader-dump', type=lambda value: Path(value).resolve(), help='diagnostic original shader dump into a new directory')
    parser.add_argument('--disable-builtin-shaders', action='store_true', help='diagnostic control: use the original translations in a locally optimized payload')
    parser.add_argument('--fp16-rte', action='store_true', help='experimental dxil-spirv payload: retain layout workaround but omit global FP16 RTZ')
    parser.add_argument('--float-guides', action='store_true')
    parser.add_argument('--switch-guides', action='store_true')
    parser.add_argument('--output-float32', action='store_true')
    parser.add_argument('--switch-output', action='store_true')
    parser.add_argument('--depth-format', choices=('r32', 'd32', 'r32-typeless', 'd32s8', 'd32s8-typeless'), default='r32')
    parser.add_argument('--reference-frames', type=lambda value: Path(value).resolve(), help='require exact components against canonical half-output frames, widening the reference for float32')
    parser.add_argument('--verify-output-widening', action='store_true', help='FSR-only: dump each composed half image and require exact output widening in the same frame')
    parser.add_argument('--max-resources', action='store_true', help='with --dynamic-extents: keep creation-sized textures and change only the render rectangle, as games do')
    parser.add_argument('--dynamic-extents', action='store_true', help='run the probe in its dynamic-extents mode (the fixture has dynamic-extents.txt)')
    parser.add_argument('--sr-bounds', action='store_true', help='Quality fixture: reject render width/height below the pinned SR dynamic minimum, then evaluate normally')
    parser.add_argument('--ini-set', action='append', default=[], metavar='KEY=VALUE',
                        help='override a [RayReconstruction] key in the staged d4r.ini (repeatable), e.g. FsrStabilityBias=auto')
    args = parser.parse_args()
    if args.switch_output and not args.output_float32:
        parser.error('--switch-output requires --output-float32')
    if args.switch_guides and not args.float_guides:
        parser.error('--switch-guides requires --float-guides')
    if args.verify_output_widening and not args.without_sr:
        parser.error('--verify-output-widening requires --without-sr')
    build = (repo / 'build').resolve()
    if not args.output.is_relative_to(build):
        parser.error('output must stay under ignored build/')
    args.output.mkdir(parents=True, exist_ok=False)
    staged = args.output / 'player'
    shutil.copytree(args.runtime, staged)
    shutil.copy2(args.shim, staged / 'd4r_nvngx.dll')
    for source, relative in ((args.core, 'ngx/_nvngx.dll'), (args.dlss, 'dlss/nvngx_dlss.dll')):
        target = staged / relative
        if args.without_sr:
            if target.exists():
                raise RuntimeError('--without-sr needs a runtime without installed SR/core DLLs')
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    fsr = staged / 'fsr'
    fsr.mkdir(exist_ok=True)
    for name in ('d4r_vkd3d_core.dll', 'amdxc64.dll'):
        source = args.fsr_dir / name
        if args.device == 'native':
            (fsr / name).unlink(missing_ok=True)
            continue  # prove that the direct route has no compatibility payload dependency
        if source.is_file():
            shutil.copy2(source, fsr / name)
    shutil.copy2(args.denoiser, staged / 'amd_fidelityfx_denoiser_dx12.dll')
    config = (repo / 'packaging/windows/d4r.ini').read_text()
    if 'Backend = off' not in config:
        raise RuntimeError('packaging d4r.ini no longer has [RayReconstruction] Backend = off')
    config = config.replace('Backend = off', 'Backend = fsr')
    if args.cache_dir:
        config = config.replace('CacheDir = auto', 'CacheDir = ' + str(args.cache_dir))
    for item in args.ini_set:
        key, value = item.split('=', 1)
        pattern = re.compile(r'^' + re.escape(key) + r'\s*=.*$', re.M)
        if not pattern.search(config):
            raise RuntimeError('d4r.ini has no ' + key + ' line to override')
        config = pattern.sub(lambda match: key + ' = ' + value, config, count=1)
    (staged / 'd4r.ini').write_text(config)
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(('D4R_', 'VKD3D_', 'DXIL_')) and key not in ('HIP_PATH', 'ZLUDA_CUDA_LIB', 'ZLUDA_LOG_DIR', 'ZLUDA_CACHE_DIR')}
    env['D4R_FSR_RR_LOG'] = '1'  # per-frame stage timings in stderr; selection itself comes only from d4r.ini
    if args.device != 'auto':
        env['D4R_FSR_RR_DEVICE'] = args.device
    if args.vkd3d_config:
        env['VKD3D_CONFIG'] = args.vkd3d_config
    if args.vkd3d_debug:
        env['VKD3D_DEBUG'] = args.vkd3d_debug
    if args.shader_dump:
        args.shader_dump.mkdir(parents=True, exist_ok=False)
        env['VKD3D_SHADER_DUMP_PATH'] = str(args.shader_dump)
    if args.disable_builtin_shaders:
        env['D4R_FSR_DISABLE_BUILTIN_SHADERS'] = '1'
    if args.fp16_rte:
        env['D4R_DXIL_F16_RTE'] = '1'
    if args.verify_output_widening:
        dump = args.output / 'converted'
        dump.mkdir()
        env['D4R_FSR_RR_DUMP_DIR'] = str(dump)
        env['D4R_FSR_RR_DUMP_FRAMES'] = str(len(list(args.fixture.glob('*-color.bin'))))
    if args.without_sr:
        # Player startup requires SR dependencies. Use the existing diagnostic environment
        # contract to exercise the supported FSR-only fallback without changing player policy.
        cache = (args.cache_dir or (Path(os.environ['LOCALAPPDATA']) / 'd4r/cache')) / 'vkd3d'
        cache.mkdir(parents=True, exist_ok=True)
        env.update(D4R_HIP_ROOT=str(staged / 'hip'), D4R_RR_BACKEND='fsr',
                   D4R_FSR_VKD3D_CORE=str(fsr / 'd4r_vkd3d_core.dll'),
                   D4R_FSR_AMDXC_SHIM=str(fsr / 'amdxc64.dll'),
                   D4R_FSR_RR_DLL=str(staged / 'amd_fidelityfx_denoiser_dx12.dll'),
                   D4R_FSR_CACHE_DIR=str(cache))
    command = [str(args.probe), str(staged / 'd4r_nvngx.dll'), str(args.fixture), '', str(args.output / 'frames'), '']
    modes = []
    if args.dynamic_extents:
        modes.append('dynamic-extents' + ('-max-resources' if args.max_resources else ''))
    if args.sr_bounds:
        modes.append('sr-bounds')
    if args.float_guides:
        modes.append('float-guides')
    if args.switch_guides:
        modes.append('switch-guides')
    if args.output_float32:
        modes.append('output-float32')
    if args.switch_output:
        modes.append('switch-output')
    if args.depth_format != 'r32':
        modes.append('depth-' + args.depth_format)
    if modes:
        command.append('-'.join(modes))
    report = {'passed': False, 'gameAccessed': False, 'qualityReference': False, 'synthetic': True,
              'selectionMode': 'diagnostic environment without SR' if args.without_sr else 'player INI',
              'requestedDevice': args.device,
              'artifacts': {str(path): sha(path) for path in (args.probe, args.shim, args.core, args.dlss, args.denoiser,
                                                              args.fsr_dir / 'd4r_vkd3d_core.dll', args.fsr_dir / 'amdxc64.dll') if path.is_file()}}
    try:
        with (args.output / 'stdout.log').open('wb') as out, (args.output / 'stderr.log').open('wb') as err:
            result = subprocess.run(command, env=env, stdout=out, stderr=err, timeout=args.timeout)
        report['exitCode'] = result.returncode
        text = (args.output / 'stdout.log').read_text(errors='replace')
        if result.returncode or 'PASS RR_NGX_LIFECYCLE' not in text:
            raise RuntimeError('NGX RR harness failed; inspect ' + str(args.output))
        if args.dynamic_extents:
            expected = [tuple(map(int, line.split())) for line in (args.fixture / 'dynamic-extents.txt').read_text().splitlines() if line.strip()]
            actual = [(int(w), int(h)) for w, h in re.findall(r'RR_NGX_EXTENT index=\d+ render=(\d+)x(\d+) exact=1', text)]
            if actual != expected or 'PASS RR_NGX_DYNAMIC_EXTENTS' not in text:
                raise RuntimeError('Dynamic render dimensions were not applied')
            if args.max_resources and 'PASS RR_NGX_MAX_RESOURCES' not in text:
                raise RuntimeError('Creation-sized resources were not retained')
            report['renderExtents'] = actual
        if args.sr_bounds:
            if 'RR_NGX_SR_BOUNDS width_below_min_rejected=1 height_below_min_rejected=1 command_list_preserved=1' not in text:
                raise RuntimeError('SR minimum bound rejections were not verified')
            report['srMinimumRejectedBeforeSubmission'] = True
        quality_command = [sys.executable, str(repo / 'tools/windows/check_fsr_rr_output.py'), '--fixture', str(args.fixture),
                           '--output', str(args.output / 'frames'), '--report', str(args.output / 'quality.json')]
        if args.output_float32:
            quality_command.append('--output-float32')
        if args.switch_output:
            quality_command.append('--switch-output')
        quality = subprocess.run(quality_command,
                                 capture_output=True, text=True)
        report['quality'] = json.loads((args.output / 'quality.json').read_text())['summary']
        if quality.returncode:
            raise RuntimeError('FSR RR output quality check failed; inspect ' + str(args.output / 'quality.json'))
        if args.reference_frames:
            sys.path.insert(0, str(repo / '.tools/python/vendor'))
            import numpy as np
            comparisons = []
            for index in range(len(list(args.fixture.glob('*-color.bin')))):
                float_output = args.output_float32 and (not args.switch_output or index % 2 == 1)
                actual = np.fromfile(args.output / 'frames' / f'{index}-out.bin', '<f4' if float_output else '<f2')
                reference = np.fromfile(args.reference_frames / f'{index}-out.bin', '<f2').astype(actual.dtype)
                comparisons.append(bool(np.array_equal(actual, reference)))
            report['exactReferenceFrames'] = comparisons
            if not all(comparisons):
                raise RuntimeError('FSR format output differs from the canonical half reference')
        if args.verify_output_widening:
            sys.path.insert(0, str(repo / '.tools/python/vendor'))
            import numpy as np
            exact = []
            for index in range(len(list(args.fixture.glob('*-color.bin')))):
                float_output = args.output_float32 and (not args.switch_output or index % 2 == 1)
                actual = np.fromfile(args.output / 'frames' / f'{index}-out.bin', '<f4' if float_output else '<f2')
                composed = np.fromfile(args.output / 'converted' / f'frame{index}-composed.bin', '<f2').astype(actual.dtype)
                exact.append(bool(np.array_equal(actual, composed)))
            report['exactSameFrameOutputConversion'] = exact
            if not all(exact):
                raise RuntimeError('FSR output differs from the same-frame composed half image')
        # player mode redirects the shim's stderr into its own log beside d4r_nvngx.dll
        err_text = (args.output / 'stderr.log').read_text(errors='replace')
        device_modes = re.findall(r'D4R_FSR_RR_DEVICE (native-d3d12|vkd3d) ', err_text)
        if not device_modes:
            raise RuntimeError('FSR RR did not identify its device route')
        report['selectedDevice'] = device_modes[-1]
        requested = {'native': 'native-d3d12', 'vkd3d': 'vkd3d'}.get(args.device)
        if requested and report['selectedDevice'] != requested:
            raise RuntimeError('FSR RR used a different device route than requested')
        playerLog = staged / 'd4r_nvngx.log'
        if playerLog.exists():
            err_text += playerLog.read_text(errors='replace')
            shutil.copy2(playerLog, args.output / 'd4r_nvngx.log')
        report['superSamplingUsed'] = 'FSR_RR super sampling submitted' in err_text
        report['builtinShaderOptimizations'] = (sorted(set(re.findall(
            r'using validated local optimization for shader ([0-9a-f]{16})', err_text)))
            if args.vkd3d_debug in ('info', 'trace') else None)
        report['builtinShadersDisabled'] = args.disable_builtin_shaders
        if args.without_sr and report['superSamplingUsed']:
            raise RuntimeError('SR ran in the FSR-only test')
        report['superSamplingFrames'] = err_text.count('FSR_RR super sampling submitted')
        expected_frames = len(list(args.fixture.glob('*-color.bin')))
        if (args.dynamic_extents or args.sr_bounds) and report['superSamplingFrames'] != expected_frames:
            raise RuntimeError('Dynamic SR fixture did not upscale every frame')
        report['srRenderLimits'] = re.findall(r'D4R_SR_RENDER_LIMITS min=(\d+x\d+) max=(\d+x\d+)', err_text)
        timings = re.findall(r'FSR_RR gpu frame \d+: prepare ([0-9.]+) ms, denoise ([0-9.]+) ms, compose\+copy ([0-9.]+) ms, total ([0-9.]+) ms', err_text)
        if len(timings) > 4:
            report['gpuTiming'] = {name: statistics.median(float(row[i]) for row in timings[4:])
                                   for i, name in enumerate(('prepareMedianMs', 'denoiseMedianMs', 'composeCopyMedianMs', 'totalMedianMs'))}
            report['gpuTiming']['warmupFrames'] = 4
            report['gpuTiming']['measuredFrames'] = len(timings) - 4
            report['gpuTiming']['tailDenoiseMedianMs'] = statistics.median(float(row[1]) for row in timings[-7:])
            report['gpuTiming']['tailFrames'] = min(7, len(timings))
        report.update(passed=True)
        print('PASS FSR_RR_NGX super_sampling=%d tail_rmse=%.4f input_rmse=%.4f ideal_nearest_rmse=%.4f' % (
            report['superSamplingUsed'], report['quality']['tailOutputRmse'], report['quality']['tailInputRmse'], report['quality'].get('idealNearestRmse', float('nan'))))
    finally:
        (args.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    if not report['passed']:
        sys.exit(1)


if __name__ == '__main__':
    main()
