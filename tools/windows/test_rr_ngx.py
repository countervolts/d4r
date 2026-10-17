"""Run the synthetic production NGX feature-13 harness in a bounded process."""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
import subprocess
import struct


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-arch', choices=('gfx1101', 'gfx1200', 'gfx1201'), default='gfx1101')
    parser.add_argument('--contracts', action='store_true', help='also check capability readiness and negative contracts')
    parser.add_argument('--inherited', action='store_true', help='inherit access from a preceding list in the same submission batch')
    parser.add_argument('--enhanced', action='store_true', help='exercise enhanced texture access and restoration')
    parser.add_argument('--split', action='store_true', help='reject a legacy split barrier, then complete the pair and evaluate')
    parser.add_argument('--unavailable', choices=('disabled', 'missing-model', 'fsr'), help='test unavailable RR configuration without inference')
    parser.add_argument('--without-sr', action='store_true', help='remove SR dependency configuration and require native RR execution')
    parser.add_argument('--float-guides', action='store_true', help='exercise equivalent float32 color/motion/normals and scalar roughness')
    parser.add_argument('--switch-guides', action='store_true', help='alternate canonical and float guide formats without recreating feature')
    parser.add_argument('--auto-exposure', action='store_true', help='also set the public NGX AutoExposure creation flag')
    parser.add_argument('--dynamic-extents', action='store_true', help='read per-frame bounded render dimensions from dynamic-extents.txt')
    parser.add_argument('--two-features', action='store_true', help='evaluate two live independent RR features on each game command list')
    parser.add_argument('--max-resources', action='store_true', help='retain creation-sized textures while dynamic input rectangles change')
    parser.add_argument('--output-float32', action='store_true', help='validate GPU half-to-float32 output conversion against widened standalone output')
    parser.add_argument('--switch-output', action='store_true', help='alternate half/float32 output formats without recreating feature')
    parser.add_argument('--depth-format', choices=('r32', 'r32-typeless', 'd32', 'd32s8', 'd32s8-typeless'), default='r32', help='exercise native depth-plane copy format')
    parser.add_argument('--player-config', action='store_true', help='stage a private player folder and select RR only through d4r.ini')
    parser.add_argument('--use-existing-ini', action='store_true', help='preserve the selected runtime INI when staging player validation')
    for name in ('probe', 'shim', 'runtime', 'core', 'dlss', 'model', 'inputs', 'reference', 'output'):
        parser.add_argument('--' + name, required=True, type=lambda value: Path(value).resolve())
    args = parser.parse_args()
    if args.use_existing_ini and not args.player_config:
        parser.error('--use-existing-ini requires --player-config')
    if args.player_config and (args.without_sr or args.unavailable or not args.contracts):
        parser.error('--player-config requires --contracts, configured SR dependencies and native RR')
    if args.enhanced and args.split:
        parser.error('--split currently covers legacy barriers; use a separate --enhanced run')
    if args.without_sr and (args.unavailable or not args.contracts):
        parser.error('--without-sr requires --contracts and a configured native RR backend')
    if args.switch_guides and not args.float_guides:
        parser.error('--switch-guides requires --float-guides')
    if args.dynamic_extents and (args.float_guides or args.switch_guides or args.depth_format != 'r32'):
        parser.error('--dynamic-extents currently uses canonical guides and R32 depth')
    if args.max_resources and not args.dynamic_extents:
        parser.error('--max-resources requires --dynamic-extents')
    if args.switch_output and (not args.output_float32 or args.max_resources):
        parser.error('--switch-output requires --output-float32 and cannot retain max resources')
    build = Path(__file__).resolve().parents[2] / 'build'
    if not args.output.is_relative_to(build.resolve()):
        parser.error('output must stay under ignored build/')
    args.output.mkdir(parents=True, exist_ok=False)
    env = {key: value for key, value in os.environ.items()
           if not key.startswith('D4R_') and key not in ('ZLUDA_CUDA_LIB', 'ZLUDA_LOG_DIR', 'ZLUDA_CACHE_DIR')}
    runtime = args.runtime
    native = runtime / 'native'
    if (native / args.gpu_arch).is_dir():
        native = native / args.gpu_arch
    env.update(HIP_PATH=str(runtime / 'hip'), D4R_HIP_ROOT=str(runtime / 'hip'), D4R_NVCUDA_DLL=str(runtime / 'zluda/nvcuda.dll'),
               ZLUDA_CUDA_LIB=str(runtime / 'zluda/nvcuda.dll'), D4R_DLSS_DLL=str(args.dlss),
               D4R_NGX_CORE=str(args.core), D4R_NVAPI_DLL=str(runtime / 'nvapi/nvapi64.dll'),
               D4R_NVAPI_BACKEND=str(runtime / 'nvapi/backend/nvapi64.dll'),
               D4R_FORMAT_MODULE=str(runtime / f'pixel_convert_{args.gpu_arch}.hsaco'),
               D4R_ZLUDA_NATIVE_DIR=str(native), D4R_QUIET_API='1',
               D4R_ZLUDA_WMMA='1', D4R_ZLUDA_WMMA_F16_REFERENCE='1',
               D4R_NGX_DATA_DIR=str(args.output / 'ngx-data'), D4R_RR_BACKEND='native', D4R_RR_MODEL_DIR=str(args.model))
    if args.unavailable == 'disabled':
        env.pop('D4R_RR_BACKEND')
    elif args.unavailable == 'missing-model':
        missing = args.output / 'missing-model'
        env['D4R_RR_MODEL_DIR'] = str(missing)
    elif args.unavailable == 'fsr':
        env['D4R_RR_BACKEND'] = 'fsr'
    if args.without_sr:
        for key in ('HIP_PATH', 'D4R_HIP_ROOT', 'D4R_NVCUDA_DLL', 'ZLUDA_CUDA_LIB', 'D4R_DLSS_DLL',
                    'D4R_NGX_CORE', 'D4R_NVAPI_DLL', 'D4R_NVAPI_BACKEND', 'D4R_FORMAT_MODULE', 'D4R_ZLUDA_NATIVE_DIR'):
            env.pop(key, None)
    else:
        env['PATH'] = str(runtime / 'hip/bin') + os.pathsep + str(runtime / 'zluda') + os.pathsep + env['PATH']
    command = [str(args.probe), str(args.shim), str(args.inputs), str(args.reference),
               str(args.output / 'frames'), str(args.model)]
    if args.player_config:
        staged = args.output / 'player'
        shutil.copytree(runtime, staged)
        shutil.copy2(args.shim, staged / 'd4r_nvngx.dll')
        for source, relative in ((args.core, 'ngx/_nvngx.dll'), (args.dlss, 'dlss/nvngx_dlss.dll')):
            target = staged / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        if not args.use_existing_ini:
            config = (Path(__file__).resolve().parents[2] / 'packaging/windows/d4r.ini').read_text()
            config = config.replace('Backend = off', 'Backend = native').replace('ModelDir = auto', 'ModelDir = ' + str(args.model))
            (staged / 'd4r.ini').write_text(config)
        env = {key: value for key, value in os.environ.items()
               if not key.startswith('D4R_') and key not in ('HIP_PATH', 'ZLUDA_CUDA_LIB', 'ZLUDA_LOG_DIR', 'ZLUDA_CACHE_DIR')}
        command[1] = str(staged / 'd4r_nvngx.dll')
    mode = '-'.join(name for name in ('contracts', 'inherited', 'enhanced', 'split', 'unavailable') if getattr(args, name))
    if args.without_sr:
        mode += '-without-sr'
    if args.float_guides:
        mode += '-float-guides'
    if args.switch_guides:
        mode += '-switch-guides'
    if args.auto_exposure:
        mode += '-auto-exposure'
    if args.dynamic_extents:
        mode += '-dynamic-extents'
    if args.two_features:
        mode += '-two-features'
    if args.max_resources:
        mode += '-max-resources'
    if args.output_float32:
        mode += '-output-float32'
    if args.switch_output:
        mode += '-switch-output'
    if args.depth_format != 'r32':
        mode += '-depth-' + args.depth_format
    if mode:
        command.append(mode)
    report = {'passed': False, 'gameAccessed': False, 'qualityReference': False,
              'artifacts': {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in (args.probe, args.shim, args.core, args.dlss)}}
    try:
        with (args.output / 'stdout.log').open('wb') as out, (args.output / 'stderr.log').open('wb') as err:
            result = subprocess.run(command, env=env, stdout=out, stderr=err, timeout=180)
        report['exitCode'] = result.returncode
        text = (args.output / 'stdout.log').read_text(errors='replace')
        if args.unavailable:
            if result.returncode or 'PASS RR_NGX_UNAVAILABLE' not in text:
                raise RuntimeError('Unavailable NGX RR checks failed; inspect ' + str(args.output))
            report.update(passed=True, unavailable=args.unavailable, rrAvailable=False, srAvailable=True,
                          creationRejected=True, frames=0)
            print('PASS NGX RR unavailable=' + args.unavailable + ' sr_available=1')
            return
        if result.returncode or 'PASS RR_NGX_LIFECYCLE' not in text:
            raise RuntimeError('NGX RR harness failed; inspect ' + str(args.output))
        if args.contracts and 'PASS RR_NGX_CONTRACTS' not in text:
            raise RuntimeError('NGX RR contract checks incomplete')
        if args.inherited and 'PASS RR_NGX_INHERITED' not in text:
            raise RuntimeError('NGX RR inherited access checks incomplete')
        for name in ('enhanced', 'split'):
            if getattr(args, name) and 'PASS RR_NGX_' + name.upper() not in text:
                raise RuntimeError('NGX RR ' + name + ' access checks incomplete')
        if args.without_sr and ('PASS RR_NGX_WITHOUT_SR' not in text or 'D4R_RUNTIME platform=Windows' in text):
            raise RuntimeError('NGX RR did not prove execution without initialized SR runtime')
        if args.float_guides and 'PASS RR_NGX_FLOAT_GUIDES' not in text:
            raise RuntimeError('NGX RR alternate float guide checks incomplete')
        if args.switch_guides and 'PASS RR_NGX_SWITCH_GUIDES' not in text:
            raise RuntimeError('NGX RR temporal guide-format switching checks incomplete')
        if args.auto_exposure and 'PASS RR_NGX_FLAGS' not in text:
            raise RuntimeError('NGX RR creation flag checks incomplete')
        if args.dynamic_extents:
            extents = [tuple(map(int, line.split())) for line in (args.inputs / 'dynamic-extents.txt').read_text().splitlines()]
            if 'PASS RR_NGX_DYNAMIC_EXTENTS' not in text:
                raise RuntimeError('NGX RR dynamic extent checks incomplete')
            for index, (width, height) in enumerate(extents):
                if f'RR_NGX_EXTENT index={index} render={width}x{height} exact=1' not in text:
                    raise RuntimeError('Incomplete NGX RR dynamic extent output')
            creation = tuple(map(int, (args.inputs / 'frames.txt').read_text().splitlines()[0].split()[1:3]))
            previous, expected_resizes = creation, []
            for size in extents:
                if size != previous:
                    expected_resizes.extend([f'D4R_RR_RESIZE render={size[0]}x{size[1]} history_reset=1 model_reused=1 packet_reused=1'] * (2 if args.two_features else 1))
                previous = size
            diagnostics = args.output / ('player/d4r_nvngx.log' if args.player_config else 'stderr.log')
            actual_resizes = [line for line in diagnostics.read_text().splitlines() if line.startswith('D4R_RR_RESIZE ')]
            if actual_resizes != expected_resizes:
                raise RuntimeError('NGX RR resize operations differ from requested sequence')
            if args.contracts and 'RR_NGX_EXTENT_BOUNDS' not in text:
                raise RuntimeError('NGX RR extent bounds checks missing')
            report['dynamicExtents'] = extents
            report.update(geometryChanges=len(expected_resizes), modelReusedAcrossResize=True,
                          packetReusedAcrossResize=True, historyResetOnResize=True)
        if args.max_resources:
            if 'PASS RR_NGX_MAX_RESOURCES' not in text:
                raise RuntimeError('NGX RR creation-sized dynamic input checks missing')
            report['creationSizedInputResourcesRetained'] = True
        depth_values = {'r32-typeless': 39, 'd32': 40, 'd32s8': 20, 'd32s8-typeless': 19}
        if args.depth_format != 'r32':
            marker = f'PASS RR_NGX_DEPTH format={depth_values[args.depth_format]} stencil_preserved={int(args.depth_format.startswith("d32s8"))}'
            if marker not in text:
                raise RuntimeError('NGX RR depth-format/stencil checks incomplete')
            if args.contracts and args.depth_format.startswith('d32') and 'RR_NGX_DEPTH_SUBRECT cropped_rejected=1' not in text:
                raise RuntimeError('NGX RR cropped depth-stencil rejection missing')
        expected = [line.split()[1] + '-out.bin' for line in (args.inputs / 'frames.txt').read_text().splitlines()
                    if line.startswith('frame ')]
        outputs = sorted(path.name for path in (args.output / 'frames').glob('*-out.bin'))
        if outputs != sorted(expected):
            raise RuntimeError('Incomplete NGX RR output set')
        def reference_bytes(name):
            data = (args.reference / name).read_bytes()
            if args.output_float32 and (not args.switch_output or expected.index(name) % 2):
                return b''.join(struct.pack('<f', value[0]) for value in struct.iter_unpack('<e', data))
            return data
        for name in expected:
            if (args.output / 'frames' / name).read_bytes() != reference_bytes(name):
                raise RuntimeError('NGX RR reference mismatch: ' + name)
        if args.two_features:
            if 'PASS RR_NGX_TWO_FEATURES' not in text:
                raise RuntimeError('NGX RR two-feature lifecycle checks incomplete')
            peer_outputs = sorted(path.name for path in (args.output / 'frames/peer').glob('*-out.bin'))
            if peer_outputs != sorted(expected):
                raise RuntimeError('Incomplete first-feature output set')
            for index, name in enumerate(expected):
                if (args.output / 'frames/peer' / name).read_bytes() != reference_bytes(name):
                    raise RuntimeError('NGX RR first-feature reference mismatch: ' + name)
                if f'RR_NGX_PEER_FRAME index={index} exact=1 both_features_live=1' not in text:
                    raise RuntimeError('NGX RR first-feature validation incomplete')
            report.update(liveFeatures=2, bothFeaturesStandaloneExact=True, separateVulkanDevices=True,
                          twoDelayedBoundariesPerCommandList=True, bothReleasedBeforeFinalSubmit=True)
        if args.output_float32:
            if 'PASS RR_NGX_OUTPUT_FLOAT32' not in text:
                raise RuntimeError('NGX RR float32 output checks missing')
            report.update(outputFormat='RGBA32F', gpuOutputConversion=True, widenedHalfReferenceExact=True)
        if args.switch_output:
            if 'PASS RR_NGX_SWITCH_OUTPUT' not in text:
                raise RuntimeError('NGX RR temporal output format switching checks missing')
            report.update(outputFormat='alternating RGBA16F/RGBA32F', outputFormatSwitchWithoutFeatureRecreation=True)
        optional_names = ('hitdistance', 'disocclusion', 'sss', 'particles')
        optional_frames = {name: sum((args.inputs / (frame.removesuffix('-out.bin') + '-' + name + '.bin')).is_file()
                                     for frame in expected) for name in optional_names}
        masks = [sum(1 << index for index, name in enumerate(optional_names)
                     if (args.inputs / (frame.removesuffix('-out.bin') + '-' + name + '.bin')).is_file())
                 for frame in expected]
        frame_lines = [line.split() for line in (args.inputs / 'frames.txt').read_text().splitlines() if line.startswith('frame ')]
        for index, mask in enumerate(masks):
            depth_mode = int(frame_lines[index][8])
            marker = (f'RR_NGX_FRAME index={index} snapshot_mutated=1 exact=1 optional_mask={mask}'
                      f' hardware_depth={int(bool(depth_mode & 2))} depth_inverted={int(bool(depth_mode & 1))}')
            if marker not in text:
                raise RuntimeError('Incomplete NGX optional-guide/depth setting checks')
        report.update(passed=True, frames=len(expected), standaloneExact=True, contracts=args.contracts, inherited=args.inherited,
                      enhanced=args.enhanced, split=args.split,
                      withoutSr=args.without_sr,
                      floatGuides=args.float_guides,
                      switchGuides=args.switch_guides,
                      creationFlags=int(next(line.split('flags=')[1] for line in text.splitlines()
                                             if line.startswith('RR_NGX_CREATION flags='))),
                      optionalGuideFrames=optional_frames, optionalMasks=masks,
                      depthFormat=args.depth_format, stencilPreserved=args.depth_format.startswith('d32s8'),
                      hardwareDepthFrames=sum(bool(int(line[8]) & 2) for line in frame_lines),
                      playerConfig=args.player_config,
                      preservedPlayerIni=args.use_existing_ini,
                      invertedDepthFrames=sum(bool(int(line[8]) & 1) for line in frame_lines),
                      snapshotMutation=True, releaseBeforeSubmit=True, parameterDestroyBeforeSubmit=True)
        print('PASS NGX RR frames=' + str(len(expected)) + ' standalone_exact=1')
    finally:
        (args.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
