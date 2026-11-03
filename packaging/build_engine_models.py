#!/usr/bin/env python3
"""Builds the native engine's models on native Windows, from the NVIDIA DLLs in the d4r folder.
The Windows counterpart of build-engine-models.sh (which renders its test frames under GE-Proton): the same model
compilers and checks (model_tools.py), with the test frames rendered natively, and every model gets its D3D12
shaders (the bundled dxc), which the native Windows engine runs. Shipped as d4r\\tools\\build-engine-models.py;
install.ps1 runs it.

usage: python build-engine-models.py [--models rr-d,rr-e,rr-f,k,k-ldr,...] [D4R_FOLDER]

Super Resolution: NVIDIA's DLSS keeps the weights packed in nvngx_dlss.dll; they are read where NGX unpacks them. The
bundled Vulkan test app renders one frame per model through this d4r folder's nvngx.dll, which runs NGX's CUDA backend
on ZLUDA (zluda\\), and the bridge (nvcuda.dll) saves what NGX uploads. The d4r folder's kernels run DLSS's
networks on RDNA3 and RDNA4, so a test frame takes seconds even with ZLUDA's cache empty (17 s for K on an RX 9070
XT). On RDNA2 ZLUDA would compile K's, L's and M's network layers with emulated matrix products (minutes each), so
their test frames run them as empty kernels (stub_network): only their launches are read.
Ray Reconstruction (rr-d, rr-e, rr-f): everything comes from nvngx_dlssd.dll 310.9.1 itself, as on Linux (about half
a minute each; no test frame, no ZLUDA).
Everything written is derived from NVIDIA's DLLs: it stays in this folder and must not be redistributed.
"""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

DLSS_SHA256 = 'be6e434a94ca32499515eb62ca0e6c274526055d568d0426e4c652dcdfb6ee6e'   # nvngx_dlss.dll 310.7.0
DLSSD_SHA256 = '4bc7ea5fcb2f32cf86bc2cb072e8d2860914a0be511cbdcb2fc206c1d9d83b80'  # nvngx_dlssd.dll 310.9.1
# model -> (preset, create flags, the dump the model is compiled from)
FRAMES = {'k': (11, 0x4b), 'e': (5, 0xa), 'f': (6, 0xa), 'l': (12, 0x4b), 'm': (13, 0x4b)}
RR_MODELS = ('rr-d', 'rr-e', 'rr-f')
MODELS = RR_MODELS + ('k', 'k-ldr', 'e', 'f', 'l', 'l-ldr', 'm', 'm-ldr')
# what the native D3D12 engine runs (E and F, DLSS 3's CNN, run on the Vulkan engine only)
DEFAULT_MODELS = RR_MODELS + ('k', 'k-ldr', 'm', 'm-ldr', 'l', 'l-ldr')


def fail(message):
    print(f'build-engine-models: {message}', file=sys.stderr)
    sys.exit(1)


def sha256(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def vkhip_kernels(runtime):
    """The release's native K layers for vkhip (RDNA3; RDNA2 unofficially, on AMD's driver only): kernels/D4R_GPU_ARCH,
    or the one gfx110x / gfx103x folder a single-GPU release holds. Windows has no KFD topology to read the GPU's
    target from; None when it is unknown."""
    candidates = [runtime / 'kernels' / os.environ['D4R_GPU_ARCH']] if os.environ.get('D4R_GPU_ARCH') else \
        sorted((runtime / 'kernels').glob('gfx110[0-3]')) + sorted((runtime / 'kernels').glob('gfx103[0-6]'))
    candidates = [c for c in candidates if c.name.startswith(('gfx110', 'gfx103')) and
                  (c / 'dltss_pwin_enc0_layer.hsaco').is_file()]
    return candidates[0] if len(candidates) == 1 else None


# Lists HIP's devices in a process of its own (a HIP runtime fault there cannot take the builder down): "index bytes
# target name" per line, through AMD's driver runtime. The target is hipDeviceProp_tR0600's gcnArchName (offset 1160,
# as tools/wine_nvcuda_bridge.c reads it), "-" when unreadable.
HIP_DEVICES = r'''
import ctypes
hip = ctypes.WinDLL("amdhip64_7.dll")
if hip.hipInit(0) == 0:
    count = ctypes.c_int(0); hip.hipGetDeviceCount(ctypes.byref(count))
    for i in range(count.value):
        size = ctypes.c_size_t(0); name = ctypes.create_string_buffer(256)
        hip.hipDeviceTotalMem(ctypes.byref(size), i); hip.hipDeviceGetName(name, 256, i)
        props = ctypes.create_string_buffer(1472 + 512)
        arch = "-"
        if hip.hipGetDevicePropertiesR0600(props, i) == 0:
            arch = props.raw[1160:1160 + 256].split(b"\0")[0].split(b":")[0].decode(errors="replace") or "-"
        print(i, size.value, arch, name.value.decode(errors="replace"))
'''


def hip_device_only():
    """With more than one AMD GPU (a Ryzen CPU's integrated graphics beside a Radeon card), AMD's Windows HIP runtime
    has crashed in its own worker thread on the frames' first allocations (RX 7900 XTX with an iGPU: 0xC0000005 in
    amdhip64_7.dll at cuMemAlloc). The test frames then see only the card with the most memory - the dedicated one, as
    integrated graphics have a share of system memory - through HIP_VISIBLE_DEVICES. A value the user set is kept.
    Returns the GPU target of the device the test frames use ("gfx1031"), '' when unknown."""
    try:
        listed = subprocess.run([sys.executable, '-c', HIP_DEVICES], capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return ''
    devices = []
    for line in listed.splitlines():
        index, size, arch, *name = line.split(' ', 3)
        devices.append((int(size), int(index), arch if arch != '-' else '', name[0] if name else ''))
    if not devices:
        return ''
    size, index, arch, name = max(devices)
    # with HIP_VISIBLE_DEVICES set, the listing only has the devices it names
    if len(devices) > 1 and not os.environ.get('HIP_VISIBLE_DEVICES'):
        os.environ['HIP_VISIBLE_DEVICES'] = str(index)
        print(f'{len(devices)} AMD GPUs: the test frames use HIP device {index} ({name}, {size >> 20} MiB) only')
    return arch


def stub_network(arch):
    """The K, L and M test frames only provide their network layers' launches (arguments and weight uploads): on RDNA2
    those layers run as empty kernels (model_tools.py stub-ptx), so ZLUDA does not compile them with emulated matrix
    products for minutes each. D4R_ENGINE_STUB_NETWORK=0 / 1 forces either, as in build-engine-models.sh."""
    forced = os.environ.get('D4R_ENGINE_STUB_NETWORK', 'auto')
    if forced in ('0', '1'):
        return forced == '1'
    return arch.startswith('gfx103')


class Builder:
    def __init__(self, runtime, kit, work, timeout):
        self.runtime, self.kit, self.work, self.timeout = runtime, kit, work, timeout
        self.tools = kit / 'model_tools.py'
        self.dll = runtime / 'nvngx_dlss.dll'
        self.total, self.done = 0, 0   # models to build and built, for the progress line
        self.stubs = None              # the empty network modules (stub_network), when the frames use them

    def status(self, log):
        """the progress line for the step that writes LOG: a bar of the models built and the step in words"""
        step = Path(log).with_suffix('').as_posix()
        model, _, part = step.partition('/')
        for suffix, words in (('-d3d12', 'D3D12 shaders'), ('-manifest', 'checking'), ('-extract', 'reading weights'),
                              ('-lut', 'tables')):
            if step.endswith(suffix):
                model, part = step.removesuffix(suffix), words
                break
        else:
            if part.startswith('run'):
                part = 'test frame' + (' (again)' if part != 'run' else '')
            elif step.startswith('ptx'):
                model, part = 'DLSS', 'extracting kernels'
            elif step == 'stubs':
                model, part = 'DLSS', 'emptying the network kernels'
            else:
                part = 'compiling'
        filled = 24 * self.done // max(self.total, 1)
        return f'  [{"#" * filled}{"-" * (24 - filled)}] {self.done}/{self.total}  {model}: {part}'

    def python(self, log, *args):
        """python ARGS... bounded by the timeout, its output in WORK/LOG (shown when it fails)"""
        self.run(log, sys.executable, *map(str, args))

    def run(self, log, *command, env=None, cwd=None, check=True):
        """COMMAND bounded by the timeout, its output in WORK/LOG; False when it failed and not CHECK"""
        result = subprocess.run([sys.executable, str(self.tools), 'run', '--log', str(self.work / log),
                                 '--timeout', str(self.timeout), '--status', self.status(log),
                                 '--', *map(str, command)], env=env, cwd=cwd)
        if result.returncode and check:
            fail(f'failed: {log} (see {self.work / log})')
        return not result.returncode

    def place(self, name, source, ignore=()):
        """replace RUNTIME/engine/NAME with the built model (renames: safe while a game runs)"""
        engine = self.runtime / 'engine'
        staged = self.work / 'place' / name
        shutil.rmtree(staged, ignore_errors=True)
        shutil.copytree(source, staged, ignore=shutil.ignore_patterns(*ignore))
        self.python(f'{name}-manifest.log', self.tools, 'manifest', staged)
        target, new, old = engine / name, engine / f'{name}.new', engine / f'{name}.old'
        engine.mkdir(exist_ok=True)
        for stale in (new, old):
            shutil.rmtree(stale, ignore_errors=True)
        shutil.move(staged, new)
        if target.exists():
            target.rename(old)
        new.rename(target)
        shutil.rmtree(old, ignore_errors=True)
        self.done += 1
        print(f'  engine\\{name} ({sum(1 for _ in target.iterdir())} files)')

    def frame(self, name):
        """one test frame of model NAME through this d4r folder; its uploads in WORK/NAME/replay"""
        preset, flags = FRAMES[name]
        w = self.work / name
        env = dict(os.environ)
        env.update({
            # the d4r folder's own files (the shim reads d4r.ini only for a game; these are its developer settings)
            'D4R_NVCUDA_BRIDGE': str(self.runtime / 'nvcuda.dll'),
            'D4R_NGX_CORE': str(self.runtime / 'ngx' / '_nvngx.dll'),
            'D4R_NGX_FEATURE_DIR': str(self.runtime),
            'D4R_NVAPI': str(self.runtime / 'nvapi' / 'nvapi64.dll'),
            'D4R_ZLUDA_LIBCUDA': str(self.runtime / 'zluda' / 'nvcuda.dll'),
            # NGX's CUDA backend, not the engine; CPU input polling, which the replay dump's synchronization needs
            'D4R_ENGINE': '0', 'D4R_SHIM_GPU_INPUT_WAIT': '0',
            'D4R_DLSS_PRESET': str(preset), 'D4R_HARNESS_CREATE_FLAGS': hex(flags),
            'D4R_SHIM_LOG': str(w / 'shim.log'), 'D4R_CUDA_TRACE': str(w / 'cuda.log'), 'D4R_CUDA_VERBOSE': '1',
            'D4R_CUDA_CAPTURE': '0',
            'D4R_CUDA_REPLAY_DUMP_DIR': str(w / 'replay'), 'D4R_CUDA_REPLAY_DUMP_LIMIT': '60',
        })
        # two frames: NGX's first (reset) frame comes out black in the Vulkan test app, so the second is the one
        # checked; the model tools read the first frame's launches
        env['D4R_HARNESS_ALLOW_BLACK'] = '1'
        # K, L and M with empty network kernels (stub_network): their frame's pixels are not checked
        stubbed = self.stubs is not None and name in ('k', 'l', 'm')
        if stubbed:
            env['D4R_CUDA_REPLACE_DIR'] = str(self.stubs)
        app = self.work / 'app'
        # AMD's Windows driver now and then loses the device in one of DLSS's kernels under ZLUDA (seen with L);
        # the next run gets through
        for attempt in (1, 2):
            shutil.rmtree(w / 'replay', ignore_errors=True)
            (w / 'replay').mkdir(parents=True)
            if self.run(f'{name}/run{"" if attempt == 1 else attempt}.log', app / 'vulkan_dlss_harness.exe',
                        self.runtime / 'nvngx.dll', w / 'out.raw', 2, env=env, cwd=app, check=attempt == 2):
                break
            print(f'  the test frame of {name} failed (log {w / "run.log"}); rendering it again...')
        result = subprocess.run([sys.executable, str(self.tools), 'capture', str(w), name,
                                 *(['--stubbed'] if stubbed else [])])
        if result.returncode:
            log = w / 'shim.log'
            if log.is_file():
                print('\n'.join(log.read_text(errors='replace').splitlines()[-15:]), file=sys.stderr)
            fail(f'model {name}: incomplete or invalid test frame (diagnostics in {w})')

    def rr(self, name):
        """Ray Reconstruction model NAME from nvngx_dlssd.dll alone, with its D3D12 shaders"""
        dll, engine, out = self.runtime / 'nvngx_dlssd.dll', self.kit / 'engine', self.work / name
        if name == 'rr-f':
            self.python('ptx-rr.log', self.kit / 'kernels' / 'tools' / 'extract_dlss_ptx.py', dll, self.work / 'ptx-rr')
            self.python('rr-f.log', engine / 'compile_rr2.py', '--dll', dll, out, '--ptx-dir', self.work / 'ptx-rr')
        else:
            self.python(f'{name}.log', engine / 'rr' / 'build_native_model.py', '--dll', dll, name[-1].upper(), out)
        self.python(f'{name}-d3d12.log', engine / 'compile_rr_dxil.py', out)
        # native-work holds the generated sources the shaders were built from; the runtime reads none of it
        self.place(name, out, ignore=('native-work',))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('runtime', nargs='?', type=Path, help='the d4r folder (default: the one above tools)')
    parser.add_argument('--models', default=','.join(DEFAULT_MODELS),
                        help=f'comma-separated, from {",".join(MODELS)} (default {",".join(DEFAULT_MODELS)})')
    parser.add_argument('--timeout', type=float, default=float(os.environ.get('D4R_MODEL_TIMEOUT', 3600)),
                        help='seconds allowed for each step (default 3600: the first frame compiles DLSS)')
    parser.add_argument('--keep', action='store_true', help='keep the captures and logs')
    a = parser.parse_args()
    if os.name != 'nt':
        fail('this builder renders its frames natively on Windows; on Linux run build-engine-models.sh')
    tools = Path(__file__).resolve().parent
    kit = tools / 'models'
    runtime = (a.runtime or tools.parent).resolve()
    models = [m for m in a.models.split(',') if m]
    for model in models:
        if model not in MODELS:
            fail(f'unknown model {model} (choose from {", ".join(MODELS)})')
    if not (runtime / 'nvngx.dll').is_file() or not (kit / 'engine').is_dir():
        fail(f'no d4r folder with the model tools at {runtime}')
    try:
        import numpy  # noqa: F401
    except ImportError:
        fail('numpy is missing (install.ps1 sets up a Python with numpy)')
    rr_models = [m for m in models if m in RR_MODELS]
    sr_models = [m for m in models if m not in RR_MODELS]
    if rr_models:
        dlssd = runtime / 'nvngx_dlssd.dll'
        if not dlssd.is_file() or sha256(dlssd) != DLSSD_SHA256:
            fail(f'Ray Reconstruction models need nvngx_dlssd.dll 310.9.1 in {runtime} (run install.ps1)')
    dll = runtime / 'nvngx_dlss.dll'
    if sr_models:
        if not dll.is_file() or sha256(dll) != DLSS_SHA256:
            fail(f'Super Resolution models need nvngx_dlss.dll 310.7.0 in {runtime} (run install.ps1)')
        for required in ('ngx/_nvngx.dll', 'nvcuda.dll', 'zluda/nvcuda.dll', 'nvapi/nvapi64.dll'):
            if not (runtime / required).is_file():
                fail(f'missing {runtime / required} (an incomplete d4r folder; extract the release again)')
    # the bundled glslangValidator and dxc (the native engine's D3D12 shaders; dxc.exe needs dxcompiler.dll and
    # dxil.dll beside it)
    os.environ['PATH'] = str(kit / 'bin') + os.pathsep + os.environ.get('PATH', '')
    if shutil.which('glslangValidator') is None:
        fail(f'glslangValidator is missing from {kit / "bin"}')
    for required in ('dxc.exe', 'dxcompiler.dll', 'dxil.dll'):
        if not (kit / 'bin' / required).is_file():
            fail(f'{required} is missing from {kit / "bin"} (an incomplete d4r folder; extract the release again)')
    os.environ['D4R_DXC'] = str(kit / 'bin' / 'dxc.exe')
    arch = hip_device_only()

    root = Path(os.environ.get('D4R_MODEL_LOG_DIR') or Path(os.environ.get('LOCALAPPDATA', tempfile.gettempdir()))
                / 'd4r' / 'model-builds')
    root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='build.', dir=root))
    b = Builder(runtime, kit, work, a.timeout)
    b.total = len(models)
    engine = kit / 'engine'
    try:
        if rr_models:
            print(f'Building the Ray Reconstruction models ({", ".join(rr_models)}) from nvngx_dlssd.dll...')
            for name in rr_models:
                b.rr(name)
        if sr_models:
            print(f'Building the Super Resolution models ({", ".join(sr_models)}) from what NGX uploads...')
            (work / 'app').mkdir()
            shutil.copy2(kit / 'bin' / 'vulkan_dlss_harness.exe', work / 'app')
            if any(m.removesuffix('-ldr') in ('k', 'l', 'm') for m in sr_models) and stub_network(arch):
                b.python('stubs.log', b.tools, 'stub-ptx', dll, work / 'stubs')
                b.stubs = work / 'stubs'
                print(f'  {arch}: the network layers run as empty kernels in the test frames', flush=True)
        frames = []
        for model in sr_models:
            base = model.removesuffix('-ldr')
            if base not in frames:
                frames.append(base)
        for name in frames:
            b.frame(name)
            if name in ('e', 'f'):
                b.python('ptx.log', kit / 'kernels' / 'tools' / 'extract_dlss_ptx.py', dll, work / 'ptx')
                b.python(f'{name}.log', engine / 'compile_sr_e.py', work / name / 'replay', dll, work / f'{name}-out',
                         '--preset', name.upper(), '--ptx-dir', work / 'ptx')
                b.place(name, work / f'{name}-out')
            elif name == 'k':
                b.python('k-extract.log', engine / 'extract_k.py', work / 'k' / 'replay', work / 'k' / 'model')
                b.python('k-lut.log', engine / 'prepare_k.py', work / 'k', work / 'k' / 'case', '--lut-only')
                # RDNA3 vkhip (games on d4r's vkd3d-proton) uses these same captured weights and the release's native
                # layers, as build_engine_models.sh packages them on Linux
                hip_args = []
                kernels = vkhip_kernels(runtime)
                if kernels is not None:
                    hip_args = ['--hip', kernels, '--capture', work / 'k']
                    accuracy = runtime / 'kernels' / 'accuracy' / kernels.name
                    if (accuracy / 'dltss_pwin_enc0_layer.hsaco').is_file():
                        hip_args += ['--hip-accuracy', accuracy]
                for v in ('k', 'k-ldr'):
                    if v in models:
                        b.python(f'{v}.log', engine / 'compile_k.py', work / 'k' / 'model', work / 'k' / 'case' /
                                 'lut.bin', work / f'{v}-out', '--dxil', 'yes', *hip_args,
                                 *(['--ldr'] if v == 'k-ldr' else []))
                        b.place(v, work / f'{v}-out')
            else:
                for v in (name, f'{name}-ldr'):
                    if v in models:
                        b.python(f'{v}.log', engine / 'compile_m.py', work / name, work / f'{v}-out', '--dxil', 'yes',
                                 *(['--ldr'] if v.endswith('-ldr') else []))
                        b.place(v, work / f'{v}-out')
            if not a.keep:
                shutil.rmtree(work / name, ignore_errors=True)
    except BaseException:
        print(f'build-engine-models: failed build and captures preserved at {work}', file=sys.stderr)
        raise
    if not a.keep:
        for leftover in work.iterdir():
            if leftover.is_dir() and not any(leftover.rglob('*.log')):
                shutil.rmtree(leftover, ignore_errors=True)
    print(f'Model build diagnostics: {work}')
    print(f'Engine models are in {runtime / "engine"}.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
