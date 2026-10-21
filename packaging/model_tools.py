#!/usr/bin/env python3
"""Bounded model-build commands and integrity checks (also shipped in d4r/tools)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

MANIFEST = 'files.sha256.json'
MODELS = ('rr-d', 'rr-e', 'rr-f', 'rr-d-gfx11', 'rr-e-gfx11', 'e', 'f', 'k', 'k-ldr', 'm', 'm-ldr', 'l', 'l-ldr')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def manifest(directory, write=False):
    directory = Path(directory)
    files = {str(p.relative_to(directory)): digest(p) for p in sorted(directory.rglob('*'))
             if p.is_file() and p.name != MANIFEST}
    if not files or not any(n in files for n in ('model.json', 'rr.txt', 'e.txt')):
        raise ValueError(f'{directory}: missing model metadata')
    if write:
        (directory / MANIFEST).write_text(json.dumps(files, indent=2, sort_keys=True) + '\n')
    elif files != json.loads((directory / MANIFEST).read_text()):
        raise ValueError(f'{directory}: missing, changed or unexpected model files; rebuild the models')


def capture(directory, name, stubbed=False):
    """A completed, finite frame and all files referenced by each network-layer capture. With stubbed (the network's
    kernels replaced by empty ones, stub_ptx) the frame's pixels are not checked: only its launches are used."""
    import numpy as np
    directory = Path(directory)
    output = directory / 'out.raw'
    if output.stat().st_size != 1280 * 720 * 8:
        raise ValueError('capture did not produce a complete RGBA16F frame')
    pixels = np.fromfile(output, dtype=np.float16).reshape(-1, 4)
    if not stubbed and (not np.isfinite(pixels).all() or not np.any(pixels[:, :3])):
        raise ValueError('capture produced a nonfinite or empty frame')
    if name in ('e', 'f'):
        launches = sorted((directory / 'replay').glob('replay-*'))
        first = next((i for i, p in enumerate(launches)
                      if p.name.split('-', 2)[-1].startswith('cuda_engine_input_kernel_rel_')), None)
        if first is None:
            raise ValueError('missing captured CNN input layer')
        frame = launches[first + 1:]
        end = next((i for i, p in enumerate(frame)
                    if p.name.split('-', 2)[-1].startswith('cuda_engine_input_kernel_rel_')), len(frame))
        network = [p for p in frame[:end] if p.name.split('-', 2)[-1].startswith('dltss_nchw8_')]
        if len(network) != 12 or 'aniso_gaussian_dfn' not in network[-1].name:
            raise ValueError('missing captured CNN network or reconstruction layer')
        selected = [launches[first], *network]
    elif name == 'k':
        kernels = ['dltss_pwin_' + s + '_layer' for s in
                   ('enc0', 'enc1', 'enc2', 'enc3', 'enc4', 'dec5', 'dec4', 'dec3', 'dec2', 'dec1', 'dec0')]
    else:
        kernels = ['rrlite_' + s + '_4x4' for s in ('enc1', 'enc2', 'enc3_tube', 'dec2', 'dec1')]
        kernels += ['rrlite_enc0_4x4_mvlo_hdr' + ('_folded' if name == 'm' else ''),
                    'rrlite_dec0_4x4' + ('_folded' if name == 'm' else '')]
    if name not in ('e', 'f'):
        selected = []
        for kernel in kernels:
            matches = sorted((directory / 'replay').glob('replay-*-' + kernel))
            if not matches:
                raise ValueError(f'missing captured layer {kernel}')
            if kernel == 'rrlite_enc3_tube_4x4':
                if len(matches) != 6:
                    raise ValueError('capture requires all six Swin tube layers')
                selected.extend(matches)
            else:
                selected.append(matches[0])
    for launch in selected:
        kernel = launch.name.split('-', 2)[-1]
        rows = [line.split() for line in (launch / 'manifest.txt').read_text().splitlines()]
        if ['kernel', kernel] not in rows:
            raise ValueError(f'{kernel}: incorrect capture metadata')
        args = next(int(r[1]) for r in rows if r[0] == 'args')
        if args <= 0 or (launch / 'args.bin').stat().st_size != args:
            raise ValueError(f'{kernel}: truncated arguments')
        allocations = {r[1]: int(r[3]) for r in rows if r[0] == 'alloc'}
        if not allocations:
            raise ValueError(f'{kernel}: no weight allocations captured')
        for index, size in allocations.items():
            if size <= 0 or (launch / f'alloc-{index}.bin').stat().st_size != size:
                raise ValueError(f'{kernel}: truncated allocation {index}')
        for r in rows:
            if r[0] == 'pointer' and (r[2] not in allocations or not 0 <= int(r[3]) < allocations[r[2]]):
                raise ValueError(f'{kernel}: invalid captured pointer')


# The network layers of presets L and M, whose launches compile_m.py reads (their arguments and weight uploads, never
# their results). On RDNA2 ZLUDA compiles each of them for minutes (matrix products emulated), so the model build
# replaces them with empty kernels of the same parameters.
NETWORK_KERNELS = re.compile(r'rrlite_(enc0_4x4_\w+|enc1_4x4|enc2_4x4|enc3_tube_4x4|dec2_4x4|dec1_4x4|dec0_4x4\w*)$')


def stub_ptx(dll, out):
    """For every CUDA fatbin of dll whose PTX has a network kernel: out/<FNV-1a 64 of the fatbin>.ptx, the PTX with
    every kernel's body empty (the wine_nvcuda_bridge D4R_CUDA_REPLACE_DIR hook loads it in place of the fatbin)"""
    tools = Path(__file__).resolve().parent
    for d in (tools / 'kernels' / 'tools', tools.parent / 'kernels' / 'tools'):
        if (d / 'extract_dlss_ptx.py').is_file():
            sys.path.insert(0, str(d))
            break
    import extract_dlss_ptx as x
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    stubs = 0
    for _, fatbin in x.fatbins(Path(dll).read_bytes()):
        entries = list(x.ptx_entries(fatbin))
        if len(entries) != 1:
            continue
        text = entries[0].rstrip(b'\0').decode('latin-1')
        kernels = re.findall(r'\.visible\s+\.entry\s+(\w+)\s*\(', text)
        if not kernels or not all(NETWORK_KERNELS.match(k) for k in kernels):
            continue
        if re.search(r'^\s*\.(global|const)\s', text, re.M):
            raise ValueError(f'{kernels[0]}: module globals; cannot stub it')
        head = re.match(r'(.*?)(?=\.visible\s+\.entry)', text, re.S).group(1)
        head = ''.join(l + '\n' for l in head.splitlines() if l.startswith(('.version', '.target', '.address_size')))
        # each kernel's declaration and performance directives up to its body, then an empty body
        bodies = [m.group(1) + '{\n\tret;\n}\n' for m in re.finditer(r'(\.visible\s+\.entry\s+\w+\s*\([^)]*\)[^{]*)\{', text)]
        if len(bodies) != len(kernels):
            raise ValueError(f'{kernels[0]}: unexpected PTX layout')
        h = 0xcbf29ce484222325
        for b in fatbin:
            h = ((h ^ b) * 0x100000001b3) & 0xffffffffffffffff
        (out / f'{h:016x}.ptx').write_text(head + '\n'.join(bodies), encoding='latin-1')
        stubs += 1
    if not stubs:
        raise ValueError(f'{dll}: no network kernels found')
    print(f'{stubs} network modules stubbed in {out}')


def run(log, timeout, command):
    if command and command[0] == '--':
        command = command[1:]
    if not command or not 0 < timeout <= 86400:
        raise ValueError('command required; timeout must be between 0 and 86400 seconds')
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    print(f'  starting {log.stem} (deadline {timeout:g}s; log {log})', flush=True)
    with log.open('wb') as output:
        child = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        def stop():
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(child.pid, sig)
                except ProcessLookupError:
                    break
                try:
                    child.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    continue
                # Kill remaining members too, even when the group leader has exited.
                if sig == signal.SIGTERM:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                break
        previous = {}
        def interrupted(sig, frame):
            raise KeyboardInterrupt
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous[sig] = signal.signal(sig, interrupted)
        result = None
        try:
            heartbeat = 0
            while child.poll() is None:
                elapsed = time.monotonic() - started
                if elapsed >= timeout:
                    stop()
                    print(f'  {log.stem}: timed out after {elapsed:.0f}s', file=sys.stderr)
                    result = 124
                    break
                if elapsed >= heartbeat + 15:
                    print(f'  {log.stem}: still running ({elapsed:.0f}s, pid {child.pid}); log {log}', flush=True)
                    heartbeat = elapsed
                time.sleep(min(.25, max(.01, timeout - elapsed)))
            if result is None:
                result = child.returncode
        except KeyboardInterrupt:
            stop()
            result = 130
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    if result:
        print('\n'.join(log.read_text(errors='replace').splitlines()[-30:]), file=sys.stderr)
    print(f'  {log.stem}: exit={result}, elapsed={time.monotonic() - started:.1f}s; log {log}', flush=True)
    return result if result >= 0 else 128 - result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    for action in ('manifest', 'validate'):
        action_parser = sub.add_parser(action)
        action_parser.add_argument('directory', type=Path)
        if action == 'validate':
            action_parser.add_argument('models', nargs='*', choices=MODELS)
    p = sub.add_parser('capture')
    p.add_argument('directory', type=Path)
    p.add_argument('name', choices=('e', 'f', 'k', 'm', 'l'))
    p.add_argument('--stubbed', action='store_true')
    p = sub.add_parser('stub-ptx')
    p.add_argument('dll', type=Path)
    p.add_argument('out', type=Path)
    p = sub.add_parser('run')
    p.add_argument('--log', required=True)
    p.add_argument('--timeout', type=float, default=900)
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = parser.parse_args()
    try:
        if a.action == 'run':
            return run(a.log, a.timeout, a.command)
        if a.action == 'capture':
            capture(a.directory, a.name, a.stubbed)
        elif a.action == 'stub-ptx':
            stub_ptx(a.dll, a.out)
        elif a.action == 'manifest':
            manifest(a.directory, write=True)
        else:
            for name in (a.models or MODELS):
                manifest(a.directory / name)
    except (OSError, ValueError, KeyError, StopIteration) as error:
        print(f'model-tools: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
