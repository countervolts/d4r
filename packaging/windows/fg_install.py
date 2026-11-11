#!/usr/bin/env python3
"""fg_install.py RUNTIME --full DLL --lite DLL --streamline DIR: NVIDIA DLSS Frame Generation for the native Windows
release (install.ps1 runs it after the engine models; the Windows counterpart of install.sh's setup_fg). On an RDNA3
(gfx11) or RDNA4 (gfx12) GPU it
  - translates both DLSS-G DLLs' CUDA kernels for this GPU's generation (d4r\\fg\\dlssg\\tools\\dlssg_setup.py) into
    %LOCALAPPDATA%\\d4r\\nvxrt-gfx11 (RDNA3) or %LOCALAPPDATA%\\d4r\\nvxrt (RDNA4),
  - puts the DLLs in d4r\\fg\\dlssg\\full (DLSS-G 310.7.0) and lite (310.9.16), Streamline's DLLs in OptiScaler's
    streamline folder (its DLSS-G output) and d4r's shim beside them as the host's next core,
  - writes d4r.ini [FrameGeneration] and routes OptiScaler's NGX path through the DLSS-G host, with the OptiScaler
    settings frame generation was set up and run with on RDNA3 (Ghost of Tsushima, 2026-11-03).
--check only reports whether this GPU can run it (install.ps1 asks before downloading NVIDIA's DLLs).
Exit codes: 0 set up (or supported), 3 not an RDNA3/RDNA4 GPU (nothing changed), 1 failure (reason printed)."""
import argparse
import configparser
import filecmp
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

HOST = 'nvngx.dll_dlssg_host.dll'
STREAMLINE = ('sl.interposer.dll', 'sl.common.dll', 'sl.dlss_g.dll', 'sl.reflex.dll', 'sl.pcl.dll', 'nvngx_dlssg.dll')
# OptiScaler.ini values DLSS-G ran with on RDNA3 (OptiScaler generates the frames from its upscaler's inputs through its
# DLSS-G output; Streamline needs Reflex, which OptiScaler's fakenvapi provides, and hardware scheduling, spoofed)
OPTISCALER = [('FrameGen', 'Enabled', 'true'), ('FrameGen', 'FGInput', 'Upscaler'), ('FrameGen', 'FGOutput', 'DLSSG'),
              ('FrameGen', 'FGNvngxReplacement', 'None'), ('Spoofing', 'Dxgi', 'true'),
              ('Spoofing', 'SpoofHAGS', 'true'), ('Spoofing', 'StreamlineSpoofing', 'false'),
              ('fakenvapi', 'UseFakenvapi', 'true'), ('fakenvapi', 'ForceReflex', '2'),
              ('fakenvapi', 'LowLatencyOutput', '4')]
D4R_INI_SECTION = """[FrameGeneration]
; NVIDIA DLSS Frame Generation (2x-6x) on RDNA3 and RDNA4 through d4r\\fg\\dlssg (install.cmd sets it up). dlssg: on.
; off: no DLSS-G (the game sees no frame generation). Takes effect at the next launch. Env: D4R_FG_BACKEND
Backend = dlssg
; true: bit-exact kernels. false: texture-unit filtering and fp32 accumulation, less GPU time.
Exact = false
; full: DLSS-G's full model (DLSS-G 310.7.0, d4r\\fg\\dlssg\\full). lite: NVIDIA's FG Lite, much faster and a little
; softer (DLSS-G 310.9.16, d4r\\fg\\dlssg\\lite). auto: Lite on RDNA3, full on RDNA4. Env: D4R_FG_MODEL
Model = auto
; Folder of the translated kernels for this GPU (install.cmd writes it). auto: %LOCALAPPDATA%\\d4r\\nvxrt-gfx11 on
; RDNA3, %LOCALAPPDATA%\\d4r\\nvxrt on RDNA4
KernelCache = auto
"""

# HIP's devices through AMD's driver runtime (as packaging/build_engine_models.py lists them): "bytes target" each
HIP_DEVICES = r'''
import ctypes
hip = ctypes.WinDLL("amdhip64_7.dll")
if hip.hipInit(0) == 0:
    count = ctypes.c_int(0); hip.hipGetDeviceCount(ctypes.byref(count))
    for i in range(count.value):
        size = ctypes.c_size_t(0); hip.hipDeviceTotalMem(ctypes.byref(size), i)
        props = ctypes.create_string_buffer(1472 + 512)
        arch = "-"
        if hip.hipGetDevicePropertiesR0600(props, i) == 0:
            arch = props.raw[1160:1160 + 256].split(b"\0")[0].split(b":")[0].decode(errors="replace") or "-"
        print(size.value, arch)
'''


def gpu_target():
    """D4R_GPU_ARCH, else the target of the AMD GPU with the most memory (a Radeon card beside integrated graphics)"""
    if os.environ.get('D4R_GPU_ARCH'):
        return os.environ['D4R_GPU_ARCH']
    try:
        out = subprocess.run([sys.executable, '-c', HIP_DEVICES], capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ''
    devices = [l.split() for l in out.splitlines() if len(l.split()) == 2 and l.split()[0].isdigit()]
    return max(devices, key=lambda d: int(d[0]))[1] if devices else ''


def update_ini(text, settings):
    """packaging/d4r-features.py's update(): keep comments, unknown settings and line endings"""
    configparser.ConfigParser(interpolation=None).read_string(text.lstrip('﻿'))   # rejects malformed INIs
    lines = text.splitlines(keepends=True)
    newline = '\r\n' if '\r\n' in text else '\n'
    for section, key, value in settings:
        starts = [i for i, line in enumerate(lines) if line.strip().casefold() == f'[{section}]'.casefold()]
        if len(starts) > 1:
            raise ValueError(f'duplicate [{section}]')
        if not starts:
            if lines and not lines[-1].endswith('\n'):
                lines[-1] += newline
            lines.extend([f'[{section}]{newline}', f'{key}={value}{newline}'])
            continue
        start = starts[0]
        end = next((i for i in range(start + 1, len(lines)) if lines[i].lstrip().startswith('[')), len(lines))
        matches = [i for i in range(start + 1, end) if re.match(rf'\s*{re.escape(key)}\s*=', lines[i], re.IGNORECASE)]
        if len(matches) > 1:
            raise ValueError(f'duplicate [{section}] {key}')
        if matches:
            ending = '\r\n' if lines[matches[0]].endswith('\r\n') else '\n' if lines[matches[0]].endswith('\n') else ''
            lines[matches[0]] = f'{key}={value}{ending}'
        else:
            if end and not lines[end - 1].endswith('\n'):
                lines[end - 1] += newline
            lines.insert(end, f'{key}={value}{newline}')
    return ''.join(lines)


def write_text(path, text):
    staged = path.with_name(path.name + '.fg-install')
    staged.write_bytes(text.encode('utf-8'))
    os.replace(staged, path)


def place(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and filecmp.cmp(source, destination, shallow=False):
        return
    staged = destination.with_name(destination.name + '.fg-install')
    shutil.copy2(source, staged)
    os.replace(staged, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('runtime', type=Path, help='the game\'s d4r folder')
    parser.add_argument('--check', action='store_true', help='only check the GPU (exit 0 supported, 3 not)')
    parser.add_argument('--full', type=Path, help='DLSS-G 310.7.0 nvngx_dlssg.dll')
    parser.add_argument('--lite', type=Path, help='DLSS-G 310.9.16 nvngx_dlssg.dll (Streamline 2.14.16)')
    parser.add_argument('--streamline', type=Path, help='folder with Streamline 2.14.16\'s DLLs')
    a = parser.parse_args()
    if not a.check and not (a.full and a.lite and a.streamline):
        parser.error('--full, --lite and --streamline are required')
    runtime = a.runtime.resolve()
    game = runtime.parent
    fg = runtime / 'fg' / 'dlssg'
    target = gpu_target()
    arch = 'gfx12' if target.startswith('gfx12') else 'gfx11' if target.startswith('gfx11') else None
    if arch is None:
        print(f'DLSS Frame Generation: not set up (it runs on RDNA3 and RDNA4 GPUs; this GPU is {target or "unknown"}).')
        return 3
    if a.check:
        print(f'DLSS Frame Generation: this GPU ({target}) can run it.')
        return 0
    for f in (fg / HOST, fg / 'nvxrt-scan.exe', fg / 'tools' / 'dlssg_setup.py', fg / 'bin' / 'spirv-val.exe',
              runtime / 'tools' / 'models' / 'bin' / 'glslangValidator.exe', runtime / 'nvngx.dll', game / 'OptiScaler.ini'):
        if not f.is_file():
            raise SystemExit(f'DLSS Frame Generation: missing {f} (extract the complete release ZIP)')
    local = Path(os.environ.get('LOCALAPPDATA', Path.home() / 'AppData' / 'Local'))
    cache = local / 'd4r' / ('nvxrt' if arch == 'gfx12' else 'nvxrt-gfx11')
    # the kernels, translated for this GPU's generation (a minute or two the first time, then a no-op)
    env = dict(os.environ)
    env['PATH'] = os.pathsep.join([str(fg / 'bin'), str(runtime / 'tools' / 'models' / 'bin'), env.get('PATH', '')])
    print(f'DLSS Frame Generation: translating its kernels for this GPU ({target}, {"RDNA4" if arch == "gfx12" else "RDNA3"}); '
          'a minute or two the first time...', flush=True)
    result = subprocess.run([sys.executable, str(fg / 'tools' / 'dlssg_setup.py'), '--arch', arch, '--cache', str(cache),
                             '--scan', str(fg / 'nvxrt-scan.exe'), str(a.full), str(a.lite)], env=env)
    if result.returncode:
        raise SystemExit('DLSS Frame Generation: not set up (the kernel translation failed, reason above)')
    # the DLLs: both models in d4r\fg\dlssg, Streamline for OptiScaler's DLSS-G output, d4r's shim as the next core
    place(a.full, fg / 'full' / 'nvngx_dlssg.dll')
    place(a.lite, fg / 'lite' / 'nvngx_dlssg.dll')
    ini = game / 'OptiScaler.ini'
    if ini.is_symlink():
        raise SystemExit('DLSS Frame Generation: OptiScaler.ini must be a regular file, not a symlink')
    text = ini.read_bytes().decode('utf-8-sig')
    parsed = configparser.ConfigParser(interpolation=None)
    parsed.read_string(text)
    opti = parsed.get('Libraries', 'OptiDllPath', fallback='auto')
    opti_dir = game / ('OptiScaler' if opti.lower() == 'auto' else opti.replace('\\', '/'))
    for name in STREAMLINE:
        place(a.lite if name == 'nvngx_dlssg.dll' else a.streamline / name, opti_dir / 'streamline' / name)
    place(runtime / 'nvngx.dll', runtime / 'd4r_dlssg_next.dll')
    # d4r.ini [FrameGeneration]: on, with the cache it was translated into
    d4r_ini = runtime / 'd4r.ini'
    d4r_text = d4r_ini.read_bytes().decode('utf-8-sig')
    if not re.search(r'(?mi)^\[FrameGeneration\]', d4r_text):
        newline = '\r\n' if '\r\n' in d4r_text else '\n'
        d4r_text = d4r_text.rstrip('\r\n') + newline * 2 + D4R_INI_SECTION.replace('\n', newline)
    write_text(d4r_ini, update_ini(d4r_text, [('FrameGeneration', 'Backend', 'dlssg'), ('FrameGeneration', 'KernelCache', str(cache))]))
    # OptiScaler: NGX through the DLSS-G host, and its DLSS-G output on
    route = 'd4r\\fg\\dlssg\\' + HOST if runtime == game / 'd4r' else str(fg / HOST)
    write_text(ini, update_ini(text, OPTISCALER + [('Libraries', 'NvngxPath', route)]))
    print(f'DLSS Frame Generation: ready ({"full model" if arch == "gfx12" else "Lite"} by default; kernels in {cache}).')
    print('  Select DLSS in the game, then turn frame generation on in OptiScaler\'s menu (Insert).')
    print('  d4r\\d4r.ini [FrameGeneration]: Model = auto / full / lite, Backend = off turns it off.')
    if arch == 'gfx12':
        print('  RDNA4 frame generation on Windows is new and untested: please report how it runs.')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError) as error:
        print(f'DLSS Frame Generation: not set up ({error})')
        sys.exit(1)
