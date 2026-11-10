#!/usr/bin/env python3
"""Opt-in frame generation and latency setup. Run with the game closed.

--fg xefg / fsrfg configure OptiScaler's FG outputs (no prefix changes). --fg dlssg runs NVIDIA's own DLSS Frame
Generation (the game's nvngx_dlssg.dll, 2x/3x/4x) on RDNA3 and RDNA4 through d4r/fg/dlssg: it translates that DLL's
CUDA kernels for this GPU (~/.cache/nvxrt on RDNA4, ~/.cache/nvxrt-gfx11 on RDNA3), routes OptiScaler's NGX path through the DLSS-G host and sets d4r.ini
[FrameGeneration] Backend = dlssg (d4r_boot.dll then enables its Vulkan layer at launch; no launch options). It also
sets HwSchMode=2 in the game's Proton prefix when it finds it (some games offer DLSS-G only with hardware GPU
scheduling). --fg off undoes the routing."""
import argparse
import configparser
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time

DLSSG_HOST = 'nvngx.dll_dlssg_host.dll'


def update(source, settings):
    """Keep comments, unknown settings and line endings; reject ambiguous INIs."""
    parsed = configparser.ConfigParser(interpolation=None)
    parsed.read_string(source.lstrip('\ufeff'))
    lines = source.splitlines(keepends=True)
    newline = '\r\n' if '\r\n' in source else '\n'
    for section, key, value in settings:
        starts = [i for i, line in enumerate(lines)
                  if line.strip().casefold() == f'[{section}]'.casefold()]
        if len(starts) > 1:
            raise ValueError(f'duplicate [{section}]')
        if not starts:
            if lines and not lines[-1].endswith('\n'):
                lines[-1] += newline
            lines.extend([f'[{section}]{newline}', f'{key}={value}{newline}'])
            continue
        start = starts[0]
        end = next((i for i in range(start + 1, len(lines)) if lines[i].lstrip().startswith('[')), len(lines))
        matches = [i for i in range(start + 1, end)
                   if re.match(rf'\s*{re.escape(key)}\s*=', lines[i], re.IGNORECASE)]
        if len(matches) > 1:
            raise ValueError(f'duplicate [{section}] {key}')
        if matches:
            i = matches[0]
            ending = '\r\n' if lines[i].endswith('\r\n') else '\n' if lines[i].endswith('\n') else ''
            lines[i] = f'{key}={value}{ending}'
        else:
            if end and not lines[end - 1].endswith('\n'):
                lines[end - 1] += newline
            lines.insert(end, f'{key}={value}{newline}')
    return ''.join(lines)


def winpath(path):
    return 'Z:' + str(path.resolve()).replace('/', '\\')


def v10_settings(parsed, ini, fg, fg_input, latency):
    """v10 hooks Proton's NVAPI in place; its integrated provider owns XeLL."""
    def local_path(value):
        value = value.replace('\\', '/')
        if value[:3].lower() == 'z:/':
            return Path(value[2:])
        if re.match(r'^[a-zA-Z]:', value):
            raise ValueError('use a relative or Z: library path for the Linux setup helper')
        path = Path(value)
        return path if path.is_absolute() else ini.parent / path

    main = parsed.get('Libraries', 'OptiDllPath', fallback='auto')
    directory = local_path('OptiScaler' if main.lower() == 'auto' else main)
    required = []
    if fg == 'xefg':
        required += [('XeFGPath', 'libxess_fg.dll'), ('XeLLPath', 'libxell.dll')]
    if fg == 'fsrfg':
        required += [('FfxDx12FGPath', 'amd_fidelityfx_framegeneration_dx12.dll'),
                     ('FfxDx12Path', 'amd_fidelityfx_loader_dx12.dll')]
    for key, name in required:
        override = parsed.get('Libraries', key, fallback='auto')
        path = directory / name if override.lower() == 'auto' else local_path(override)
        if path.is_dir():
            path /= name
        if not path.is_file():
            raise ValueError(f'missing {path}; extract the complete OptiScaler/ folder')
    # Do not set NvapiPath to a proxy: v10 loads system32 NVAPI and hooks only
    # the interfaces it translates. Synthetic topology must stay disabled.
    return [('FrameGen', 'Enabled', 'true' if fg != 'off' else 'false'),
            ('FrameGen', 'FGInput', fg_input if fg != 'off' else 'nofg'),
            ('FrameGen', 'FGOutput', fg if fg != 'off' else 'nofg'),
            ('fakenvapi', 'UseFakenvapi', 'false'),
            ('fakenvapi', 'LowLatencyOutput', 'auto' if latency == 'auto' else '6'),
            ('fakenvapi', 'ForceXeLL', 'false')]


def ngx_route(ini, runtime, dlssg):
    """NvngxPath for d4r's shim, or for the DLSS-G host in front of it, relative to the INI's folder"""
    target = runtime / 'fg/dlssg' / DLSSG_HOST if dlssg else runtime / 'nvngx.dll'
    return os.path.relpath(target, ini.parent).replace('/', '\\')


def configure(ini, runtime, fg, fg_input, latency, optifg=False):
    if ini.is_symlink():
        raise ValueError('OptiScaler.ini must be a regular file, not a symlink')
    source = ini.read_bytes().decode('utf-8-sig')
    # Keep the NGX path and all SR/RR choices exactly as they are.
    parsed = configparser.ConfigParser(interpolation=None)
    parsed.read_string(source)
    route = parsed.get('Libraries', 'NvngxPath', fallback='').replace('\\', '/').lower()
    if not route.endswith(('/nvngx.dll', '/d4r_nvngx.dll', '/d4r.asi', '/' + DLSSG_HOST)):
        raise ValueError('configure d4r NvngxPath before enabling features')
    via_host = route.endswith('/' + DLSSG_HOST)
    if fg == 'dlssg':
        # DLSS-G reaches NGX through Streamline; OptiScaler's own FG stays off. Streamline's DLSS-G needs Reflex,
        # which dxvk-nvapi cannot provide on AMD (NvAPI_D3D_SetSleepMode etc. are not implemented; Ghost of Tsushima
        # crashes when DLSS-G is switched on with OptiScaler 0.9.4): OptiScaler v10's integrated latency provider is
        # set up as with --latency. 0.9's d4r NVAPI router with fakenvapi is not: with it in the path Streamline
        # reports "Disabling DLSS-G since it is not supported on current hardware". Dxgi: the game sees an NVIDIA
        # adapter; SpoofHAGS: Streamline offers DLSS-G only with hardware-accelerated GPU scheduling.
        if not parsed.has_option('fakenvapi', 'UseFakenvapi'):
            raise ValueError('DLSS-G needs OptiScaler v10 (its Reflex provider); this OptiScaler.ini is from an older '
                             'version. Extract the current d4r zip into the game folder first.')
        settings = [x for x in v10_settings(parsed, ini, 'off', None, latency) if x[0] != 'FrameGen']
        # optifg: the game has no DLSS-G of its own; OptiScaler generates the frames from its upscaler's inputs through
        # its DLSS-G output (Streamline in OptiScaler/streamline)
        fg_mode = [('FrameGen', 'Enabled', 'true'), ('FrameGen', 'FGInput', 'upscaler'), ('FrameGen', 'FGOutput', 'dlssg')] \
            if optifg else [('FrameGen', 'Enabled', 'false'), ('FrameGen', 'FGInput', 'nofg'), ('FrameGen', 'FGOutput', 'nofg')]
        settings += fg_mode + [('Libraries', 'NvngxPath', ngx_route(ini, runtime, True)),
                               ('Spoofing', 'Dxgi', 'true'), ('Spoofing', 'SpoofHAGS', 'true')]
        return write_ini(ini, source, settings)
    if fg != 'off' and fg_input is None:
        raise ValueError('--input is required with FG: dlssg, fsrfg, fsrfg30 or upscaler')
    if fg == 'xefg' and latency == 'proton':
        raise ValueError('XeFG requires --latency auto for XeLL marker delivery')
    v10 = parsed.has_option('fakenvapi', 'UseFakenvapi')
    if v10:
        settings = v10_settings(parsed, ini, fg, fg_input, latency)
    else:
        settings = legacy_settings(runtime, fg, fg_input, latency)
    if via_host:  # leaving DLSS-G: OptiScaler loads d4r's shim directly again
        settings.append(('Libraries', 'NvngxPath', ngx_route(ini, runtime, False)))
    return write_ini(ini, source, settings)


def write_ini(ini, source, settings):
    """applies SETTINGS with an exact backup; returns the backup's path, or None when nothing changed"""
    result = update(source, settings).encode('utf-8')
    original = ini.read_bytes()
    if original.startswith(b'\xef\xbb\xbf'):
        result = b'\xef\xbb\xbf' + result
    if result == original:
        return None
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
    backup = ini.with_name(ini.name + '.d4r-features-backup-' + stamp)
    with backup.open('xb') as handle:
        handle.write(original)
    # Validate everything before backup/write, then replace atomically on the same filesystem.
    descriptor, temporary = tempfile.mkstemp(prefix='.d4r-features-', dir=ini.parent)
    try:
        with os.fdopen(descriptor, 'wb') as handle:
            handle.write(result)
        os.chmod(temporary, ini.stat().st_mode & 0o777)
        os.replace(temporary, ini)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return backup


def legacy_settings(runtime, fg, fg_input, latency):
    """Keep existing 0.9 ASI users' external Fakenvapi/router setup working."""
    required = []
    if latency == 'auto':
        required += ['d4r_nvapi64.dll', 'fakenvapi.dll', 'libxell.dll']
    if fg == 'xefg':
        required += ['libxess_fg.dll']
    if fg == 'fsrfg':
        required += ['amd_fidelityfx_framegeneration_dx12.dll']
    for name in required:
        if not (runtime / name).is_file():
            raise ValueError(f'missing {runtime / name}; use the full d4r package or matching OptiScaler dependencies')
    settings = [('FrameGen', 'Enabled', 'true' if fg != 'off' else 'false'),
                ('FrameGen', 'FGInput', fg_input if fg != 'off' else 'nofg'),
                ('FrameGen', 'FGOutput', fg if fg != 'off' else 'nofg'),
                ('NvApi', 'OverrideNvapiDll', 'true' if latency == 'auto' else 'false')]
    if latency == 'auto' or fg != 'off':
        # v0.9.4 misreads the per-Intel overrides into XeSSPath. Its general
        # library directory is reliable, also for private ASI plugin folders.
        # The explicit NGX/SR paths above remain authoritative.
        settings += [('Libraries', 'OptiDllPath', winpath(runtime))]
    if latency == 'auto':
        settings += [('Libraries', 'NvapiPath', winpath(runtime / 'd4r_nvapi64.dll')),
                     # v0.9.4 misreads this as XeSSPath: a directory is safe for
                     # both the fixed loader and its older fallback probes.
                     ('Libraries', 'XeLLPath', winpath(runtime)),
                     ('Hooks', 'UseNtdllHooks', 'false')]
    if fg == 'xefg':
        settings += [('Libraries', 'XeFGPath', winpath(runtime))]
    if fg == 'fsrfg':
        settings += [('Libraries', 'FfxDx12FGPath', winpath(runtime / 'amd_fidelityfx_framegeneration_dx12.dll'))]
    return settings


# ---------------------------------------------------------------------------------------------- DLSS-G (fg/dlssg)

def gpu_target():
    """the gfx target of the largest KFD GPU node (as d4r-check.sh), or None"""
    best, target = -1, None
    for props in Path('/sys/class/kfd/kfd/topology/nodes').glob('*/properties'):
        values = dict(line.split(' ', 1) for line in props.read_text().splitlines() if ' ' in line)
        simds, version = int(values.get('simd_count', 0)), int(values.get('gfx_target_version', 0))
        if simds > best and version:
            best = simds
            target = 'gfx%d%d%x' % (version // 10000, version // 100 % 100, version % 100)
    return target


def find_dlssg(game):
    """the game's nvngx_dlssg.dll: in the executable's folder tree, else up to the game's install root
    (Unreal keeps it under Engine/Plugins/.../Streamline)"""
    roots = [game]
    if game.parent.name != 'common':  # below the install root (Binaries/Win64): search up to that root
        for parent in game.parents:
            roots.append(parent)
            if parent.parent.name == 'common' or len(roots) > 5:
                break
    for root in roots:
        hits = sorted((p for p in root.rglob('nvngx_dlssg.dll')
                       if 'd4r' not in p.relative_to(root).parts and 'backup' not in str(p).lower()),
                      key=lambda p: len(p.parts))
        if hits:
            return hits[0]
    return None


STREAMLINE_DLLS = ('sl.interposer.dll', 'sl.common.dll', 'sl.dlss_g.dll', 'sl.reflex.dll', 'sl.pcl.dll')


def game_streamline(game):
    """the game ships Streamline's DLSS-G plugin (sl.dlss_g.dll): it drives DLSS-G itself"""
    roots = [game] + ([p for p in game.parents][:5] if game.parent.name != 'common' else [])
    for root in roots:
        if root.name == 'common':
            break
        if any('d4r' not in x.relative_to(root).parts and 'optiscaler' not in str(x).lower()
               for x in root.rglob('sl.dlss_g.dll')):
            return True
        if root.parent.name == 'common':
            break
    return False


def stage_streamline(ini, src):
    """OptiScaler's own DLSS-G output (games without DLSS-G): Streamline's DLLs and nvngx_dlssg.dll in its
    OptiDllPath/streamline folder, where OptiScaler looks for them"""
    parsed = configparser.ConfigParser(interpolation=None)
    parsed.read_string(ini.read_text(encoding='utf-8-sig'))
    main = parsed.get('Libraries', 'OptiDllPath', fallback='auto').strip()
    folder = ini.parent / ('OptiScaler' if main.lower() == 'auto' else main.replace('\\', '/')) / 'streamline'
    folder.mkdir(parents=True, exist_ok=True)
    for name in STREAMLINE_DLLS + ('nvngx_dlssg.dll',):
        if not (src / name).is_file():
            raise ValueError(f'missing {src / name}')
        shutil.copy2(src / name, folder / name)
    return folder


def ini_value(path, section, key, fallback=None):
    parsed = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=(';',))
    parsed.read_string(path.read_text(encoding='utf-8-sig'))
    return parsed.get(section, key, fallback=fallback)


def proton_prefix(game):
    """the Proton prefix of the Steam game installed above GAME, or None"""
    for parent in (game, *game.parents):
        if parent.parent.name == 'common' and parent.parent.parent.name == 'steamapps':
            steamapps = parent.parent.parent
            for manifest in steamapps.glob('appmanifest_*.acf'):
                text = manifest.read_text(errors='replace')
                if re.search(r'"installdir"\s+"%s"' % re.escape(parent.name), text):
                    appid = re.search(r'"appid"\s+"(\d+)"', text).group(1)
                    prefix = steamapps / 'compatdata' / appid / 'pfx'
                    return prefix if (prefix / 'system.reg').is_file() else None
    return None


def set_hags(prefix):
    """HwSchMode=2 under GraphicsDrivers in the prefix's system.reg (Wine reports no GPU scheduling otherwise)"""
    reg = prefix / 'system.reg'
    text = reg.read_text(encoding='utf-8', errors='surrogateescape')
    line = '"HwSchMode"=dword:00000002'
    section = '[System\\\\ControlSet001\\\\Control\\\\GraphicsDrivers]'
    if re.search(r'(?m)^' + re.escape(section) + r'.*\n(?:#time=.*\n)?(?:(?!\[).*\n)*?' + re.escape(line), text):
        return False
    backup = reg.with_name('system.reg.d4r-features-backup-' + datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S'))
    shutil.copy2(reg, backup)
    i = text.find('\n' + section)
    if i < 0:
        text = text.rstrip('\n') + '\n\n%s %d\n%s\n' % (section, int(time.time()), line)
    else:
        j = text.find('\n', i + 1)
        if text.startswith('#time=', j + 1):
            j = text.find('\n', j + 1)
        k = text.find('\n[', j)
        body = re.sub(r'(?m)^"HwSchMode"=.*\n', '', text[j + 1:k if k >= 0 else len(text)])
        text = text[:j + 1] + line + '\n' + body + (text[k:] if k >= 0 else '')
    reg.write_text(text, encoding='utf-8', errors='surrogateescape')
    return backup


def setup_dlssg(game, runtime, dll, exact, lite, force):
    """everything d4r/fg/dlssg needs before the game starts; the OptiScaler route is set by configure()"""
    fg = runtime / 'fg/dlssg'
    for name in (DLSSG_HOST, 'layer/libVkLayer_nvxrt.so', 'layer/nvxrt_layer.json', 'tools/dlssg-setup'):
        if not (fg / name).is_file():
            raise ValueError(f'missing {fg / name}; this d4r package has no DLSS frame generation')
    if not (runtime / 'nvngx.dll').is_file() or not (runtime / 'nvcuda.dll').is_file():
        raise ValueError(f'{runtime} is not a d4r folder of the drag-in package (DLSS-G needs nvngx.dll and nvcuda.dll)')
    target = gpu_target() or ''
    arch = 'gfx12' if target.startswith('gfx12') else 'gfx11' if target.startswith('gfx11') else None
    if arch is None:
        if not force:
            raise ValueError(f'DLSS frame generation runs on RDNA3 (gfx11) and RDNA4 (gfx12) GPUs; this GPU is '
                             f'{target or "unknown"} (--force translates for RDNA4; the layer stays off elsewhere)')
        arch = 'gfx12'
    cache = Path.home() / '.cache' / ('nvxrt' if arch == 'gfx12' else 'nvxrt-gfx11')
    dll = Path(dll).resolve() if dll else find_dlssg(game)
    if not dll or not dll.is_file():
        raise ValueError('no nvngx_dlssg.dll in the game: it does not ship DLSS frame generation (or pass --dll)')
    # kernels first: a DLL the translator cannot handle (DLSS-G 3.x) leaves everything untouched
    env = dict(os.environ, NVXRT_ARCH=arch, NVXRT_CACHE=str(cache))
    kit = runtime / 'tools/models/bin'
    if kit.is_dir():
        env['PATH'] = f'{kit}:{env.get("PATH", "")}'
    if subprocess.run(['bash', str(fg / 'tools/dlssg-setup'), str(dll)], env=env).returncode:
        raise ValueError('kernel translation failed; nothing changed (see the message above)')
    # a DLL from outside the game (install.sh's NVIDIA download) is copied next to the host: the install stays whole
    if game not in dll.parents:
        local = fg / 'nvngx_dlssg.dll'
        if not local.is_file() or local.read_bytes() != dll.read_bytes():
            shutil.copy2(dll, local)
        dll = local
    relative = Path(os.path.relpath(dll, game))
    fg_path = (str(relative).replace('/', '\\') if relative.parts.count('..') <= 4
               else 'Z:' + str(dll).replace('/', '\\'))
    (fg / 'dlssg_host.ini').write_text(
        '; written by d4r-features.py --fg dlssg. Relative paths are taken from the game executable\'s folder.\n'
        '; ArchID 0x1b0 (Blackwell) enables multi-frame generation; Lite=1 is NVIDIA\'s FG Lite (DLSS-G >= 310.9.16).\n'
        '[Host]\nNext=d4r\\d4r_dlssg_next.dll\nFG=%s\nArchID=0x1b0\nLite=%d\nLogLevel=1\nTiming=1\nLog=1\n'
        % (fg_path, 1 if lite else 0))
    shutil.copy2(runtime / 'nvngx.dll', runtime / 'd4r_dlssg_next.dll')  # d4r_boot.dll keeps it current
    return dll, arch, cache


def set_d4r_ini(runtime, backend, exact=False, cache=None, lite=False):
    ini = runtime / 'd4r.ini'
    if not ini.is_file():
        return
    settings = [('FrameGeneration', 'Backend', backend)]
    source = ini.read_text(encoding='utf-8-sig')
    if backend == 'dlssg':
        # Exact and Model are the user's choices in d4r.ini: set only when asked (--exact, --lite) or still missing
        parsed = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=(';',))
        parsed.read_string(source)
        def missing(key):
            return not parsed.has_option('FrameGeneration', key) or not parsed.get('FrameGeneration', key).strip()
        if exact or missing('Exact'):
            settings.append(('FrameGeneration', 'Exact', 'true' if exact else 'false'))
        settings.append(('FrameGeneration', 'KernelCache', str(cache)))
        if lite or missing('Model'):
            settings.append(('FrameGeneration', 'Model', 'lite' if lite else 'auto'))
    result = update(source, settings)
    if result != source:
        ini.write_text(result, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('game', type=Path, help='folder containing OptiScaler.ini')
    parser.add_argument('--runtime', type=Path, help='d4r folder (required for a custom ASI plugin folder)')
    parser.add_argument('--fg', choices=('off', 'xefg', 'fsrfg', 'dlssg'), default='off',
                        help='dlssg: NVIDIA DLSS Frame Generation on RDNA3/RDNA4 (d4r/fg/dlssg)')
    parser.add_argument('--input', choices=('dlssg', 'fsrfg', 'fsrfg30', 'upscaler'))
    parser.add_argument('--latency', choices=('proton', 'auto'), default='auto',
                        help='auto: OptiScaler selects latency output; proton: request Reflex (v10) or original NVAPI (0.9)')
    parser.add_argument('--dll', help='dlssg: the DLSS-G DLL to run (default: the game\'s nvngx_dlssg.dll); 310.x only')
    parser.add_argument('--exact', action='store_true',
                        help='dlssg: bit-exact kernels (default: 15-20%% faster, 58 dB against exact on a game capture)')
    parser.add_argument('--lite', action='store_true', help='dlssg: NVIDIA FG Lite (DLSS-G >= 310.9.16)')
    parser.add_argument('--force', action='store_true', help='dlssg: skip the RDNA3/RDNA4 check')
    parser.add_argument('--streamline', type=Path,
                        help='dlssg: folder with Streamline\'s DLLs and nvngx_dlssg.dll (install.sh\'s NVIDIA download): '
                             'games without DLSS-G of their own get it through OptiScaler\'s DLSS-G output')
    args = parser.parse_args()
    runtime = args.runtime or args.game / 'd4r'
    ini = args.game / 'OptiScaler.ini'
    try:
        optifg = False
        if args.fg == 'dlssg':
            optifg = args.streamline is not None and not game_streamline(args.game.resolve())
            dll, arch, cache = setup_dlssg(args.game.resolve(), runtime.resolve(), args.dll, args.exact, args.lite,
                                           args.force)
            if args.streamline is not None:   # OptiScaler looks for Streamline there (its DLSS-G output, its checks)
                stage_streamline(ini, args.streamline.resolve())
        backup = configure(ini, runtime, args.fg, args.input, args.latency, optifg)
        set_d4r_ini(runtime, 'dlssg' if args.fg == 'dlssg' else 'off', args.exact,
                    cache if args.fg == 'dlssg' else None, args.lite)
    except (OSError, ValueError, configparser.Error) as error:
        parser.exit(1, f'{error}\n')
    print(f'Backup: {backup}' if backup else 'Already configured.')
    if args.fg == 'dlssg':
        prefix = proton_prefix(args.game.resolve())
        hags = set_hags(prefix) if prefix else None
        print(f'DLSS-G: {dll} ({arch} kernels in {cache}, {"exact" if args.exact else "fast"} mode'
              f'{", FG Lite" if args.lite else ""}).')
        if arch == 'gfx11':
            print('RDNA3 cost at 1440p (RX 7700 XT), GPU ms per real frame: FG Lite 2.5 for 2x, 3.3 for 4x; full model 5.0 '
                  'for 2x, 10.9 for 4x. d4r.ini [FrameGeneration] Model = auto uses Lite on RDNA3.')
        print(f'Hardware GPU scheduling set in {prefix} (backup {hags}).' if hags else
              'Hardware GPU scheduling already set in the prefix.' if prefix else
              'Proton prefix not found: if the game hides DLSS-G, launch it once, then run this again.')
        print('The game has no DLSS-G of its own: OptiScaler generates the frames (DLSS in the game, FG in OptiScaler\'s '
              'menu).' if optifg else
              'Start the game and enable DLSS Frame Generation (2x/3x/4x) in its settings; no launch options needed.')
        print('Logs: d4r/d4r_boot.log, d4r/fg/dlssg/dlssg_host.log, d4r/fg/dlssg/nvxrt.log.')
        print('After a game update replaces nvngx_dlssg.dll, run this again to translate the new kernels.')
        return
    print('Restart the game. Enable Reflex in-game for latency conversion.')
    if args.fg != 'off':
        print('Enable the selected native FG input in-game (unless upscaler), then activate FG in OptiScaler.')
        print('XeFG: use borderless mode and SDR or HDR10. Verify interpolation with the FG debug view.')


if __name__ == '__main__':
    main()
