#!/usr/bin/env python3
"""Opt-in OptiScaler FG/latency setup. Run with the game closed; no prefix changes."""
import argparse
import configparser
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import tempfile


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


def configure(ini, runtime, fg, fg_input, latency):
    if ini.is_symlink():
        raise ValueError('OptiScaler.ini must be a regular file, not a symlink')
    source = ini.read_bytes().decode('utf-8-sig')
    # Keep the NGX path and all SR/RR choices exactly as they are.
    parsed = configparser.ConfigParser(interpolation=None)
    parsed.read_string(source)
    route = parsed.get('Libraries', 'NvngxPath', fallback='').replace('\\', '/').lower()
    if not route.endswith(('/nvngx.dll', '/d4r_nvngx.dll', '/d4r.asi')):
        raise ValueError('configure d4r NvngxPath before enabling features')
    if fg != 'off' and fg_input is None:
        raise ValueError('--input is required with FG: dlssg, fsrfg, fsrfg30 or upscaler')
    if fg == 'xefg' and latency == 'proton':
        raise ValueError('XeFG requires --latency auto for XeLL marker delivery')
    v10 = parsed.has_option('fakenvapi', 'UseFakenvapi')
    if v10:
        settings = v10_settings(parsed, ini, fg, fg_input, latency)
    else:
        settings = legacy_settings(runtime, fg, fg_input, latency)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('game', type=Path, help='folder containing OptiScaler.ini')
    parser.add_argument('--runtime', type=Path, help='d4r folder (required for a custom ASI plugin folder)')
    parser.add_argument('--fg', choices=('off', 'xefg', 'fsrfg'), default='off')
    parser.add_argument('--input', choices=('dlssg', 'fsrfg', 'fsrfg30', 'upscaler'))
    parser.add_argument('--latency', choices=('proton', 'auto'), default='auto',
                        help='auto: OptiScaler selects latency output; proton: request Reflex (v10) or original NVAPI (0.9)')
    args = parser.parse_args()
    try:
        backup = configure(args.game / 'OptiScaler.ini', args.runtime or args.game / 'd4r',
                           args.fg, args.input, args.latency)
    except (OSError, ValueError, configparser.Error) as error:
        parser.exit(1, f'{error}\n')
    print(f'Backup: {backup}' if backup else 'Already configured.')
    print('Restart the game. Enable Reflex in-game for latency conversion.')
    if args.fg != 'off':
        print('Enable the selected native FG input in-game (unless upscaler), then activate FG in OptiScaler.')
        print('XeFG: use borderless mode and SDR or HDR10. Verify interpolation with the FG debug view.')


if __name__ == '__main__':
    main()
