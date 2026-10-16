#!/usr/bin/env python3
"""Write the d4r-kernels.txt manifest of a native kernel directory.

usage: kernel_manifest.py KERNEL_DIR NVNGX_DLSS_DLL [NVNGX_DLSS_DLL...]

For every NAME.hsaco in KERNEL_DIR, finds the PTX module of each DLL that defines `.entry NAME` and
lists the FNV-1a 64 hash of the module's text (trailing NUL bytes removed), one "NAME HASH" line per
distinct hash. The Wine CUDA bridge serves a native kernel only while DLSS loads a module with a
listed hash, so a DLSS version that changed the kernel runs ZLUDA's own compile of it instead.
The manifest is written to KERNEL_DIR/d4r-kernels.txt; it holds hashes, not NVIDIA code.
"""
import hashlib
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_dlss_ptx import fatbins, ptx_entries  # noqa: E402

ENTRY = re.compile(rb'\.entry[ \t\n]+([A-Za-z0-9_$]+)[ \t\n\r]*\(')


def fnv1a64(data: bytes) -> int:
    value = 0xcbf29ce484222325
    for byte in data:
        value = ((value ^ byte) * 0x100000001b3) & 0xffffffffffffffff
    return value


def module_hashes(dll: str) -> dict:
    """NAME -> set of module hashes for every PTX entry of one DLL."""
    data = open(dll, 'rb').read()
    found = {}
    for _, fatbin in fatbins(data):
        for ptx in ptx_entries(fatbin):
            text = ptx.rstrip(b'\0')
            value = fnv1a64(text)
            for match in ENTRY.finditer(text):
                found.setdefault(match.group(1).decode(), set()).add(value)
    return found


def main(argv):
    if len(argv) < 3:
        sys.exit(__doc__)
    directory = argv[1]
    names = sorted(name[:-len('.hsaco')] for name in os.listdir(directory) if name.endswith('.hsaco'))
    lines = ['# d4r native kernels: NAME and the FNV-1a 64 hash of the DLSS PTX module each was written for',
             '# (written by kernels/tools/kernel_manifest.py from:']
    listed = {name: set() for name in names}
    for dll in argv[2:]:
        digest = hashlib.sha256(open(dll, 'rb').read()).hexdigest()[:16]
        lines.append(f'#   {os.path.basename(dll)} sha256 {digest}...)')
        hashes = module_hashes(dll)
        for name in names:
            listed[name] |= hashes.get(name, set())
    missing = [name for name in names if not listed[name]]
    if missing:
        sys.exit(f'no DLL defines {", ".join(missing)}; not writing a manifest')
    for name in names:
        for value in sorted(listed[name]):
            lines.append(f'{name} {value:016x}')
    with open(os.path.join(directory, 'd4r-kernels.txt'), 'w') as out:
        out.write('\n'.join(lines) + '\n')
    print(f'{len(names)} kernels listed in {os.path.join(directory, "d4r-kernels.txt")}')


if __name__ == '__main__':
    main(sys.argv)
