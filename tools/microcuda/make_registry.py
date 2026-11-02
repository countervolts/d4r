#!/usr/bin/env python3
"""Offline inventory/registry. No proprietary PTX is emitted into the header.

make_registry.py DLL NATIVE_DIR OUT_DIR [NAME ...]
Only audited K enc0 export is enabled for now. Inventory covers every PTX.
Hash identity: existing PTX FNV plus full fatbin FNV+size to avoid runtime LZ4.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'kernels/tools'))
from extract_dlss_ptx import fatbins, ptx_entries
from kernel_manifest import fnv1a64

ENTRY = re.compile(rb'\.entry\s+([\w$]+)\s*\((.*?)\)', re.S)
AUDITED = {'dltss_pwin_enc0_layer'}


def generate(dll, native, out, names):
    if not set(names) <= AUDITED:
        raise ValueError('Only K enc0 is ABI/phase audited; extend AUDITED after validation')
    data = dll.read_bytes()
    manifest = set()
    for line in (native / 'd4r-kernels.txt').read_text().splitlines():
        fields = line.split()
        if len(fields) == 2 and not line.startswith('#'):
            manifest.add((fields[0], int(fields[1], 16)))
    inventory, enabled = [], []
    for i, (_, fatbin) in enumerate(fatbins(data)):
        for j, ptx in enumerate(ptx_entries(fatbin)):
            ptx = ptx.rstrip(b'\0')
            entries = ENTRY.findall(ptx)
            phash, fhash = fnv1a64(ptx), fnv1a64(fatbin)
            row = {'fatbin_index': i, 'ptx_index': j, 'ptx_hash': f'{phash:016x}',
                   'fatbin_hash': f'{fhash:016x}', 'fatbin_bytes': len(fatbin),
                   'kernels': []}
            for raw_name, params in entries:
                name = raw_name.decode()
                m = re.fullmatch(rb'\s*\.param\s+\.align\s+(\d+)\s+\.b8\s+\w+\[(\d+)\]\s*', params)
                abi = {'align': int(m[1]), 'bytes': int(m[2])} if m else None
                row['kernels'].append({'name': name, 'aggregate_abi': abi,
                    'native_manifest_match': (name, phash) in manifest,
                    'native_file': (native / (name + '.hsaco')).is_file()})
                if name not in names:
                    continue
                if abi != {'align': 8, 'bytes': 176}:
                    raise ValueError(f'{name}: unsupported module/ABI')
                if (name, phash) not in manifest:
                    raise ValueError(f'{name}: PTX hash not verified by native manifest')
                hsaco = native / (name + '.hsaco')
                notes = subprocess.check_output(['llvm-readobj', '--notes', str(hsaco)], text=True)
                # Verify both prep and main kernargs before generating a runnable registry.
                for symbol in [name + '_prep', name]:
                    blocks = notes.split('  - .args:')[1:]
                    block = next((b for b in blocks if re.search(r'\.name:\s+' + re.escape(symbol) + r'\s', b)), '')
                    if not (re.search(r'\.size:\s+176\s', block) and
                            re.search(r'\.offset:\s+0\s', block) and
                            re.search(r'\.kernarg_segment_size:\s+176\s', block) and
                            block.count('.value_kind:') == 1 and
                            re.search(r'\.value_kind:\s+by_value\s', block) and
                            re.search(r'\.kernarg_segment_align:\s+8\s', block)):
                        raise ValueError(f'{symbol}: HSACO argument ABI mismatch')
                arch = re.search(r'amdhsa.target:\s+amdgcn-amd-amdhsa--(gfx\d+)', notes)
                if not arch:
                    raise ValueError('Missing exact GPU target in HSACO')
                enabled.append((phash, len(ptx), fhash, len(fatbin), name, abi['bytes'],
                                arch[1], hashlib.sha256(hsaco.read_bytes()).hexdigest()))
            inventory.append(row)
    if {r[4] for r in enabled} != set(names):
        raise ValueError('Requested kernel absent from DLL')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'inventory.json').write_text(json.dumps({'dll_sha256': hashlib.sha256(data).hexdigest(),
                                                   'modules': inventory}, indent=2) + '\n')
    lines = ['// Generated offline; do not edit. Native bytes/ABI were verified at generation.',
             'static const RegistryEntry registry[] = {']
    for ph, ps, fh, fs, name, size, arch, digest in sorted(set(enabled)):
        lines.append(f'    {{0x{ph:016x}ULL, {ps}, 0x{fh:016x}ULL, {fs}, "{name}", {size}, "{arch}", "{digest}"}},')
    lines += ['};', '']
    (out / 'registry.h').write_text('\n'.join(lines))
    print(f'{len(inventory)} PTX records inventoried; {len(enabled)} registry entries enabled')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('dll', type=Path)
    p.add_argument('native', type=Path)
    p.add_argument('out', type=Path)
    p.add_argument('names', nargs='*', default=['dltss_pwin_enc0_layer'])
    a = p.parse_args()
    generate(a.dll, a.native, a.out, a.names)


if __name__ == '__main__':
    main()
