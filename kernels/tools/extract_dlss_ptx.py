#!/usr/bin/env python3
"""Extract the PTX modules embedded in NVIDIA's nvngx_dlss.dll.

usage: extract_dlss_ptx.py NVNGX_DLSS_DLL OUT_DIR

Writes OUT_DIR/dlss-NNNN-MM.ptx for PTX entry MM of the NNNN-th CUDA fatbin found in the DLL (in file
order). The texture-kernel build (kernels/tex) edits some of these modules; the DLL is NVIDIA's and is
never part of this repository, and neither are the extracted files.
"""
import os
import struct
import sys

FATBIN_MAGIC = 0xBA55ED50


def lz4_block(src: bytes, size: int) -> bytes:
    dst = bytearray()
    i = 0
    while i < len(src):
        token = src[i]
        i += 1
        literals = token >> 4
        if literals == 15:
            while True:
                b = src[i]
                i += 1
                literals += b
                if b != 255:
                    break
        dst += src[i:i + literals]
        i += literals
        if i >= len(src) or len(dst) >= size:
            break
        offset = src[i] | (src[i + 1] << 8)
        i += 2
        match = token & 15
        if match == 15:
            while True:
                b = src[i]
                i += 1
                match += b
                if b != 255:
                    break
        match += 4
        for _ in range(match):
            dst.append(dst[-offset])
    return bytes(dst[:size])


def fatbins(data: bytes):
    """Yields (offset, bytes) of every fatbin container (magic, version 1) in file order."""
    magic = struct.pack('<I', FATBIN_MAGIC)
    start = data.find(magic)
    while start >= 0:
        _, version, header_size, size = struct.unpack_from('<IHHQ', data, start)
        if version == 1 and header_size == 16 and start + header_size + size <= len(data):
            yield start, data[start:start + header_size + size]
            start = data.find(magic, start + header_size + size)
        else:
            start = data.find(magic, start + 4)


def ptx_entries(fatbin: bytes):
    """Yields the (decompressed) PTX text of each PTX entry of one fatbin."""
    _, _, header_size, size = struct.unpack_from('<IHHQ', fatbin, 0)
    offset = header_size
    while offset < header_size + size:
        kind, _, entry_header, entry_size = struct.unpack_from('<HHIQ', fatbin, offset)
        flags = struct.unpack_from('<Q', fatbin, offset + 40)[0]
        decompressed = struct.unpack_from('<Q', fatbin, offset + 56)[0] if entry_header >= 64 else 0
        payload = fatbin[offset + entry_header:offset + entry_header + entry_size]
        if kind == 1:  # PTX
            yield lz4_block(payload, decompressed) if (flags & 0x2000 and decompressed) else payload
        offset += entry_header + entry_size


def main(argv):
    if len(argv) != 3:
        sys.exit(__doc__)
    data = open(argv[1], 'rb').read()
    os.makedirs(argv[2], exist_ok=True)
    total = 0
    for index, (_, fatbin) in enumerate(fatbins(data)):
        for entry, ptx in enumerate(ptx_entries(fatbin)):
            ptx = ptx.rstrip(b'\0')
            with open(os.path.join(argv[2], f'dlss-{index:04d}-{entry:02d}.ptx'), 'wb') as out:
                out.write(ptx)
            total += 1
    print(f'{total} PTX modules written to {argv[2]}')


if __name__ == '__main__':
    main(sys.argv)
