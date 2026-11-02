#!/usr/bin/env python3
"""Make a 64-bit PE DLL import DLL!SYMBOL before all of its other imports.

usage: pe_add_import.py IN OUT [--dll d4r_boot.dll] [--symbol d4r_boot]

The release uses it on OptiScaler (the game folder's dxgi.dll) so that d4r_boot.dll is loaded, and its DllMain run,
before OptiScaler's own DllMain. A new section holds a copy of the import directory with the new descriptor first;
the original descriptors and their thunk arrays are left where they are, so no code or data address changes. Bound
imports and an Authenticode signature (invalid after any change) are dropped and the checksum is recomputed.
A file that already imports DLL is copied unchanged."""
import argparse
import struct
import sys
from pathlib import Path

SECTION_NAME = b'.d4rimp\0'
SCN_FLAGS = 0x40 | 0x40000000 | 0x80000000  # initialized data, read, write (the IAT is written by the loader)


def align(value, to):
    return (value + to - 1) // to * to


def checksum(data, checksum_offset):
    total = 0
    padded = data + b'\0' * (len(data) % 2)
    for (word,) in struct.iter_unpack('<H', padded):
        total += word
        total = (total & 0xffff) + (total >> 16)
    # the checksum field itself counts as zero
    for (word,) in struct.iter_unpack('<H', data[checksum_offset:checksum_offset + 4]):
        total -= word
        if total < 0:
            total += 0xffff
        total = (total & 0xffff) + (total >> 16)
    total = (total & 0xffff) + (total >> 16)
    return (total & 0xffff) + len(data)


class Pe:
    def __init__(self, data):
        self.data = bytearray(data)
        if data[:2] != b'MZ':
            raise ValueError('not a PE file')
        self.nt = struct.unpack_from('<I', data, 0x3c)[0]
        if data[self.nt:self.nt + 4] != b'PE\0\0':
            raise ValueError('not a PE file')
        coff = self.nt + 4
        self.machine, self.nsections = struct.unpack_from('<HH', data, coff)
        self.opt_size = struct.unpack_from('<H', data, coff + 16)[0]
        self.opt = coff + 20
        if struct.unpack_from('<H', data, self.opt)[0] != 0x20b:
            raise ValueError('only PE32+ (64-bit) files are supported')
        self.section_alignment, self.file_alignment = struct.unpack_from('<II', data, self.opt + 32)
        self.size_of_headers = struct.unpack_from('<I', data, self.opt + 60)[0]
        self.ndirs = struct.unpack_from('<I', data, self.opt + 108)[0]
        self.sections_at = self.opt + self.opt_size

    def directory(self, index):
        return struct.unpack_from('<II', self.data, self.opt + 112 + 8 * index) if index < self.ndirs else (0, 0)

    def set_directory(self, index, rva, size):
        struct.pack_into('<II', self.data, self.opt + 112 + 8 * index, rva, size)

    def sections(self):
        for i in range(self.nsections):
            at = self.sections_at + 40 * i
            name = bytes(self.data[at:at + 8])
            vsize, va, raw_size, raw_ptr = struct.unpack_from('<IIII', self.data, at + 8)
            yield name, vsize, va, raw_size, raw_ptr

    def offset(self, rva):
        for _, vsize, va, raw_size, raw_ptr in self.sections():
            if va <= rva < va + max(vsize, raw_size):
                return raw_ptr + rva - va
        raise ValueError('RVA %#x is outside every section' % rva)

    def cstring(self, rva):
        at = self.offset(rva)
        return bytes(self.data[at:self.data.index(0, at)])

    def import_descriptors(self):
        rva, _ = self.directory(1)
        out = []
        if not rva:
            return out
        at = self.offset(rva)
        while True:
            d = bytes(self.data[at:at + 20])
            if d == b'\0' * 20:
                return out
            out.append(d)
            at += 20


def import_names(data):
    """the DLL names of the import descriptors, in load order (objdump -p shows only those in the first section)"""
    pe = Pe(data)
    return [pe.cstring(struct.unpack_from('<I', d, 12)[0]).decode() for d in pe.import_descriptors()]


def add_import(data, dll, symbol):
    pe = Pe(data)
    descriptors = pe.import_descriptors()
    for d in descriptors:
        name_rva = struct.unpack_from('<I', d, 12)[0]
        if pe.cstring(name_rva).lower() == dll.lower().encode():
            return bytes(data), False

    # an Authenticode signature sits at the end of the file and is no longer valid
    sec_ptr, sec_size = pe.directory(4)
    if sec_ptr:
        if sec_ptr + sec_size < len(pe.data) - 8:
            raise ValueError('signature is not at the end of the file')
        del pe.data[sec_ptr:]
        pe.set_directory(4, 0, 0)
    # data after the last section (a COFF symbol table, an overlay) stays where it is; the new section follows it
    sections = list(pe.sections())
    header_end = pe.sections_at + 40 * (pe.nsections + 1)
    first_raw = min(raw_ptr for *_, raw_ptr in sections if raw_ptr)
    if header_end > pe.size_of_headers or header_end > first_raw:
        raise ValueError('no room for another section header')

    va = align(max(va + max(vsize, 1) for _, vsize, va, _, _ in sections), pe.section_alignment)
    # layout: descriptors (new, originals, terminator) | ILT[2] | IAT[2] | hint/name | dll name
    ndesc = len(descriptors) + 2
    ilt = 20 * ndesc
    ilt = align(ilt, 8)
    iat = ilt + 16
    hint = iat + 16
    name = align(hint + 2 + len(symbol) + 1, 2)
    end = name + len(dll) + 1
    body = bytearray(end)
    struct.pack_into('<IIIII', body, 0, va + ilt, 0, 0, va + name, va + iat)
    for i, d in enumerate(descriptors):
        body[20 * (i + 1):20 * (i + 2)] = d
        struct.pack_into('<I', body, 20 * (i + 1) + 4, 0)  # TimeDateStamp: not bound (the bound directory goes)
    struct.pack_into('<QQ', body, ilt, va + hint, 0)
    struct.pack_into('<QQ', body, iat, va + hint, 0)
    struct.pack_into('<H', body, hint, 0)
    body[hint + 2:hint + 2 + len(symbol)] = symbol.encode()
    body[name:name + len(dll)] = dll.encode()

    raw_ptr = align(len(pe.data), pe.file_alignment)
    raw_size = align(len(body), pe.file_alignment)
    pe.data += b'\0' * (raw_ptr - len(pe.data))
    pe.data += body + b'\0' * (raw_size - len(body))
    header = SECTION_NAME + struct.pack('<IIIIIIHHI', len(body), va, raw_size, raw_ptr, 0, 0, 0, 0, SCN_FLAGS)
    pe.data[pe.sections_at + 40 * pe.nsections:header_end] = header
    struct.pack_into('<H', pe.data, pe.nt + 6, pe.nsections + 1)
    struct.pack_into('<I', pe.data, pe.opt + 56, align(va + len(body), pe.section_alignment))  # SizeOfImage
    struct.pack_into('<I', pe.data, pe.opt + 8, struct.unpack_from('<I', pe.data, pe.opt + 8)[0] + raw_size)  # SizeOfInitializedData
    pe.set_directory(1, va, 20 * ndesc)
    if pe.ndirs > 11:
        pe.set_directory(11, 0, 0)  # bound imports: the copied descriptors' time stamps are kept, binding is not
    struct.pack_into('<I', pe.data, pe.opt + 64, 0)
    struct.pack_into('<I', pe.data, pe.opt + 64, checksum(bytes(pe.data), pe.opt + 64))
    return bytes(pe.data), True


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('input', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--dll', default='d4r_boot.dll')
    p.add_argument('--symbol', default='d4r_boot')
    a = p.parse_args()
    try:
        data, changed = add_import(a.input.read_bytes(), a.dll, a.symbol)
    except ValueError as error:
        sys.exit('%s: %s' % (a.input, error))
    a.output.write_bytes(data)
    print('%s: %s' % (a.output, 'imports %s first' % a.dll if changed else 'already imports %s' % a.dll))


if __name__ == '__main__':
    main()
