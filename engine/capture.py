"""Capture decoding shared by full-frame validation and model packaging.

Captures are reference data only. The runtime has no dependency on NGX, CUDA,
ZLUDA, PTX or the pointer addresses contained in a capture.
"""
from dataclasses import dataclass
from pathlib import Path
import re
import struct
import numpy as np

INPUT = 'hiluma_engine_input_depthinv_mvlo_hdr_v2_rel'
OUTPUT = 'hiluma_engine_output_depthinv_mvlo_hdr_max_v2_rel'
ORDER = ['enc0', 'enc1', 'enc2', 'enc3', 'enc4', 'dec5', 'dec4', 'dec3', 'dec2', 'dec1', 'dec0']


@dataclass
class Resource:
    path: Path
    width: int
    height: int
    channels: int
    bits: int

    def validate(self):
        if self.path.stat().st_size != self.width * self.height * self.channels * self.bits // 8:
            raise ValueError(f'truncated texture/surface: {self.path}')


class Launch:
    def __init__(self, capture, directory):
        self.capture, self.directory = Path(capture), Path(directory)
        self.sequence = int(re.search(r'replay-(\d+)-', self.directory.name).group(1))
        self.args = (self.directory / 'args.bin').read_bytes()
        self.manifest = (self.directory / 'manifest.txt').read_text().splitlines()
        self.kernel = self.manifest[0].split()[1]
        self.resources = {}
        for f in (self.capture / 'dump').glob(f'launch-{self.sequence:03d}-{self.kernel}-arg*'):
            match = re.search(r'-arg(\d+)-(texture|surface)-(\d+)x(\d+)x(\d+)-fmt(\d+)\.bin$', f.name)
            if match:
                offset, kind, w, h, c, fmt = match.groups()
                resource = Resource(f.resolve(), int(w), int(h), int(c), 32 if fmt == '32' else 16)
                self.resources[int(offset)] = resource

    def f2(self, offset): return struct.unpack_from('<2f', self.args, offset)
    def i2(self, offset): return struct.unpack_from('<2i', self.args, offset)
    def f1(self, offset): return struct.unpack_from('<f', self.args, offset)[0]
    def h1(self, offset): return float(np.frombuffer(self.args, np.float16, 1, offset)[0])
    def exposure(self, offset):
        self.resources[offset].validate()
        value = float(np.fromfile(self.resources[offset].path, np.float16, 1)[0])
        return value if value != 0 else 1.0

    def buffer(self, offset, size):
        pointer = struct.unpack_from('<Q', self.args, offset)[0]
        # Post-launch allocation captures include each entire allocation, not just this argument's slice.
        for path in (self.capture / 'dump').glob(f'launch-{self.sequence:03d}-{self.kernel}-arg*-buffer-*'):
            base = int(re.search(r'-base0x([0-9a-f]+)', path.name).group(1), 16)
            delta = pointer - base
            if 0 <= delta and delta + size <= path.stat().st_size:
                with path.open('rb') as stream:
                    stream.seek(delta)
                    return stream.read(size)
        for line in self.manifest:
            fields = line.split()
            if fields[0] == 'pointer' and int(fields[1]) == offset:
                path = self.directory / f'alloc-{fields[2]}.bin'
                if path.exists():
                    with path.open('rb') as stream:
                        stream.seek(int(fields[3])); data = stream.read(size)
                    if len(data) == size: return data
        raise ValueError(f'{self.directory}: missing {size}-byte buffer at argument {offset}')

    def buffer_location(self, offset):
        """(file holding the whole allocation, byte offset of the argument's pointer within it)"""
        pointer = struct.unpack_from('<Q', self.args, offset)[0]
        for path in (self.capture / 'dump').glob(f'launch-{self.sequence:03d}-{self.kernel}-arg*-buffer-*'):
            base = int(re.search(r'-base0x([0-9a-f]+)', path.name).group(1), 16)
            if 0 <= pointer - base < path.stat().st_size:
                return path, pointer - base
        for line in self.manifest:
            fields = line.split()
            if fields[0] == 'pointer' and int(fields[1]) == offset and (self.directory / f'alloc-{fields[2]}.bin').exists():
                return self.directory / f'alloc-{fields[2]}.bin', int(fields[3])
        raise ValueError(f'{self.directory}: missing buffer at argument {offset}')

    def buffer_offset(self, offset):
        return self.buffer_location(offset)[1]

    def allocation(self, offset, size):
        data = self.buffer_location(offset)[0].read_bytes()
        if len(data) != size:
            raise ValueError(f'{self.directory}: the allocation at argument {offset} has {len(data)} bytes, expected {size}')
        return data


def launches(capture, kernel):
    return [Launch(capture, d) for d in sorted((Path(capture) / 'replay').glob('replay-*-'+kernel))]


def input_params(p):
    f = np.float32
    a, b, c = (f(p.exposure(o)) for o in (208, 200, 216))
    f152, f158 = f(p.f1(144)) * a, f(p.f1(144)) * b
    expo = f(p.f1(156)) * f158
    if expo == 0: expo = f(1)
    expo_in, ratio = f(p.f1(148)) * c / f158, f158 / f152
    blob = struct.pack('<14f', *p.f2(40), *p.f2(24), *p.f2(8), *p.f2(16), *p.f2(120), *p.f2(136), *p.f2(128))
    blob += struct.pack('<12i', *p.i2(72), *p.i2(80), *p.i2(88), *p.i2(96), *p.i2(104), *p.i2(112))
    blob += struct.pack('<iI4f', struct.unpack_from('<i', p.args, 64)[0], p.args[32], float(expo), float(expo_in), float(ratio), p.f1(160))
    assert len(blob) == 128
    return blob


def output_params(p):
    if any(p.args[89:92]):
        raise ValueError('native K supports standard HDR output only (flag bytes 89..91 must be zero)')
    f = np.float32
    a, b = f(p.exposure(264)), f(p.exposure(256))
    s160, s168 = f(p.f1(160)), f(p.f1(168))
    expo = s168 * (s160 * b)
    if expo == 0: expo = f(1)
    ratio = (s160 * b) / (s160 * a)
    blob = struct.pack('<18f', *p.f2(40), *p.f2(16), *p.f2(56), *p.f2(48), *p.f2(64), *p.f2(0), *p.f2(8), *p.f2(144), *p.f2(152))
    blob += struct.pack('<16i', *p.i2(208), *p.i2(216), *p.i2(96), *p.i2(104), *p.i2(112), *p.i2(120), *p.i2(128), *p.i2(136))
    # flags: reset, and in bits 8..11 the reconstruction table the launch pointed at within NGX's sixteen
    blob += struct.pack('<2iI', *p.i2(80), p.args[32] | (p.buffer_offset(280) // 32768) << 8)
    blob += struct.pack('<10f', float(expo), float(ratio), *struct.unpack_from('<3f', p.args, 172), p.h1(188), p.h1(190), p.h1(192), p.h1(184), p.h1(186))
    assert len(blob) == 188
    return blob
