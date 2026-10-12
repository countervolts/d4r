"""Ray Reconstruction captures: the launches of one denoiser frame as the CUDA bridge recorded them.

A capture directory holds what the bridge writes with
  D4R_CUDA_VERBOSE=1 D4R_CUDA_REPLAY_DUMP_DIR=CAP/replay D4R_CUDA_REPLAY_DUMP_FILTER=cuda_dldn
  D4R_CUDA_LAUNCH_STATS=1 D4R_CUDA_LAUNCH_STATS_FILTER=cuda_dldn D4R_CUDA_LAUNCH_DUMP_DIR=CAP/dump
plus the run's log as CAP/run/run.log (texture sampler settings and array shapes come from the verbose
cuTexObjectCreate / cuSurfObjectCreate / cuArrayCreate lines).

replay/replay-NNNNNN-KERNEL/  args.bin, manifest.txt and every buffer the launch points into, *before* it ran
dump/launch-NNN-KERNEL-argOOO-{buffer,texture,surface}-...  every resource *after* the launch ran

Captures are reference data only: nothing at runtime reads them.
"""
from dataclasses import dataclass, field
from pathlib import Path
import re
import struct

# CUarray_format codes the denoiser uses, and how a Vulkan image of the same texels is described
# (kvk's BITS column: 8 = UNORM bytes, 9 = SNORM bytes, 16 = binary16, 32 = binary32)
ARRAY_FORMATS = {16: 16, 32: 32, 1: 8, 200: 9, 194: 8}


@dataclass
class Image:
    offset: int
    kind: str                # 'texture' or 'surface'
    width: int
    height: int
    channels: int
    format: int              # CUarray_format
    path: Path               # post-launch contents
    linear: bool = False     # textures: bilinear filtering

    @property
    def bits(self):
        return ARRAY_FORMATS[self.format]

    @property
    def texel_bytes(self):
        return self.channels * (1 if self.bits in (8, 9) else self.bits // 8)


@dataclass
class Pointer:
    offset: int
    alloc: int
    delta: int


@dataclass
class Launch:
    sequence: int
    kernel: str
    directory: Path
    grid: tuple
    block: tuple
    args: bytes
    allocs: dict = field(default_factory=dict)      # index -> (base, size)
    pointers: list = field(default_factory=list)
    images: list = field(default_factory=list)

    def pre(self, alloc):
        """an allocation's contents before the launch"""
        return (self.directory / f'alloc-{alloc}.bin').read_bytes()

    def pointer(self, offset):
        return next(p for p in self.pointers if p.offset == offset)

    def image(self, offset):
        return next(i for i in self.images if i.offset == offset)

    @property
    def weights(self):
        """the weight image (argument 40) as NGX uploaded it; 16 zero bytes for a stage without one (the output stage)"""
        pointer = next((p for p in self.pointers if p.offset == 40), None)
        return self.pre(pointer.alloc) if pointer is not None else bytes(16)

    @property
    def arena_alloc(self):
        """the plane arena: the allocation every arena pointer of the layer points into"""
        candidates = [p.alloc for p in self.pointers if p.offset != 40]
        return max(set(candidates), key=lambda a: self.allocs[a][1]) if candidates else None

    def post_buffer(self, capture, alloc):
        base = self.allocs[alloc][0]
        for path in (capture.root / 'dump').glob(f'launch-{self.sequence:03d}-{self.kernel}-arg*-buffer-*'):
            m = re.search(r'-base0x([0-9a-f]+)\+', path.name)
            if m and int(m.group(1), 16) == base:
                return path.read_bytes()
        raise FileNotFoundError(f'no post-launch dump of allocation {alloc} for launch {self.sequence}')

    def u32(self, offset):
        return struct.unpack_from('<I', self.args, offset)[0]

    def f32(self, offset):
        return struct.unpack_from('<f', self.args, offset)[0]


class Capture:
    def __init__(self, root):
        self.root = Path(root)
        log_path = self.root / 'run' / 'run.log'
        self.log = log_path.read_text(errors='replace') if log_path.exists() else ''
        # (log position, texture handle, filter mode) of every texture object creation, in order
        self._tex_events = [(m.start(), int(m.group(2), 16), int(m.group(1))) for m in re.finditer(
            r'cuTexObjectCreate .*? filter=(\d+) .*? object=(0x[0-9a-f]+)', self.log)]
        self.launches = []
        for d in sorted((self.root / 'replay').iterdir()):
            self.launches.append(self._launch(d))

    def filter_at(self, handle, position):
        value = None
        for pos, h, f in self._tex_events:
            if pos > position:
                break
            if h == handle:
                value = f
        return value

    def _launch(self, directory):
        manifest = (directory / 'manifest.txt').read_text().split('\n')
        kernel = manifest[0].split()[1]
        launch = [int(x) for x in manifest[1].split()[1:]]
        sequence = int(re.search(r'replay-(\d+)-', directory.name).group(1))
        args = (directory / 'args.bin').read_bytes()
        result = Launch(sequence, kernel, directory, tuple(launch[0:3]), tuple(launch[3:6]), args)
        for line in manifest[2:]:
            f = line.split()
            if not f:
                continue
            if f[0] == 'alloc':
                result.allocs[int(f[1])] = (int(f[2], 16), int(f[3]))
            elif f[0] == 'pointer':
                result.pointers.append(Pointer(int(f[1]), int(f[2]), int(f[3])))
        # the log position of this launch, so a reused texture handle resolves to its settings at the time
        position = self.log.find(f'launch[{sequence}] {kernel}')
        if position < 0:
            position = len(self.log)
        for path in sorted((self.root / 'dump').glob(f'launch-{sequence:03d}-{kernel}-arg*')):
            m = re.search(r'-arg(\d+)-(texture|surface)-(\d+)x(\d+)x(\d+)-fmt(-?\d+)\.bin$', path.name)
            if not m:
                continue
            offset = int(m.group(1))
            image = Image(offset, m.group(2), int(m.group(3)), int(m.group(4)), int(m.group(5)), int(m.group(6)), path)
            if image.kind == 'texture':
                handle = struct.unpack_from('<Q', args, offset)[0]
                image.linear = self.filter_at(handle, position) == 1
            result.images.append(image)
        result.images.sort(key=lambda i: (i.kind != 'texture', i.offset))
        return result

    def find(self, kernel_part, frame=None):
        """the launches whose kernel name contains KERNEL_PART, in order; FRAME (1-based) selects one"""
        found = [l for l in self.launches if kernel_part in l.kernel]
        return found if frame is None else found[frame - 1]
