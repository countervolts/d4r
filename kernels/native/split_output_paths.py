#!/usr/bin/env python3
"""Split a translated hiluma output kernel into its two per-block paths (opt-in experiment).

usage: split_output_paths.py KERNEL.hip > KERNEL_split.hip

NVIDIA's hiluma_engine_output_* kernels pick one of two code paths per block: path B (an LDS tile of the
block's render footprint) when the footprint is under 24 texels in both axes, which holds for every block at
Quality .. Ultra Performance, else path A (direct fetches, DLAA). The choice depends only on the block index and
kernel parameters, so it is uniform per block, and blocks share nothing. Path A needs more registers than path B
(LDR mvlo: 101 vs 88 VGPRs), and one kernel carries the larger allocation for every block.

This pass emits the same translated body twice:
  NAME        runs path B and returns where path A would start;
  NAME_post1  runs path A and returns where path B would start.
The ZLUDA/MicroCUDA native hook launches NAME_post1 right after NAME on the same stream with the same parameters
and grid (d4r_post1_grid_x/_y = 0), with the 16x16 blocks the PTX hard-codes (tid.y << 4). Each block executes
exactly the instructions it executed before; only the block order across the two paths changes, which no output
depends on (blocks never read what other blocks write in the same launch).

The pass refuses anything but the expected entry structure:
    pA = ((int32_t)rX < (int32_t)24);  pB = ((int32_t)rY < (int32_t)24);  pC = (pA && pB);
    if (pC) goto LB;
    goto LA;
"""
import re
import sys


def split(source):
    head, sep, main = source.partition('extern "C" __attribute__((global))')
    if not sep or 'extern "C" __attribute__((global))' in main:
        raise SystemExit('expected exactly one kernel')
    m = re.match(r'\s*__attribute__\(\(amdgpu_flat_work_group_size\(1, 256\)\)\) void (\w+)\((\w+)\)', main)
    if not m:
        raise SystemExit('unexpected kernel signature')
    name = m.group(1)
    pattern = re.compile(r'^    (p\d+) = \(\(int32_t\)r\d+ < \(int32_t\)24\);\n'
                         r'    (p\d+) = \(\(int32_t\)r\d+ < \(int32_t\)24\);\n'
                         r'    (p\d+) = \((p\d+) && (p\d+)\);\n'
                         r'    if \((p\d+)\) goto (L__BB0_\d+);\n'
                         r'    goto (L__BB0_\d+);\n', re.M)
    found = list(pattern.finditer(main))
    if len(found) != 1:
        raise SystemExit(f'path selection found {len(found)} times, expected once')
    g = found[0]
    pa, pb, pc, x, y, pc2, path_b, path_a = g.groups()
    if {x, y} != {pa, pb} or pc != pc2:
        raise SystemExit('path selection predicate has an unexpected form')
    # the predicate must not be reassigned anywhere (it is the block's path choice)
    if len(re.findall(rf'^    {pc} = ', main, re.M)) != 1:
        raise SystemExit(f'{pc} is assigned more than once')
    sel = g.group(0)
    lines = sel.splitlines(keepends=True)
    main_b = main.replace(sel, ''.join(lines[:4]) + '    return; // path A blocks: NAME_post1\n', 1)
    main_a = main.replace(sel, ''.join(lines[:3]) + f'    if ({pc}) return; // path B blocks: NAME\n'
                          + f'    goto {path_a};\n', 1)
    main_a = main_a.replace(f' void {name}(', f' void {name}_post1(', 1)
    globals_ = ('\n// d4r native hook: launch NAME_post1 after NAME with the launch grid and the PTX\'s 16x16 blocks\n'
                'extern "C" __attribute__((device, used)) uint32_t d4r_post1_grid_x = 0;\n'
                'extern "C" __attribute__((device, used)) uint32_t d4r_post1_grid_y = 0;\n'
                'extern "C" __attribute__((device, used)) uint32_t d4r_post1_block_x = 16;\n'
                'extern "C" __attribute__((device, used)) uint32_t d4r_post1_block_y = 16;\n'
                'extern "C" __attribute__((device, used)) uint32_t d4r_post1_block_z = 1;\n')
    banner = '// split_output_paths.py: NAME = path B only, NAME_post1 = path A only (per-block choice unchanged)\n'
    return (banner + head + globals_ + sep + main_b + '\n' + sep + main_a, path_b, path_a)


if __name__ == '__main__':
    text, b, a = split(open(sys.argv[1]).read())
    sys.stderr.write(f'split: path B {b}, path A {a}\n')
    sys.stdout.write(text)
