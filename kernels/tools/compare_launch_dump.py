#!/usr/bin/env python3
"""Compare two D4R_CUDA_LAUNCH_DUMP_DIR trees byte for byte.

usage: compare_launch_dump.py A_DIR B_DIR [--kernel SUBSTR] [--quiet]

The bridge writes one file per launch and per argument named
`launch-NNN-<kernel>-argNNN-<kind>-<shape>-fmtN.bin` (wine_nvcuda_bridge.c
summarize_resource). The launch number is a global sequence, so it is not a
stable key: a native kernel that runs its own prep shifts every later number.
Files are therefore keyed by (kernel, argument offset, kind, shape, format) and
by how many times that key has occurred so far, which survives a shift as long
as each kernel is launched the same number of times.

Every dump is written after the launch returns, so a surface file is that
layer's own output and a read-only argument (a weight image, a texture) is its
input. Any difference in an output is the kernel's; identical read-only
arguments are the check that the two runs are aligned at all.
"""
import argparse
import os
import re
import sys

NAME = re.compile(r'^launch-(\d+)-(.+)-arg(\d+)-(buffer|surface|texture)-(.+)-fmt(-?\d+)\.bin$')
# A buffer's shape string embeds the device pointer it was found at, which differs
# between runs. Keep the offset within the allocation, drop the base address.
BASE = re.compile(r'base0x[0-9a-f]+')


def index(directory):
    """(kernel, arg, kind, shape, fmt, occurrence) -> (sequence, path)."""
    entries = {}
    counts = {}
    for name in sorted(os.listdir(directory)):
        match = NAME.match(name)
        if match is None:
            continue
        sequence, kernel, argument, kind, shape, fmt = match.groups()
        key = (kernel, int(argument), kind, BASE.sub('base', shape), int(fmt))
        occurrence = counts.get(key, 0)
        counts[key] = occurrence + 1
        entries[key + (occurrence,)] = (int(sequence), os.path.join(directory, name))
    return entries


def first_difference(a, b):
    """(offset, count) of the differing bytes, or None when equal."""
    if len(a) != len(b):
        return None
    count = 0
    first = -1
    for index_ in range(len(a)):
        if a[index_] != b[index_]:
            if first < 0:
                first = index_
            count += 1
    return (first, count) if count else None


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('a')
    parser.add_argument('b')
    parser.add_argument('--kernel', default='', help='only keys whose kernel name contains this')
    parser.add_argument('--quiet', action='store_true', help='report only keys that differ')
    args = parser.parse_args(argv[1:])

    left, right = index(args.a), index(args.b)
    keys = sorted(set(left) | set(right))
    mismatched = missing = compared = 0
    for key in keys:
        if args.kernel and args.kernel not in key[0]:
            continue
        label = '%s arg%03d %s %s fmt%d #%d' % key
        if key not in left or key not in right:
            print('MISSING %s: %s' % (label, 'only in A' if key in left else 'only in B'))
            missing += 1
            continue
        with open(left[key][1], 'rb') as handle:
            a = handle.read()
        with open(right[key][1], 'rb') as handle:
            b = handle.read()
        compared += 1
        difference = first_difference(a, b)
        if difference is None:
            if not args.quiet:
                print('ok      %s (%d bytes)' % (label, len(a)))
        else:
            offset, count = difference
            print('DIFFERS %s: %d of %d bytes, first at %d (A %02x B %02x)' %
                  (label, count, len(a), offset, a[offset], b[offset]))
            mismatched += 1

    print('%d arguments compared, %d differ, %d missing' % (compared, mismatched, missing))
    return 1 if mismatched or missing else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
