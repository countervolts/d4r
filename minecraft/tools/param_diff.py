#!/usr/bin/env python3 -I
"""Which bytes of each kernel's parameter block vary between launches?
usage: param_diff.py nvxrt.log   (log made with NVXRT_TRACE=1 NVXRT_TRACE_RAW=1 via fg/dlssg/harness/run.sh)
Varying offsets that are 8-byte aligned and small ints are bindless image handles; ffff8001... are device pointers;
anything else varying is a real per-frame scalar the native engine has to compute."""
import re, sys, collections
name = None; blocks = collections.defaultdict(list)
for l in open(sys.argv[1], errors="replace"):
    m = re.search(r"launch #\d+ (\w+)\[", l)
    if m: name = m.group(1); continue
    m = re.match(r"nvxrt:\s+raw ([0-9a-f]+)", l)
    if m and name: blocks[name].append(m.group(1)); name = None
for k, b in blocks.items():
    n = len(b[0]) // 2
    var = [i for i in range(n) if len({x[2*i:2*i+2] for x in b if len(x) == len(b[0])}) > 1]
    print(f"{k:40s} launches {len(b):3d} bytes {n:3d} varying {var}")
