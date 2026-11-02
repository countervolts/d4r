#!/usr/bin/env python3
"""Retarget d4r's vkd3d forwarder to a private core DLL without changing code RVAs."""
from pathlib import Path
import sys

def retarget(data):
    original = b'd3d12core.dll\0'
    private = b'd4r12core.dll\0'
    if not data.startswith(b'MZ') or data.count(original) != 2:
        raise ValueError('Expected the d4r-built vkd3d forwarder with two core-loader strings')
    result = data.replace(original, private)
    assert len(result) == len(data)
    return result

if __name__ == '__main__':
    try:
        Path(sys.argv[2]).write_bytes(retarget(Path(sys.argv[1]).read_bytes()))
    except (ValueError, OSError) as error:
        sys.exit(str(error))
