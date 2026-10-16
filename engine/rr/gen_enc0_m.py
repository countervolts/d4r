#!/usr/bin/env python3
"""Generate engine/rr/enc0_m.comp's source: encoder 0 with ptx2rr.py's translation of NVIDIA's per-pixel feature gather
(run by wave 0, as NVIDIA runs it in one warp) and the native rest (engine/rr/enc0_m_decl.glsl, enc0_m_tail.glsl).

usage: gen_enc0_m.py ENC0.generated.comp OUT.comp

The translation's main is cut at its first workgroup barrier: by then the 10 x 10 pixels' features and the noise grid
are in its shared buffer sh and the state surfaces 304 / 1016 are written. Model data derived from the user's DLL is in
the translated part; do not commit the output.
"""
import re
import sys


def main():
    src, out = sys.argv[1:3]
    lines = open(src).read().split('\n')
    m = next(i for i, l in enumerate(lines) if l.startswith('void main()'))
    head = '\n'.join(lines[:m]).replace('layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;',
                                         'layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;')
    body = lines[m + 2:]
    first = next(i for i, l in enumerate(body) if not re.match(r'    (precise |bool |uint64_t |float |uint |int )', l))
    bar = next(i for i, l in enumerate(body) if l.strip() == 'memoryBarrierShared(); barrier();')
    decls, prologue = body[:first], body[first:bar]
    # NVIDIA's warp gathers the 96 key pixels in a loop of three 32-pixel steps, then the four corners: the steps go to
    # waves 0-2 (the setup before the loop runs in every wave, the corners in wave 0)
    do_ = next(i for i, l in enumerate(prologue) if l.strip() == 'do')
    end = next(i for i, l in enumerate(prologue) if l.strip().startswith('} while (again_'))
    loop = '\n'.join(prologue[do_ - 1:end + 1])
    inc = re.search(r'(r\d+) = uint\((r\d+) \+ 32u\);\n\s+(p\d+) = \(\1 < 96u\);', loop)
    counter = inc.group(1)
    loop = loop.replace(inc.group(0), inc.group(0).replace('+ 32u', '+ 96u'))
    pre = [l.replace(f'{counter} = uint(0u);', f'{counter} = uint(32u * gl_SubgroupID);') for l in prologue[:do_ - 1]]
    post = prologue[end + 1:]
    head = head.replace('shared uint mms[384];', 'shared uint mms[384 * 4];')
    text = (head + '\n#extension GL_GOOGLE_include_directive : require\n#include "enc0_m_decl.glsl"\n\nvoid main()\n{\n' +
            '\n'.join(decls) + '\n    // the translated feature gather (NVIDIA\'s one warp, its pixel loop split over the waves)\n' +
            '\n'.join(pre) + '\n    if (gl_SubgroupID < 3u) {\n' + loop + '\n    }\n    if (gl_SubgroupID == 0u) {\n' +
            '\n'.join(post) + '\n    }\n#include "enc0_m_tail.glsl"\n}\n')
    open(out, 'w').write(text)
    print(f'{out}: {len(decls)} declaration lines, {len(prologue)} prologue lines')


if __name__ == '__main__':
    main()
