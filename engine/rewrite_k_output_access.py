#!/usr/bin/env python3
"""Specialize the 32-bit-address PTX-to-GLSL K reconstruction's read-only accesses.

Run after address narrowing and before glslang/nocontract. The input must be a locally
generated K output shader, with head base zero and the LUT tagged at bit 31. No NVIDIA
source or model data is included here. The descriptor layout and bytes stay unchanged.
LUT records are 16 bytes, as in output_k.comp; all remaining G16/G32 uses read the head.
Unsupported generator layouts fail rather than leaving a partly rewritten shader.
"""
import argparse
import re
from pathlib import Path


def rewrite(source: str) -> str:
    declarations = 'buffer LutBits { uint16_t bits[]; }'
    scalar = ('return a>=0x80000000u ? uint(lut.bits[(a-0x80000000u)>>1]) '
              ': uint(head.bits[a>>1]);')
    if source.count(declarations) != 1 or source.count(scalar) != 1:
        raise ValueError('expected the 32-bit-address K head/LUT buffer layout')
    param = re.search(r'uint PARAM64\(uint o\)\s*\{(.*?)\}', source, re.S)
    body = re.sub(r'//[^\n]*|/\*.*?\*/', '', param[1], flags=re.S) if param else ''
    body = re.sub(r'\s+', '', body)
    if body != 'if(o==280u)return0x80000000u+((param.w[36]>>8)&15u)*32768u;return0u;':
        raise ValueError('expected zero-based head addresses from PARAM64')

    # Union all definitions, including branches and loop-carried address updates.
    # Index-only arithmetic carries no tag; actual base addresses must trace to
    # PARAM64(272), never to the tagged PARAM64(280) table.
    expressions = {}
    for name, expression in re.findall(r'^\s+(rd\d+) = uint\((.*?)\);', source, re.M):
        expressions.setdefault(name, []).append(expression)
    roots = {name: set() for name in expressions}
    for _ in range(len(roots) + 1):
        changed = False
        for name, definitions in expressions.items():
            previous = set(roots[name])
            for expression in definitions:
                roots[name].update(re.findall(r'PARAM64\((\d+)u\)', expression))
                for dependency in re.findall(r'\brd\d+\b', expression):
                    roots[name].update(roots.get(dependency, set()))
            changed |= previous != roots[name]
        if not changed:
            break
    pattern = re.compile(
        r'(?P<indent> +)r(?P<a>\d+) = uint\(G32\((?P<base>rd\d+) \+ uint\(0u\)\)\);\n'
        r' +r(?P<b>\d+) = uint\(G32\((?P=base) \+ uint\(4u\)\)\);\n'
        r' +r(?P<c>\d+) = uint\(G32\((?P=base) \+ uint\(8u\)\)\);\n'
        r' +r(?P<d>\d+) = uint\(G32\((?P=base) \+ uint\(12u\)\)\);')

    def vector_record(match: re.Match) -> str:
        base = match['base']
        definitions = re.findall(r'\b' + base + r' = uint\((rd\d+) \+ (rd\d+)\);', source)
        if len(definitions) != 1:
            raise ValueError(f'ambiguous LUT address {base}')
        tag, index = definitions[0]
        if f'{tag} = uint(PARAM64(280u));' not in source:
            raise ValueError(f'{base} is not based on the reconstruction LUT')
        if not re.search(r'\b' + index + r' = uint\(uint\(int\(int\(r\d+\)\) \* int\(16\)\)\);', source):
            raise ValueError(f'{base} is not a 16-byte-aligned LUT record')
        indent = match['indent']
        rows = [indent + '{ uvec4 block = lut.blocks[(' + base + '-0x80000000u)>>4];']
        rows += [indent + '  r' + match[k] + ' = block.' + xy + ';'
                 for k, xy in zip('abcd', 'xyzw')]
        return '\n'.join(rows + [indent + '}'])

    result, count = pattern.subn(vector_record, source)
    if count != 2 or result.count('= uint(G32(') != 2:
        raise ValueError('expected two LUT records and two remaining head-word reads')
    for base in re.findall(r'G(?:16|32)\((rd\d+) \+ uint\(\d+u\)\)', result):
        if roots.get(base) != {'272'}:
            raise ValueError(f'non-LUT access {base} does not trace exclusively to the head')
    # PARAM64(272) is the zero head base. The address-narrowing pass proves the
    # head offsets below bit 31; the only tagged accesses are the two records above.
    result = result.replace(declarations, 'buffer LutBits { uvec4 blocks[]; }')
    result = result.replace(scalar, 'return uint(head.bits[a>>1]); // LUT records specialized below')
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.write_text(rewrite(args.source.read_text()))


if __name__ == '__main__':
    main()
