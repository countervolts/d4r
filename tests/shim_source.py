"""The shim's sources as one string, for the tests that read the shim's text.

tools/d4r_nvngx_shim.cpp is split into several translation units that share
tools/d4r_nvngx_shim_internal.h, so a snippet test must search all of them. The
units are concatenated in the order the one big file used to cover: the shared
header first, then the units in the order their code originally appeared.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SHIM_UNITS = (
    'tools/d4r_nvngx_shim_internal.h',
    'tools/d4r_nvngx_shim_config.cpp',
    'tools/d4r_nvngx_shim_aux.cpp',
    'tools/d4r_nvngx_shim_vram.cpp',
    'tools/d4r_nvngx_shim_split.cpp',
    'tools/d4r_nvngx_shim_pipeline.cpp',
    'tools/d4r_nvngx_shim_native.cpp',
    'tools/d4r_nvngx_shim_native_d3d12.cpp',
    'tools/d4r_nvngx_shim.cpp',   # the exported entry points, the DLL entry and the backends' .inl files
)


def shim_files(root=ROOT):
    """The units that exist, in order; also picks up units a later split adds."""
    files = [Path(root) / name for name in SHIM_UNITS]
    files += sorted(p for p in (Path(root) / 'tools').glob('d4r_nvngx_shim_*.cpp')
                    if p not in files)
    return files


def shim_source(root=ROOT):
    """Every shim unit, concatenated, with blank lines between them."""
    return '\n\n'.join(p.read_text() for p in shim_files(root))


def definition_body(source, signature):
    """One function definition, from `signature` to its closing brace.

    A split keeps a helper next to the code that calls it in its own unit, so a test that
    compiles a helper out of the shim asks for its whole definition, not for the text up to
    some comment that used to sit below it in the one big file. The shared header is the one
    declaration that carries a parameter's default value, so the definition comes back with
    that default written back in: a test compiles the definition on its own, with no header.
    """
    at = definition(source, signature)
    i = at + len(signature) - 1            # the '(' the signature ends with
    depth = 0
    while i < len(source):
        depth += (source[i] == '(') - (source[i] == ')')
        if depth == 0:
            break
        i += 1
    head, close = source[at:i + 1], i
    first = source.find(signature)
    if first != at and '=' not in head:
        end = source.find(';', first)
        if end != -1 and '=' in source[first:end]:
            head = source[first:end]       # the prototype, its ';' traded for the body
    j = source.index('{', close)
    depth = 0
    while j < len(source):
        depth += (source[j] == '{') - (source[j] == '}')
        if depth == 0:
            return head + source[close + 1:j + 1]
        j += 1
    raise ValueError('unbalanced braces in ' + signature)


def definition(source, signature, start=0):
    """Where `signature` (say 'bool record_feature_vram_output(') defines its function.

    The units share tools/d4r_nvngx_shim_internal.h, which prototypes them all, so a plain
    search often lands on the prototype; a definition is the one whose parameter list is
    followed by its opening brace.
    """
    at = start
    while True:
        at = source.index(signature, at)
        i = at + len(signature) - 1            # the '(' the signature ends with
        depth = 0
        while i < len(source):
            depth += (source[i] == '(') - (source[i] == ')')
            if depth == 0:
                break
            i += 1
        j = i + 1
        while j < len(source) and source[j] in ' \n':
            j += 1
        if j < len(source) and source[j] == '{':
            return at
        at += 1
