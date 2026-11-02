"""The tests' shell steps on Linux and on native Windows (Git Bash or MSYS2, with MinGW-w64 compilers on PATH).

On Windows, subprocess.run(['bash', ...]) searches System32 before PATH and starts WSL's bash, which sees none of
the repository's paths; BASH is the bash on PATH instead. Paths passed to it are written with forward slashes
(C:/...), which Git Bash and MSYS2 accept and which no shell step mistakes for escapes.
"""
import glob
import os
import shutil
import subprocess
from pathlib import Path, PurePath

WINDOWS = os.name == 'nt'
EXE = '.exe' if WINDOWS else ''


def _find_bash():
    if not WINDOWS:
        return shutil.which('bash')
    system = Path(os.environ.get('SystemRoot', r'C:\Windows')).resolve()
    for directory in os.environ.get('PATH', '').split(os.pathsep):
        candidate = Path(directory) / 'bash.exe'
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        # skip WSL's launchers (System32, WindowsApps)
        if candidate.is_file() and system not in resolved.parents and 'WindowsApps' not in resolved.parts:
            return str(candidate)
    for fallback in (r'C:\Program Files\Git\bin\bash.exe', r'C:\msys64\usr\bin\bash.exe'):
        if Path(fallback).is_file():
            return fallback
    return None


BASH = _find_bash()


def shell_path(value):
    """A path as the shell should see it; other arguments unchanged."""
    if isinstance(value, PurePath):
        return value.as_posix() if WINDOWS else str(value)
    if WINDOWS and isinstance(value, str) and len(value) > 2 and value[1] == ':' and value[2] in '\\/':
        return value.replace('\\', '/')
    return value


def bash(*args):
    """The argument list that runs bash with ARGS (a script and its arguments, or -c and a command)."""
    return [BASH or 'bash', *(shell_path(a) for a in args)]


def build_elf_library(output, source):
    """Links C SOURCE into a Linux x86-64 shared library at OUTPUT -> True, or False without a suitable compiler.
    The host cc on Linux; on Windows (where cc makes a DLL) a clang with lld, such as the HIP SDK's, cross-links."""
    if not WINDOWS:
        subprocess.run(['cc', '-shared', '-x', 'c', '-o', str(output), '-'], input=source, text=True, check=True)
        return True
    candidates = [shutil.which('clang')]
    if os.environ.get('D4R_ROCM_DIR'):
        candidates.append(str(Path(os.environ['D4R_ROCM_DIR']) / 'lib/llvm/bin/clang.exe'))
    candidates += sorted(glob.glob(r'C:\ROCm\*\lib\llvm\bin\clang.exe'), reverse=True)
    for clang in filter(None, candidates):
        result = subprocess.run([clang, '--target=x86_64-linux-gnu', '-shared', '-nostdlib', '-fuse-ld=lld', '-x', 'c',
                                 '-o', str(output), '-'], input=source, text=True, capture_output=True)
        if result.returncode == 0:
            return True
    return False


def vulkan_flags():
    """Compiler arguments that find and link the Vulkan loader: -lvulkan on Linux, the Vulkan SDK's vulkan-1 on
    Windows (VULKAN_SDK)."""
    if not WINDOWS:
        return ['-lvulkan']
    sdk = Path(os.environ.get('VULKAN_SDK', ''))
    return ['-I' + str(sdk / 'Include'), '-L' + str(sdk / 'Lib'), '-lvulkan-1']
