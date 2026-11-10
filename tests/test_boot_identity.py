"""d4r_boot.dll (the in-process NVIDIA identity) and the PE import injection that loads it before OptiScaler."""
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from pe_add_import import add_import, import_names  # noqa: E402

MINGW = shutil.which('x86_64-w64-mingw32-gcc')


def imports(path):
    out = subprocess.run(['x86_64-w64-mingw32-objdump', '-p', str(path)], text=True, capture_output=True,
                         check=True).stdout
    return re.findall(r'DLL Name: (\S+)', out)


@unittest.skipUnless(MINGW, 'x86_64-w64-mingw32-gcc not installed')
class BootIdentityTests(unittest.TestCase):
    def test_import_is_first_and_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            dll = Path(directory) / 'opti.dll'
            subprocess.run([MINGW, '-shared', '-x', 'c', '-o', str(dll), '-'], check=True, text=True,
                           input='#include <windows.h>\n__declspec(dllexport) int f(void) '
                                 '{ return (int)(INT_PTR)GetModuleHandleA(0); }\n')
            before = imports(dll)
            data, changed = add_import(dll.read_bytes(), 'd4r_boot.dll', 'd4r_boot')
            self.assertTrue(changed)
            out = Path(directory) / 'dxgi.dll'
            out.write_bytes(data)
            self.assertEqual(import_names(data), ['d4r_boot.dll'] + before)
            again, changed = add_import(data, 'd4r_boot.dll', 'd4r_boot')
            self.assertFalse(changed)
            self.assertEqual(again, data)

    def test_not_a_pe(self):
        with self.assertRaises(ValueError):
            add_import(b'fixture', 'd4r_boot.dll', 'd4r_boot')

    def test_boot_dll_builds_and_exports_its_import(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'd4r_boot.dll'
            subprocess.run([MINGW, '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror', '-shared', '-municode',
                            str(ROOT / 'tools/d4r_boot.c'), '-o', str(out)], check=True)
            dump = subprocess.run(['x86_64-w64-mingw32-objdump', '-p', str(out)], text=True, capture_output=True,
                                  check=True).stdout
            self.assertRegex(dump, r'(?m)\+base\[\s*1\]\s+\w+\s+d4r_boot$')
            # only modules every Windows/Wine process has
            self.assertLessEqual({n.lower() for n in imports(out)} - {'kernel32.dll', 'user32.dll'},
                                 {n.lower() for n in imports(out) if n.lower().startswith('api-ms-win-crt-')})

    def test_release_ini_documents_identity(self):
        text = (ROOT / 'packaging/d4r.ini').read_text()
        self.assertIn('[Identity]', text)
        self.assertRegex(text, r'(?m)^SpoofGpu = auto$')
        self.assertRegex(text, r'(?m)^GpuArch = auto$')


if __name__ == '__main__':
    unittest.main()
