"""CPU regression checks for the upstream OptiScaler coexistence patch."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / 'patches/optiscaler/0001-proton-external-dlss-fsr4-coexistence.patch'


class OptiScalerCoexistenceTests(unittest.TestCase):
    def test_backend_identity_policy(self):
        patch = PATCH.read_text()
        start = patch.index('diff --git a/OptiScaler/spoofing/BackendIdentity.h ')
        section = patch[start:].split('\ndiff --git ', 1)[0]
        # Compile the exact added header that an upstream build will consume.
        header = '\n'.join(line[1:] for line in section.splitlines()
                           if line.startswith('+') and not line.startswith('+++'))
        code = r'''
#include "BackendIdentity.h"
#include <cassert>
int main() {
    for (bool nvidia : {false, true})
        for (bool optedIn : {false, true})
            for (bool file : {false, true})
                assert(BackendIdentity::AllowDlss(nvidia, optedIn, file) ==
                       (nvidia || (optedIn && file)));
    const char* amd[] = {"amdxc64.dll", "AMDXCFFX64.DLL", "amd_fidelityfx_dx12.dll",
        "amd_fidelityfx_loader_dx12.dll", "amd_fidelityfx_upscaler_dx12.dll",
        "amd_fidelityfx_framegeneration_dx12.dll", "amd_fidelityfx_denoiser_dx12.dll",
        "amd_fidelityfx_radiancecache_dx12.dll"};
    for (auto caller : amd) assert(BackendIdentity::IsAmdFsrCaller(caller));
    const char* game[] = {"Game.exe", "Game-Win64-Shipping.exe", "nvngx.dll",
        "nvngx_dlss.dll", "notamdxc64.dll", "amdxc64.dll.exe", "", "AMDXC64"};
    for (auto caller : game) assert(!BackendIdentity::IsAmdFsrCaller(caller));
}
'''
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / 'BackendIdentity.h').write_text(header)
            runner = directory / 'identity'
            subprocess.run(['c++', '-std=c++17', '-Wall', '-Wextra', '-Werror', '-I', temporary,
                            '-x', 'c++', '-', '-o', str(runner)], input=code,
                           text=True, capture_output=True, check=True)
            subprocess.run([str(runner)], check=True)

    def test_upstream_integration_preserves_hardware_identity(self):
        additions = '\n'.join(line[1:] for line in PATCH.read_text().splitlines()
                              if line.startswith('+') and not line.startswith('+++'))
        self.assertEqual(additions.count('BackendIdentity::IsAmdFsrCaller(caller)'), 4)
        self.assertIn('DLSSAllowExternalBackend { false }', additions)
        self.assertIn('readBool("DLSS", "AllowExternalBackend")', additions)
        self.assertIn('ini.SetValue("DLSS", "AllowExternalBackend"', additions)
        self.assertIn('std::filesystem::is_regular_file(backendPath, error)', additions)
        self.assertIn('backendPath = Util::ExePath().parent_path() / backendPath', additions)
        self.assertIn('NvngxPath.set_volatile_value(backendPath.wstring())', additions)
        self.assertIn('if (State::Instance().isRunningOnNvidia && !Config::Instance()->DxgiSpoofing.has_value())', additions)
        self.assertNotIn('isRunningOnNvidia = true', additions)
        self.assertNotIn('isRunningOnRDNA4 = true', additions)

    def test_checker_launch_options_follow_external_backend_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            game = Path(temporary)
            (game / 'd4r').mkdir()
            for flag in ('true', 'false', 'auto'):
                (game / 'OptiScaler.ini').write_bytes(
                    (f'[DLSS]\r\nAllowExternalBackend={flag}\r\n'
                     '[Libraries]\r\nNvngxPath=d4r\\nvngx.dll\r\n').encode())
                result = subprocess.run(['sh', str(ROOT / 'packaging/d4r-check.sh'), temporary],
                                        text=True, capture_output=True)
                if flag == 'true':
                    self.assertIn('WINE_HIDE_AMD_GPU=0 %command%', result.stdout)
                    self.assertNotIn('PROTON_FORCE_NVAPI=1', result.stdout)
                    self.assertNotIn('requires the game\'s working directory', result.stdout)
                else:
                    self.assertIn('PROTON_FORCE_NVAPI=1', result.stdout)


if __name__ == '__main__':
    unittest.main()
