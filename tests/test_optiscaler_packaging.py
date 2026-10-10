"""Release NGX routing: real INI generation, full/clean ZIP staging, install diagnostics.

Package tests use disposable runtime files and stand-ins for GPU kernel compilation.
Actual OptiScaler/Proton loading additionally needs the D3D12 harness.
"""
import configparser
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = ROOT / 'packaging/optiscaler.settings'
GENERATOR = ROOT / 'scripts/configure_optiscaler.py'


def upstream_ini():
    sections = {}
    for line in SETTINGS.read_text().splitlines():
        if not line or line.startswith('#'):
            continue
        path, _ = line.split('=', 1)
        section, key = path.split('.', 1)
        sections.setdefault(section, []).append(key)
    # Keep NvngxPath independent of the settings file so removing the release
    # override reproduces the original bug rather than removing the fixture key.
    if 'NvngxPath' not in sections['Libraries']:
        sections['Libraries'].append('NvngxPath')
    return '; upstream comment\n' + ''.join(
        f'[{section}]\n' + ''.join(f'{key}=auto\n' for key in keys)
        for section, keys in sections.items()) + '[Menu]\nShowFps=true\n'


class OptiScalerConfigTests(unittest.TestCase):
    def generate(self, directory, source):
        source_path = directory / 'upstream.ini'
        output = directory / 'OptiScaler.ini'
        source_path.write_bytes(source.encode())
        result = subprocess.run(['python3', str(GENERATOR), str(source_path), str(SETTINGS), str(output)],
                                capture_output=True, text=True)
        return result, output

    def assert_routing(self, data):
        ini = configparser.ConfigParser()
        ini.read_string(data.decode())
        self.assertEqual(ini['Libraries']['NvngxPath'], r'd4r\nvngx.dll')
        self.assertEqual(ini['Libraries']['OptiDllPath'], 'd4r')
        self.assertEqual(ini['Upscalers']['Dx12Upscaler'], 'dlss')
        self.assertEqual(ini['DLSS']['Enabled'], 'true')
        self.assertEqual(ini['Hooks']['HookOriginalNvngxOnly'], 'true')
        self.assertEqual(ini['Menu']['ShowFps'], 'true')

    def test_generated_config_and_line_endings(self):
        for newline in ('\n', '\r\n'):
            with self.subTest(newline=repr(newline)), tempfile.TemporaryDirectory() as temporary:
                source = upstream_ini().replace('\n', newline)
                result, output = self.generate(Path(temporary), source)
                self.assertEqual(result.returncode, 0, result.stderr)
                data = output.read_bytes()
                self.assert_routing(data)
                self.assertTrue(data.startswith(b'; upstream comment' + newline.encode()))
                self.assertEqual(data.count(b'\r\n'), data.count(b'\n') if newline == '\r\n' else 0)
                self.assertEqual(source.count(newline), data.count(newline.encode()))

    def test_missing_or_duplicate_override_fails_without_output(self):
        for replacement in ('', 'NvngxPath=auto\nnvngxpath=d4r\\ngx\\_nvngx.dll\n'):
            with self.subTest(replacement=replacement), tempfile.TemporaryDirectory() as temporary:
                result, output = self.generate(Path(temporary), upstream_ini().replace('NvngxPath=auto\n', replacement))
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('exactly one NvngxPath', result.stderr)
                self.assertFalse(output.exists())

    def test_missing_or_duplicate_libraries_section_fails(self):
        for source in (upstream_ini().replace('[Libraries]', '[Other]'),
                       upstream_ini() + '[libraries]\nNvngxPath=auto\n'):
            with tempfile.TemporaryDirectory() as temporary:
                result, output = self.generate(Path(temporary), source)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('exactly one [Libraries]', result.stderr)
                self.assertFalse(output.exists())

    def test_full_and_clean_release_archives(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary) / 'repo'
            inputs = Path(temporary) / 'inputs'
            inputs.mkdir()
            (repo / 'scripts').mkdir(parents=True)
            for name in ('package_release.sh', 'configure_optiscaler.py', 'check_glibc_compat.sh'):
                shutil.copy2(ROOT / 'scripts' / name, repo / 'scripts' / name)
            shutil.copytree(ROOT / 'packaging', repo / 'packaging')
            for name in ('LICENSE', 'NOTICE'):
                shutil.copy2(ROOT / name, repo / name)
            for component in ('zluda', 'vkd3d-proton'):
                (repo / 'patches' / component).mkdir(parents=True)
                (repo / 'patches' / component / 'fixture.patch').write_text('fixture\n')
            # All shell staging, INI generation, ABI checks and archiving are real;
            # replace only expensive GPU builds/manifests with fixture artifacts.
            (repo / 'kernels/tools').mkdir(parents=True)
            kernel_build = repo / 'kernels/build.sh'
            kernel_build.write_text('#!/bin/sh\nmkdir -p "$2"\nprintf fixture > "$2/layer.hsaco"\n')
            kernel_build.chmod(0o755)
            (repo / 'kernels/tools/kernel_manifest.py').write_text(
                'import pathlib, sys\npathlib.Path(sys.argv[1], "d4r-kernels.txt").write_text("layer 1234\\n")\n')

            def fixture(path, data=b'fixture'):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                return path

            # A real minimal ELF lets the package's existing GLIBC checker run.
            elf = inputs / 'library.so'
            subprocess.run(['cc', '-shared', '-x', 'c', '-o', str(elf), '-'],
                           input='int fixture(void) { return 0; }\n', text=True, check=True)
            fixture(repo / 'build/d4r_nvngx.dll')
            fixture(repo / 'build/wine-nvcuda/x86_64-unix/nvcuda.dll.so', elf.read_bytes())
            fixture(inputs / 'zluda/libnvcuda.so', elf.read_bytes())
            fixture(inputs / 'rocm/lib/libamdhip64.so.7', elf.read_bytes())
            fixture(inputs / 'rocm/licenses/runtime.txt')
            for name in ('d3d12.dll', 'd3d12core.dll'):
                fixture(inputs / 'vkd3d' / name)
            fixture(inputs / 'opti/OptiScaler.dll')
            fixture(inputs / 'opti/OptiScaler.ini', upstream_ini().replace('\n', '\r\n').encode())
            dlss = fixture(inputs / 'nvngx_dlss.dll')
            core = fixture(inputs / '_nvngx.dll')
            license_path = fixture(inputs / 'license.txt')
            for name in ('LICENSE-APACHE', 'LICENSE-MIT', 'ext/llvm-project/llvm/LICENSE.TXT', 'LICENSE', 'COPYING'):
                fixture(inputs / 'source' / name)
            # Prebuilt L texture names keep the full flow from calling an emitter.
            names = ['rrlite_dec0_4x4']
            for mv in ('mvhi', 'mvlo'):
                for dynamic_range in ('hdr', 'ldr'):
                    names.append(f'rrlite_enc0_4x4_{mv}_{dynamic_range}')
                    names.extend(f'rrlite_post_{v}_{mv}_{dynamic_range}' for v in ('3_1', '3_2'))
            for prefix in ('gfx1101', 'accuracy/gfx1101'):
                for name in names:
                    fixture(inputs / 'tex' / prefix / (name + '.hsaco'))
                fixture(inputs / 'tex' / prefix / 'd4r-accuracy.txt', b'1\n')
            env = {k: v for k, v in os.environ.items() if not k.startswith('D4R_')}
            env.update(D4R_SKIP_BUILD='1', D4R_GPU_ARCHS='gfx1101', SOURCE_DATE_EPOCH='1700000000',
                       D4R_OPTISCALER=str(inputs / 'opti'), D4R_DLSS_DLLS=str(dlss),
                       D4R_ZLUDA_DIR=str(inputs / 'zluda'), D4R_VKD3D_DIR=str(inputs / 'vkd3d'),
                       D4R_ROCM_RUNTIME=str(inputs / 'rocm'), D4R_OPTISCALER_LICENSE=str(license_path),
                       D4R_ZLUDA_SRC=str(inputs / 'source'), D4R_VKD3D_SRC=str(inputs / 'source'),
                       D4R_BUNDLE_DLSS=str(dlss), D4R_BUNDLE_NGX=str(core), D4R_BUNDLE_TEX=str(inputs / 'tex'))
            version = (repo / 'packaging/VERSION').read_text().strip()
            configs = []
            # A compiled engine model: only d4r's shaders and marker may reach a zip.
            for name in ('input_k.spv', 'output_k.spv', 'exposure_k0.spv', 'exposure_k1.spv', 'enc0.spv',
                         'direct_origins.bin', 'weights.bin', 'offsets.bin', 'lut.bin', 'hipnet.bin', 'model.json'):
                fixture(inputs / 'engine' / name)
            env['D4R_BUNDLE_ENGINE'] = str(inputs / 'engine')
            for bundled in ('1', '0'):
                with self.subTest(bundled=bundled):
                    env['D4R_BUNDLE_NVIDIA'] = bundled
                    result = subprocess.run(['bash', str(repo / 'scripts/package_release.sh'), str(repo / 'dist')],
                                            env=env, text=True, capture_output=True)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    name = f'd4r-{version}' + ('-nonvidia' if bundled == '0' else '')
                    with zipfile.ZipFile(repo / 'dist' / (name + '.zip')) as archive:
                        self.assertIsNone(archive.testzip())
                        data = archive.read('OptiScaler.ini')
                        self.assert_routing(data)
                        configs.append(data)
                        self.assertEqual(data, (repo / 'dist' / name / 'OptiScaler.ini').read_bytes())
                        self.assertIn('d4r/nvngx.dll', archive.namelist())
                        self.assertNotIn('nvngx.dll', archive.namelist())
                        self.assertNotIn('d4r/_nvngx.dll', archive.namelist())
                        self.assertEqual('d4r/ngx/_nvngx.dll' in archive.namelist(), bundled == '1')
                        self.assertEqual('d4r/nvngx_dlss.dll' in archive.namelist(), bundled == '1')
                        engine = sorted(n for n in archive.namelist() if n.startswith('d4r/engine/'))
                        self.assertEqual(engine, ['d4r/engine/k/' + n for n in (
                            'README.txt', 'direct_origins.bin', 'enc0.spv', 'exposure_k0.spv', 'exposure_k1.spv',
                            'input_k.spv', 'output_k.spv')])
                        self.assertEqual(archive.read('d4r/d4r-check.sh'), (ROOT / 'packaging/d4r-check.sh').read_bytes())
            self.assertEqual(configs[0], configs[1])


class InstallCheckTests(unittest.TestCase):
    def test_routing_separate_from_file_presence(self):
        with tempfile.TemporaryDirectory() as temporary:
            game = Path(temporary) / 'game with spaces'
            for name in ('Game.exe', 'dxgi.dll', 'd3d12.dll', 'd3d12core.dll', 'd4r/nvngx.dll',
                         'd4r/nvcuda.dll', 'd4r/zluda/libcuda.so', 'd4r/d4r.ini', 'd4r/nvngx_dlss.dll',
                         'd4r/ngx/_nvngx.dll', 'd4r/rocm/lib/libamdhip64.so.7'):
                path = game / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            for value, good in ((r'd4r\nvngx.dll', True), ('./d4r/nvngx.dll', True),
                                ('Z:' + str(game / 'd4r/nvngx.dll').replace('/', '\\'), True),
                                ('auto', False), ('d4r', False), (r'd4r\ngx\_nvngx.dll', False),
                                (r'C:\windows\system32\_nvngx.dll', False), ('', False)):
                with self.subTest(value=value):
                    setting = f'NvngxPath={value}\n' if value else ''
                    (game / 'OptiScaler.ini').write_bytes(
                        ('[Libraries]\nOptiDllPath=d4r\n' + setting).replace('\n', '\r\n').encode())
                    result = subprocess.run(['sh', str(ROOT / 'packaging/d4r-check.sh'), str(game)],
                                            text=True, capture_output=True)
                    self.assertIn('ok       d4r/nvngx.dll', result.stdout)
                    if good:
                        self.assertIn('OptiScaler NGX routing: NvngxPath=', result.stdout)
                        self.assertNotIn('CONFIG', result.stdout)
                    else:
                        self.assertIn('CONFIG', result.stdout)
                        self.assertIn('OptiDllPath alone can load system _nvngx.dll', result.stdout)
                        self.assertNotIn('checks passed', result.stdout)
                        self.assertNotEqual(result.returncode, 0)

    def test_section_and_duplicate_keys(self):
        for ini in ('[Other]\nNvngxPath=d4r\\nvngx.dll\n[Libraries]\nOptiDllPath=d4r\n',
                    '[Libraries]\nNvngxPath=d4r\\nvngx.dll\nnvngxpath=auto\n'):
            with tempfile.TemporaryDirectory() as temporary:
                game = Path(temporary)
                (game / 'd4r').mkdir()
                (game / 'OptiScaler.ini').write_text(ini)
                result = subprocess.run(['sh', str(ROOT / 'packaging/d4r-check.sh'), temporary],
                                        capture_output=True, text=True)
                self.assertIn('CONFIG', result.stdout)
                self.assertNotEqual(result.returncode, 0)

    def test_correct_routing_does_not_hide_missing_shim(self):
        with tempfile.TemporaryDirectory() as temporary:
            game = Path(temporary)
            (game / 'd4r').mkdir()
            (game / 'OptiScaler.ini').write_text('[Libraries]\nNvngxPath=d4r\\nvngx.dll\n')
            result = subprocess.run(['sh', str(ROOT / 'packaging/d4r-check.sh'), temporary],
                                    capture_output=True, text=True)
            self.assertIn('MISSING  d4r/nvngx.dll', result.stdout)
            self.assertIn('OptiScaler NGX routing: NvngxPath=', result.stdout)
            self.assertNotIn('checks passed', result.stdout)
            self.assertNotEqual(result.returncode, 0)


if __name__ == '__main__':
    unittest.main()
