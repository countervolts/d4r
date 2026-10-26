"""Release NGX routing: real INI generation, full/clean ZIP staging, install diagnostics.

Package tests use disposable runtime files and stand-ins for GPU kernel compilation.
Actual OptiScaler/Proton loading additionally needs the D3D12 harness.
"""
import configparser
import hashlib
import itertools
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
        self.assertEqual(ini['Libraries']['OptiDllPath'], 'OptiScaler')
        self.assertEqual(ini['Upscalers']['Dx12Upscaler'], 'dlss')
        self.assertEqual(ini['DLSS']['Enabled'], 'true')
        self.assertEqual(ini['Hooks']['HookOriginalNvngxOnly'], 'true')
        self.assertEqual(ini['Menu']['ShowFps'], 'true')
        self.assertEqual(ini['fakenvapi']['UseFakenvapi'], 'false')

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
            for name in ('package_release.sh', 'configure_optiscaler.py', 'stage_optiscaler.py', 'check_glibc_compat.sh',
                         'check_redistributable.py', 'private_d3d12_core.py', 'stage_engine_kit.sh'):
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

            # Exercise the actual installer-kit stager with representative source files.
            # No model builder, shader compiler or GPU fixture is executed here.
            for name in ('engine/compile_k.py', 'engine/test_k.py', 'engine/test_m.py',
                         'engine/rr/test_layer.py', 'vulkan/ptx2glsl.py'):
                destination = repo / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / name, destination)
            fixture(repo / 'engine/weights.bin', b'must not enter the model kit')
            fixture(repo / 'engine/test_unrelated.py', b'must not enter the model kit')
            fixture(repo / 'build/d3d12_dlss_harness.exe')
            validator = fixture(inputs / 'glslang/bin/glslangValidator', b'packaging-only compiler fixture')

            # A real minimal ELF lets the package's existing GLIBC checker run.
            elf = inputs / 'library.so'
            subprocess.run(['cc', '-shared', '-x', 'c', '-o', str(elf), '-'],
                           input='int fixture(void) { return 0; }\n', text=True, check=True)
            fixture(repo / 'build/d4r_nvngx.dll')
            fixture(repo / 'build/d4r-setup.exe')
            fixture(repo / 'build/d4r_nvapi64.dll')
            fixture(repo / 'build/wine-nvcuda/x86_64-unix/nvcuda.dll.so', elf.read_bytes())
            fixture(inputs / 'zluda/libnvcuda.so', elf.read_bytes())
            fixture(inputs / 'rocm/lib/libamdhip64.so.7', elf.read_bytes())
            fixture(inputs / 'rocm/licenses/runtime.txt')
            for name in ('d3d12.dll', 'd3d12core.dll'):
                fixture(inputs / 'vkd3d' / name, b'MZ fixture d3d12core.dll\0 then d3d12core.dll\0' if name == 'd3d12.dll' else b'fixture')
            fixture(inputs / 'opti/OptiScaler.dll')
            for name in ('OptiScaler/amd_fidelityfx_loader_dx12.dll', 'OptiScaler/amd_fidelityfx_upscaler_dx12.dll',
                         'Licenses/FidelityFX_v2_LICENSE.md', 'Licenses/XeSS_LICENSE.txt',
                         'OptiScaler/libxell.dll', 'OptiScaler/libxess_fg.dll', 'OptiScaler/libxess.dll',
                         'OptiScaler/dlssg_to_fsr3_amd_is_better.dll',
                         'OptiScaler/D3D12_OptiScaler/D3D12Core.dll',
                         'OptiScaler/amd_fidelityfx_framegeneration_dx12.dll'):
                fixture(inputs / 'opti' / name)
            fixture(inputs / 'opti/OptiScaler.ini', upstream_ini().replace('\n', '\r\n').encode())
            dlss = fixture(inputs / 'nvngx_dlss.dll')
            core = fixture(inputs / '_nvngx.dll')
            installer = (ROOT / 'install.sh').read_text()
            installer = '\n'.join('DLSS_SHA256=' + hashlib.sha256(dlss.read_bytes()).hexdigest()
                                  if line.startswith('DLSS_SHA256=') else line
                                  for line in installer.splitlines()) + '\n'
            (repo / 'install.sh').write_text(installer)
            shutil.copy2(ROOT / 'uninstall.sh', repo / 'uninstall.sh')
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
                       D4R_BUNDLE_DLSS=str(dlss), D4R_BUNDLE_NGX=str(core), D4R_BUNDLE_TEX=str(inputs / 'tex'),
                       D4R_GLSLANG=str(inputs / 'glslang'), D4R_GLSLANG_LICENSE=str(license_path))
            version = (repo / 'packaging/VERSION').read_text().strip()
            configs = []
            # A compiled engine model: only d4r's shaders and marker may reach a zip.
            for name in ('input_k.spv', 'output_k.spv', 'exposure_k0.spv', 'exposure_k1.spv', 'enc0.spv',
                         'direct_origins.bin', 'weights.bin', 'offsets.bin', 'lut.bin', 'hipnet.bin', 'model.json'):
                fixture(inputs / 'engine' / name)
            env['D4R_BUNDLE_ENGINE'] = str(inputs / 'engine')
            for mode, bundled in itertools.product(('proxy', 'asi'), ('1', '0', 'installer')):
                with self.subTest(mode=mode, bundled=bundled):
                    env['D4R_BUNDLE_NVIDIA'] = bundled
                    env['D4R_INSTALL_MODE'] = mode
                    if mode == 'asi':
                        env.pop('D4R_OPTISCALER', None)
                    result = subprocess.run(['bash', str(repo / 'scripts/package_release.sh'), str(repo / 'dist')],
                                            env=env, text=True, capture_output=True)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    name = f'd4r-{version}' + ('-nonvidia' if bundled == '0' else '')
                    if mode == 'asi':
                        name += '-asi'
                    with zipfile.ZipFile(repo / 'dist' / (name + '.zip')) as archive:
                        self.assertIsNone(archive.testzip())
                        if mode == 'proxy':
                            data = archive.read('OptiScaler.ini')
                            self.assert_routing(data)
                            cfg = configparser.ConfigParser()
                            cfg.read_string(data.decode())
                            self.assertEqual(cfg['FrameGen']['Enabled'], 'false')
                            self.assertEqual(cfg['fakenvapi']['UseFakenvapi'], 'false')
                            self.assertNotIn('d4r/d4r_nvapi64.dll', archive.namelist())
                            self.assertNotIn('d4r/fakenvapi.dll', archive.namelist())
                            self.assertIn('uninstall.sh', archive.namelist())
                            for dependency in ('d4r-features.py', 'licenses/XeSS-LICENSE.txt', 'source/optiscaler.json'):
                                self.assertIn('d4r/' + dependency, archive.namelist())
                            configs.append(data)
                            self.assertEqual(data, (repo / 'dist' / name / 'OptiScaler.ini').read_bytes())
                            # Providers retain upstream's v10 folder, including all FG implementations.
                            for ffx in ('OptiScaler/amd_fidelityfx_loader_dx12.dll', 'OptiScaler/amd_fidelityfx_upscaler_dx12.dll',
                                        'OptiScaler/amd_fidelityfx_framegeneration_dx12.dll',
                                        'OptiScaler/libxell.dll', 'OptiScaler/libxess_fg.dll',
                                        'OptiScaler/dlssg_to_fsr3_amd_is_better.dll',
                                        'OptiScaler/D3D12_OptiScaler/D3D12Core.dll',
                                        'd4r/licenses/FidelityFX-LICENSE.txt'):
                                self.assertIn(ffx, archive.namelist())
                        else:
                            self.assertNotIn('OptiScaler.ini', archive.namelist())
                            self.assertNotIn('dxgi.dll', archive.namelist())
                            self.assertEqual(archive.read('d4r.asi'), archive.read('d4r/nvngx.dll'))
                            self.assertIn('d4r/d4r-setup.exe', archive.namelist())
                            self.assertEqual('d4r/install.sh' in archive.namelist(), bundled == 'installer')
                            self.assertIn('d4r/uninstall.sh', archive.namelist())
                            if bundled != 'installer':
                                self.assertFalse(any(n.endswith('.py') and n != 'd4r/d4r-features.py'
                                                     for n in archive.namelist()))
                            self.assertIn(b'ASI add-on', archive.read('d4r/README.txt'))
                            self.assertEqual({n.split('/')[0] for n in archive.namelist()}, {'d4r.asi', 'd4r'})
                            self.assertIn('d4r/d3d12.dll', archive.namelist())
                            self.assertIn('d4r/d4r12core.dll', archive.namelist())
                            self.assertNotIn('d4r/d3d12core.dll', archive.namelist())
                        kit = 'd4r/tools/models/'
                        self.assertEqual('d4r/tools/build-engine-models.sh' in archive.namelist(), bundled == 'installer')
                        if bundled == 'installer':
                            self.assertEqual(archive.read('d4r/tools/build-engine-models.sh'),
                                             (ROOT / 'packaging/build_engine_models.sh').read_bytes())
                            for executable in ('d4r/tools/build-engine-models.sh', kit + 'bin/glslangValidator'):
                                self.assertTrue((archive.getinfo(executable).external_attr >> 16) & 0o111)
                            for source in ('engine/compile_k.py', 'engine/test_k.py', 'engine/test_m.py',
                                           'engine/rr/test_layer.py', 'vulkan/ptx2glsl.py'):
                                self.assertEqual(archive.read(kit + source), (ROOT / source).read_bytes())
                            self.assertEqual(archive.read(kit + 'bin/glslangValidator'), validator.read_bytes())
                            self.assertEqual(archive.read('d4r/licenses/glslang-LICENSE.txt'), license_path.read_bytes())
                            self.assertIn(kit + 'bin/d3d12_dlss_harness.exe', archive.namelist())
                            self.assertFalse(any(n.endswith('.py') and n != 'd4r/d4r-features.py' and not n.startswith(kit)
                                                 for n in archive.namelist()))
                            self.assertNotIn(kit + 'engine/weights.bin', archive.namelist())
                            self.assertNotIn(kit + 'engine/test_unrelated.py', archive.namelist())
                        else:
                            self.assertFalse(any(n.startswith(kit) for n in archive.namelist()))
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
    def test_selected_features_check_dependencies_and_router_recursion_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            game = Path(temporary)
            (game / 'd4r').mkdir()
            ini = ('[Libraries]\nNvngxPath=d4r\\nvngx.dll\nNvapiPath=d4r\\d4r_nvapi64.dll\n'
                   '[FrameGen]\nEnabled=true\nFGInput=dlssg\nFGOutput=xefg\n'
                   '[NvApi]\nOverrideNvapiDll=true\n[Hooks]\nUseNtdllHooks=false\n')
            (game / 'OptiScaler.ini').write_text(ini)
            def check():
                return subprocess.run(['sh', str(ROOT / 'packaging/d4r-check.sh'), temporary],
                                      capture_output=True, text=True).stdout
            output = check()
            self.assertIn('MISSING  libxess_fg.dll', output)
            self.assertIn('MISSING  fakenvapi.dll', output)
            for name in ('libxess_fg.dll', 'libxell.dll', 'fakenvapi.dll', 'd4r_nvapi64.dll'):
                (game / 'd4r' / name).touch()
            output = check()
            self.assertIn('ok       libxess_fg.dll (optional feature)', output)
            self.assertIn('ok       fakenvapi.dll (latency provider)', output)
            self.assertNotIn('requires Hooks.UseNtdllHooks=false', output)
            (game / 'OptiScaler.ini').write_text(ini.replace('UseNtdllHooks=false', 'UseNtdllHooks=true'))
            self.assertIn('requires Hooks.UseNtdllHooks=false', check())

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
