"""Model builds must time out, retain failures, and reject incomplete/corrupt artifacts."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('model_tools', ROOT / 'packaging/model_tools.py')
model_tools = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_tools)


class ModelBuildTests(unittest.TestCase):
    def test_installer_key_tracks_model_arithmetic_options(self):
        text = (ROOT / 'install.sh').read_text()
        key = text[text.index('key() {'):text.index('have() {')]
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            (root / 'tools').mkdir()
            (root / 'tools/compiler.py').write_text('compiler fixture')
            for name in ('nvngx_dlss.dll', 'nvngx_dlssd.dll'): (root / name).write_bytes(b'DLL fixture')
            env = {k: v for k, v in os.environ.items() if not k.startswith(('D4R_RR_', 'K_', 'M_'))}
            command = ['bash', '-c', 'RUNTIME=$1\n' + key + '\nkey 0', '_', t]
            baseline = subprocess.check_output(command, env=env)
            for name in ('K_DEFS_ENC0', 'M_EXACT', 'D4R_RR_MIX'):
                self.assertNotEqual(baseline, subprocess.check_output(command, env=dict(env, **{name: '1'})))

    def test_child_exit_status_and_timeout_are_preserved(self):
        with tempfile.TemporaryDirectory() as t, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            log = Path(t) / 'capture.log'
            self.assertEqual(model_tools.run(log, 5, [sys.executable, '-c', 'print("failure detail");raise SystemExit(37)']), 37)
            self.assertIn('failure detail', log.read_text())
            self.assertEqual(model_tools.run(log, .15, [sys.executable, '-c', 'import time;time.sleep(60)']), 124)
            self.assertTrue(log.exists())

    def test_manifest_rejects_missing_modified_and_extra_files(self):
        with tempfile.TemporaryDirectory() as t:
            directory = Path(t)
            (directory / 'model.json').write_text('{}')
            weights = directory / 'weights.bin'
            weights.write_bytes(b'weights')
            model_tools.manifest(directory, write=True)
            model_tools.manifest(directory)
            weights.write_bytes(b'corrupt')
            with self.assertRaises(ValueError): model_tools.manifest(directory)
            weights.unlink()
            with self.assertRaises(ValueError): model_tools.manifest(directory)
            weights.write_bytes(b'weights')
            (directory / 'unexpected.spv').write_bytes(b'extra')
            with self.assertRaises(ValueError): model_tools.manifest(directory)

    def test_capture_requires_completed_frame_and_all_layers(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as t:
            directory = Path(t)
            pixels = np.ones((720, 1280, 4), dtype=np.float16)
            pixels.tofile(directory / 'out.raw')
            replay = directory / 'replay'
            replay.mkdir()
            for layer in ('enc0', 'enc1', 'enc2', 'enc3', 'enc4', 'dec5', 'dec4', 'dec3', 'dec2', 'dec1', 'dec0'):
                kernel = 'dltss_pwin_' + layer + '_layer'
                d = replay / ('replay-000001-' + kernel)
                d.mkdir()
                (d / 'manifest.txt').write_text(f'kernel {kernel}\nargs 8\nalloc 0 0x1000 4\npointer 0 0 0\n')
                (d / 'args.bin').write_bytes(bytes(8))
                (d / 'alloc-0.bin').write_bytes(bytes(4))
            model_tools.capture(directory, 'k')
            (d / 'alloc-0.bin').write_bytes(b'')
            with self.assertRaises(ValueError): model_tools.capture(directory, 'k')
            (d / 'alloc-0.bin').write_bytes(bytes(4))
            pixels[0, 0, 0] = np.nan
            pixels.tofile(directory / 'out.raw')
            with self.assertRaises(ValueError): model_tools.capture(directory, 'k')

    def test_dump_does_not_accept_a_failed_child_with_a_replay_marker(self):
        text = (ROOT / 'packaging/build_engine_models.sh').read_text()
        start = text.index('  dump() {')
        function = text[start:text.index('\n  }\n', start) + len('\n  }\n')]
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            (root / 'proton').mkdir(); (root / 'app').mkdir()
            proton = root / 'proton/proton'
            proton.write_text('#!/bin/sh\nmkdir -p "$D4R_CUDA_REPLAY_DUMP_DIR/replay-000001-dltss_pwin_dec0_layer"\nexit 37\n')
            proton.chmod(0o755)
            prefix = 'set -euo pipefail\nWORK=$1; APP=$WORK/app; PROTON=$WORK/proton; STEAM_ROOT=$WORK; RUNTIME=$WORK\n'
            prefix += 'MODEL_TOOLS=$2; winpath() { printf "%s" "$1"; }; fail() { echo "$*" >&2; exit 1; }\n'
            result = subprocess.run(['bash', '-c', prefix + function + '\ndump k 11 ignored\n', '_', t,
                                     str(ROOT / 'packaging/model_tools.py')], capture_output=True, text=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('capture process failed', result.stderr)
            self.assertTrue((root / 'k/run.log').exists())
