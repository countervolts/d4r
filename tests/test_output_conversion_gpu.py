"""R11G11B10 storage conversion must match the original blit for all half inputs."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from platform_support import vulkan_flags

ROOT=Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.environ.get('D4R_RUN_GPU_TESTS')=='1','opt-in GPU test')
class OutputConversionGpuTests(unittest.TestCase):
    def test_fused_r11_matches_blit(self):
        with tempfile.TemporaryDirectory() as temp:
            runner=Path(temp)/'conversion'
            subprocess.run(['c++','-std=c++17','-O2','-Wall','-Wextra','-Werror',
                            '-Wno-missing-field-initializers','-I'+str(ROOT/'tools'),
                            '-I'+str(ROOT/'build/vulkan-staging'),
                            str(ROOT/'tools/vulkan_output_conversion_probe.cpp'),*vulkan_flags(),'-o',str(runner)],
                           check=True,capture_output=True,text=True)
            subprocess.run([str(runner)],check=True,capture_output=True,text=True,timeout=30)
