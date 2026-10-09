"""Lock down the x64 NGX Vulkan resource ABI used by game-owned descriptors."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class VulkanAbiTests(unittest.TestCase):
    def test_x64_resource_layout_on_host_and_windows(self):
        code = r'''
#include <cstddef>
#include "d4r_vulkan_ngx.h"
static_assert(sizeof(void*) == 8);
// NVIDIA nvsdk_ngx_defs_vk.h x64 ABI: union + enum + bool.
static_assert(sizeof(NgxVkImage) == 48);
static_assert(offsetof(NgxVkImage, view) == 0);
static_assert(offsetof(NgxVkImage, image) == 8);
static_assert(offsetof(NgxVkImage, range) == 16);
static_assert(offsetof(NgxVkImage, format) == 36);
static_assert(offsetof(NgxVkImage, width) == 40);
static_assert(offsetof(NgxVkImage, height) == 44);
static_assert(sizeof(NgxVkResource) == 56);
static_assert(offsetof(NgxVkResource, resource) == 0);
static_assert(offsetof(NgxVkResource, type) == 48);
static_assert(offsetof(NgxVkResource, readWrite) == 52);
'''
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / 'abi.cpp'
            source.write_text(code)
            include = directory / 'include'
            include.mkdir()
            (include / 'vulkan').symlink_to('/usr/include/vulkan', target_is_directory=True)
            (include / 'vk_video').symlink_to('/usr/include/vk_video', target_is_directory=True)
            for compiler in ('c++', 'x86_64-w64-mingw32-g++'):
                with self.subTest(compiler=compiler):
                    if not shutil.which(compiler):
                        continue
                    subprocess.run([compiler, '-std=c++20', '-Wall', '-Wextra', '-Werror',
                                    '-I', str(ROOT / 'tools'), '-I', str(include),
                                    '-c', str(source), '-o', str(directory / 'abi.o')],
                                   check=True, capture_output=True, text=True)


if __name__ == '__main__':
    unittest.main()
