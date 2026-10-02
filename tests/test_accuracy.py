"""CPU checks for accuracy policy, config precedence and native-set selection.

Run: python3 -m unittest discover -s tests -v
Kernel arithmetic additionally needs kernel/model replays and an RTX reference.
"""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
POLICY = ("D4R_ZLUDA_IGNORE_DENORMAL", "D4R_ZLUDA_WMMA_F32ACC", "D4R_ZLUDA_FAST_MATH",
          "D4R_ZLUDA_WAVE64", "D4R_ELIDE_NGX_SYNC")


class AccuracyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build = tempfile.TemporaryDirectory()
        cls.runner = Path(cls.build.name) / "selection"
        source = r'''
#include "d4r_native_selection.h"
int main(int argc, char** argv) {
    if (argc == 2 && strcmp(argv[1], "policy") == 0) {
        printf("%d", d4r_apply_accuracy_policy());
        const char* keys[] = {"D4R_ZLUDA_IGNORE_DENORMAL", "D4R_ZLUDA_WMMA_F32ACC",
            "D4R_ZLUDA_FAST_MATH", "D4R_ZLUDA_WAVE64", "D4R_ELIDE_NGX_SYNC"};
        for (unsigned i = 0; i < sizeof(keys) / sizeof(keys[0]); ++i)
            printf(" %s", getenv(keys[i]) ? getenv(keys[i]) : "unset");
        puts("");
        return 0;
    }
    if (argc == 3 && strcmp(argv[1], "kernel") == 0) {
        printf("%d\n", d4r_native_kernel_allowed(argv[2])); return 0;
    }
    if (argc != 5) return 2;
    char path[1024] = {0};
    int kind = d4r_select_native_source(argv[1], argv[2], atoi(argv[3]), atoi(argv[4]), path, sizeof(path));
    printf("%d %s\n", kind, path);
    return 0;
}
'''
        subprocess.run(["cc", "-x", "c", "-std=gnu11", "-Wall", "-Wextra", "-Werror",
                        "-I", str(ROOT / "tools"), "-o", str(cls.runner), "-"],
                       input=source, text=True, check=True, capture_output=True)

    @classmethod
    def tearDownClass(cls):
        cls.build.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def native(self, directory, accuracy=False, manifest=True):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "kernel.hsaco").write_bytes(b"fixture")
        if manifest:
            (directory / "d4r-kernels.txt").write_text("kernel 1234\n")
        if accuracy:
            (directory / "d4r-accuracy.txt").write_text("1\n")

    def select(self, architecture="gfx1101", accuracy=True, fp8=True, directory=None):
        output = subprocess.check_output([str(self.runner), str(directory or self.root), architecture,
                                          str(int(accuracy)), str(int(fp8))], text=True).strip()
        kind, _, path = output.partition(" ")
        return int(kind), Path(path)

    def test_select_accuracy_for_every_release_target(self):
        for arch in ("gfx1100", "gfx1101", "gfx1102", "gfx1103", "gfx1200", "gfx1201"):
            self.native(self.root / arch)
            accurate = self.root / "accuracy" / arch
            self.native(accurate, accuracy=True)
            self.assertEqual(self.select(arch), (1, accurate))
            self.assertEqual(self.select(arch, accuracy=False), (1, self.root / arch))

    def test_rdna4_fp8_selection_and_widened_fallback(self):
        for arch in ("gfx1200", "gfx1201"):
            accurate = self.root / "accuracy" / arch
            fp8 = self.root / "accuracy" / (arch + "-fp8")
            self.native(accurate, accuracy=True)
            self.assertEqual(self.select(arch), (1, accurate))
            self.native(fp8, accuracy=True)
            self.assertEqual(self.select(arch), (1, fp8))
            self.assertEqual(self.select(arch, fp8=False), (1, accurate))

    def test_missing_accuracy_never_selects_fast_or_wrong_gpu(self):
        self.native(self.root / "gfx1101")
        self.native(self.root / "accuracy" / "gfx1201", accuracy=True)
        self.assertEqual(self.select()[0], 0)
        self.native(self.root / "accuracy" / "gfx1101")  # unmarked binaries
        self.assertEqual(self.select()[0], 0)

    def test_flat_accuracy_developer_set_requires_valid_marker(self):
        self.native(self.root, manifest=False)
        self.assertEqual(self.select()[0], 0)
        self.assertEqual(self.select(accuracy=False), (2, self.root))
        (self.root / "d4r-accuracy.txt").write_text("1\n")
        self.assertEqual(self.select(), (2, self.root))
        (self.root / "d4r-accuracy.txt").write_text("0\n")
        self.assertEqual(self.select()[0], 0)

    def test_missing_fast_target_does_not_select_release_root(self):
        self.native(self.root / "gfx1201")
        self.assertEqual(self.select(accuracy=False)[0], 0)

    def test_accuracy_policy_overrides_conflicting_fast_settings(self):
        env = dict(os.environ, D4R_PREFER_ACCURACY="1", **dict.fromkeys(POLICY, "1"))
        output = subprocess.check_output([str(self.runner), "policy"], env=env, text=True).strip()
        self.assertEqual(output, "1 0 0 0 0 0")

    def test_default_off_preserves_existing_policy(self):
        env = {k: v for k, v in os.environ.items() if k != "D4R_PREFER_ACCURACY"}
        env.update(dict.fromkeys(POLICY, "1"))
        output = subprocess.check_output([str(self.runner), "policy"], env=env, text=True).strip()
        self.assertEqual(output, "0 1 1 1 1 1")

    def config(self, text, extra_env=None, expected=0):
        path = self.root / "d4r.ini"
        path.write_text(text)
        env = {k: v for k, v in os.environ.items() if not k.startswith("D4R_")}
        env.update(extra_env or {})
        result = subprocess.run(["python3", str(ROOT / "scripts/d4r_config.py"), "--config", str(path)],
                                env=env, text=True, capture_output=True)
        self.assertEqual(result.returncode, expected, result.stderr)
        return result

    def test_config_default_and_boolean_spellings(self):
        self.assertIn("export D4R_PREFER_ACCURACY=0", self.config("").stdout)
        for value in ("true", "on", "yes", "1"):
            result = self.config(f"[Kernels]\nPreferAccuracy = {value}\nNativeKernels = off\n")
            self.assertIn("export D4R_PREFER_ACCURACY=1", result.stdout)
            self.assertNotIn("D4R_ZLUDA_NATIVE_DIR", result.stdout)
        for value in ("false", "off", "no", "0"):
            self.assertIn("export D4R_PREFER_ACCURACY=0",
                          self.config(f"[Kernels]\nPreferAccuracy = {value}\n").stdout)

    def test_explicit_environment_wins_and_invalid_setting_is_reported(self):
        self.assertNotIn("export D4R_PREFER_ACCURACY",
                         self.config("[Kernels]\nPreferAccuracy=true\n", {"D4R_PREFER_ACCURACY": "0"}).stdout)
        result = self.config("[Kernels]\nPreferAccuracy=perhaps\n", expected=2)
        self.assertIn("PreferAccuracy must be true or false", result.stderr)

    def test_swin_encoder_fallback_keeps_other_layers(self):
        encoders = ("rrlite_enc1_4x4", "rrlite_enc2_4x4")
        others = ("rrlite_enc3_tube_4x4", "rrlite_dec1_4x4", "rrlite_dec2_4x4",
                  "rrlite_enc0_4x4_mvhi_hdr", "rrlite_post_3_1_mvhi_hdr", "dltss_pwin_enc1_layer")
        for value in ("0", "1", None):
            env = {k: v for k, v in os.environ.items() if k != "D4R_NATIVE_SWIN_ENCODERS"}
            if value is not None:
                env["D4R_NATIVE_SWIN_ENCODERS"] = value
            for name in encoders + others:
                got = subprocess.check_output([str(self.runner), "kernel", name], env=env, text=True).strip()
                self.assertEqual(got, "0" if value == "0" and name in encoders else "1")

    def test_swin_encoder_config_and_environment_override(self):
        for value, expected in (("false", "0"), ("true", "1"), ("auto", "1")):
            self.assertIn(f"export D4R_NATIVE_SWIN_ENCODERS={expected}",
                          self.config(f"[Kernels]\nNativeSwinEncoders = {value}\n").stdout)
        result = self.config("[Kernels]\nNativeSwinEncoders = false\n",
                             {"D4R_NATIVE_SWIN_ENCODERS": "1"})
        self.assertNotIn("export D4R_NATIVE_SWIN_ENCODERS", result.stdout)
        self.config("[Kernels]\nNativeSwinEncoders = maybe\n", expected=2)

    def test_motion_dilation_config_bounds_and_environment_precedence(self):
        for value, expected in (("0", "0"), ("1", "1"), ("2", "2"), ("auto", "0")):
            self.assertIn(f"export D4R_MOTION_DILATION={expected}",
                          self.config(f"[Interop]\nMotionVectorDilation = {value}\n").stdout)
        for value in ("-1", "3", "true", "1.5"):
            self.config(f"[Interop]\nMotionVectorDilation = {value}\n", expected=2)
        self.assertNotIn("export D4R_MOTION_DILATION",
                         self.config("[Interop]\nMotionVectorDilation = 2\n",
                                     {"D4R_MOTION_DILATION": "0"}).stdout)

    def test_shipped_configs_default_off(self):
        for file in ("config/d4r.ini.default", "packaging/d4r.ini"):
            self.assertRegex((ROOT / file).read_text(), r"(?m)^PreferAccuracy\s*=\s*false$")

    def test_builder_refuses_to_certify_mixed_fast_binaries(self):
        compiler = self.root / "rocm/lib/llvm/bin/clang++"
        compiler.parent.mkdir(parents=True)
        compiler.write_text("#!/bin/sh\nexit 99\n")
        compiler.chmod(0o755)
        output = self.root / "mixed"
        self.native(output, manifest=False)
        env = dict(os.environ, D4R_PREFER_ACCURACY="1", D4R_ROCM_DIR=str(self.root / "rocm"))
        for marker in (None, "0\n"):
            if marker is not None:
                (output / "d4r-accuracy.txt").write_text(marker)
            result = subprocess.run(["bash", str(ROOT / "kernels/build.sh"), "k", str(output)],
                                    env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("accuracy kernels need an empty output directory", result.stderr)


if __name__ == "__main__":
    unittest.main()
