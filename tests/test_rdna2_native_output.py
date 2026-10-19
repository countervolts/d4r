"""RDNA2 (gfx103x) native hiluma kernels: build gating, packaging allowlist and the f16 rounding contract.

Run: python3 -m unittest tests.test_rdna2_native_output -v
The compile checks need ROCm's clang (D4R_ROCM_DIR, default /opt/rocm) and are skipped without it. The full
texture build additionally needs the extracted DLSS 310.7 PTX (kernels/extracted/ptx) and
D4R_RUN_KERNEL_BUILD_TESTS=1, since it compiles all twelve kernels.
Arithmetic equality with ZLUDA's compile is checked in the harness (docs/native-kernels.md), not here.
"""
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
ROCM = Path(os.environ.get("D4R_ROCM_DIR", "/opt/rocm"))
CLANG = ROCM / "lib/llvm/bin/clang++"
PTX_DIR = ROOT / "kernels/extracted/ptx"
TOOLS = [shutil.which(t) for t in ("llvm-readelf", "llvm-objdump")]
OUTPUTS = {f"hiluma_engine_output_{d}_{mv}_hdr_max_v2_rel" for d in ("depthinv", "depthreg") for mv in ("mvhi", "mvlo")}
INPUTS = {f"hiluma_engine_input_{d}_{mv}_{r}_v2_rel"
          for d in ("depthinv", "depthreg") for mv in ("mvhi", "mvlo") for r in ("hdr", "ldr")}
# cvt.rn.f16.f32 of an fma (and the packed form of a product) of widened f16 values: LLVM would fold the
# conversion into v_fma_mixlo/mixhi_f16, which rounds once instead of twice
TINY_PTX = """.version 8.0
.target sm_80
.address_size 64
.visible .entry tiny(
	.param .align 8 .b8 tiny_param_0[16]
)
{
	.reg .b16 	%rs<6>;
	.reg .f32 	%f<8>;
	.reg .b32 	%r<3>;
	.reg .b64 	%rd<3>;
	ld.param.u64 	%rd1, [tiny_param_0];
	cvta.to.global.u64 	%rd2, %rd1;
	ld.global.u16 	%rs2, [%rd2+8];
	ld.global.u16 	%rs3, [%rd2+10];
	ld.global.u16 	%rs4, [%rd2+12];
	cvt.f32.f16 	%f3, %rs2;
	cvt.f32.f16 	%f4, %rs3;
	cvt.f32.f16 	%f5, %rs4;
	fma.rn.f32 	%f2, %f3, %f4, %f5;
	cvt.rn.f16.f32 	%rs1, %f2;
	mul.rn.f32 	%f6, %f3, %f4;
	cvt.rn.f16x2.f32 	%r1, %f6, %f2;
	st.global.u16 	[%rd2], %rs1;
	st.global.u32 	[%rd2+4], %r1;
	ret;
}
"""


def run(*args, **kwargs):
    return subprocess.run([str(a) for a in args], text=True, capture_output=True, **kwargs)


class Rdna2BuildGating(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        # a compiler that fails if anything reaches it: the refusals must come first
        compiler = self.root / "rocm/lib/llvm/bin/clang++"
        compiler.parent.mkdir(parents=True)
        compiler.write_text("#!/bin/sh\nexit 99\n")
        compiler.chmod(0o755)

    def test_wmma_families_are_refused_on_every_rdna2_target(self):
        for arch in ("gfx1030", "gfx1032", "gfx1036"):
            for family in ("m", "l", "rr", "rrswin"):
                env = dict(os.environ, D4R_ROCM_DIR=str(self.root / "rocm"), D4R_GPU_ARCH=arch)
                result = run("bash", ROOT / "kernels/build.sh", family, self.root / "out", env=env)
                self.assertEqual(result.returncode, 2, (arch, family, result.stderr))
                self.assertIn("need WMMA", result.stderr)

    def test_rdna2_texture_build_does_not_need_zluda(self):
        env = dict(os.environ, D4R_ROCM_DIR=str(self.root / "rocm"), D4R_GPU_ARCH="gfx1032",
                   D4R_DLSS_DLL=str(self.root / "missing.dll"), D4R_DLSS_PTX_DIR=str(self.root / "no-ptx"))
        env.pop("D4R_ZLUDA_EMIT", None)
        result = run("bash", ROOT / "kernels/build.sh", "tex", self.root / "out", env=env)
        self.assertNotIn("D4R_ZLUDA_EMIT", result.stderr)
        self.assertIn("missing extracted PTX directory", result.stderr)

    def test_rdna3_texture_build_still_needs_zluda(self):
        env = dict(os.environ, D4R_ROCM_DIR=str(self.root / "rocm"), D4R_GPU_ARCH="gfx1101",
                   D4R_DLSS_DLL=str(self.root / "missing.dll"))
        env.pop("D4R_ZLUDA_EMIT", None)
        result = run("bash", ROOT / "kernels/build.sh", "tex", self.root / "out", env=env)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("D4R_ZLUDA_EMIT", result.stderr)


class Rdna2Packaging(unittest.TestCase):
    def rdna2_branch(self, path):
        text = (ROOT / path).read_text()
        start = text.index('if [[ "$arch" == gfx103* ]]; then')
        return text[start:text.index("\n  fi\n", start)]

    def test_package_copies_only_the_validated_hiluma_kernels(self):
        branch = self.rdna2_branch("scripts/package_release.sh")
        copies = re.findall(r'"\$tex_dir"/([^ ;]+\.hsaco)', branch)
        self.assertEqual(sorted(copies), ["hiluma_engine_input_*_v2_rel.hsaco",
                                          "hiluma_engine_output_*_hdr_max_v2_rel.hsaco"])
        self.assertIn('"$VARIANT" != clean', branch)  # the no-NVIDIA zip ships no PTX-derived code
        self.assertNotIn("build.sh\" m", branch)
        # the accuracy set gets K's layers only (ZLUDA keeps compiling hiluma there)
        self.assertNotRegex(branch, r'PREFER_ACCURACY=1[^\n]*\n?[^\n]*build.sh" tex')
        self.assertIn("kernel_manifest.py", branch)

    def test_release_builder_builds_rdna2_fast_textures_only(self):
        text = (ROOT / "scripts/build_release_glibc241.sh").read_text()
        start = text.index('if [[ "$arch" == gfx103* ]]; then')
        branch = text[start:text.index("fi\n", start)]
        self.assertIn("D4R_PREFER_ACCURACY=0", branch)
        self.assertNotIn("D4R_PREFER_ACCURACY=1", branch)
        self.assertIn('kernels/build.sh" tex', branch)
        self.assertIn("return 0", branch)


class F16RoundingContract(unittest.TestCase):
    def translate(self, directory):
        ptx = directory / "tiny.ptx"
        ptx.write_text(TINY_PTX)
        result = run("python3", ROOT / "kernels/native/ptx2hip.py", ptx, "tiny", check=True)
        return result.stdout

    def test_translator_routes_f16_conversions_through_f2h_rn(self):
        with tempfile.TemporaryDirectory() as temp:
            source = self.translate(Path(temp))
        self.assertEqual(source.count("f2h_rn("), 3)
        self.assertNotRegex(source, r"\(h1\)f\d")

    @unittest.skipUnless(CLANG.is_file(), "ROCm clang not found (set D4R_ROCM_DIR)")
    def test_no_single_rounding_mix_instructions(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            (temp / "tiny.hip").write_text(self.translate(temp))
            for arch, wave in (("gfx1032", "32"), ("gfx1101", "64")):
                env = dict(os.environ, D4R_ROCM_DIR=str(ROCM), D4R_GPU_ARCH=arch, WAVE=wave)
                run("bash", ROOT / "kernels/native/build_native.sh", temp / "tiny.hip", temp / f"{arch}.hsaco",
                    env=env, check=True)
                asm = (temp / f"{arch}.s").read_text()
                self.assertNotRegex(asm, r"v_fma_mix(lo|hi)_f16", arch)
            # canary: a plain cast in the same kernel is fused, so the check above can fail
            (temp / "cast.hip").write_text((temp / "tiny.hip").read_text().replace("f2h_rn(", "(h1)("))
            env = dict(os.environ, D4R_ROCM_DIR=str(ROCM), D4R_GPU_ARCH="gfx1032", WAVE="32")
            run("bash", ROOT / "kernels/native/build_native.sh", temp / "cast.hip", temp / "cast.hsaco",
                env=env, check=True)
            self.assertRegex((temp / "cast.s").read_text(), r"v_fma_mix(lo|hi)_f16")


@unittest.skipUnless(os.environ.get("D4R_RUN_KERNEL_BUILD_TESTS") == "1", "set D4R_RUN_KERNEL_BUILD_TESTS=1")
@unittest.skipUnless(CLANG.is_file() and PTX_DIR.is_dir() and all(TOOLS), "needs ROCm clang, LLVM tools and PTX")
class Rdna2TextureBuild(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.temp.name) / "gfx1032"
        env = dict(os.environ, D4R_ROCM_DIR=str(ROCM), D4R_GPU_ARCH="gfx1032", D4R_DLSS_DLL="unused",
                   D4R_DLSS_PTX_DIR=str(PTX_DIR))
        run("bash", ROOT / "kernels/build.sh", "tex", cls.out, env=env, check=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def notes(self, name):
        return run(TOOLS[0], "--notes", "-h", self.out / f"{name}.hsaco", check=True).stdout

    def test_exactly_the_validated_kernels(self):
        self.assertEqual({p.stem for p in self.out.glob("*.hsaco")}, OUTPUTS | INPUTS)

    def test_code_objects_are_wave32_gfx1032_without_scratch_or_wmma(self):
        for name in OUTPUTS | INPUTS:
            notes = self.notes(name)
            self.assertIn("gfx1032", notes, name)
            self.assertRegex(notes, r"\.wavefront_size:\s+32", name)
            self.assertRegex(notes, r"\.private_segment_fixed_size:\s+0\b", name)
            code = run(TOOLS[1], "-d", "--mcpu=gfx1032", self.out / f"{name}.hsaco", check=True).stdout
            self.assertNotRegex(code, r"v_wmma|v_fma_mix(lo|hi)_f16|v_dot\d", name)

    def test_output_kernels_use_wgp_mode(self):
        for name in OUTPUTS:
            asm = (self.out / f"{name}.s").read_text()
            self.assertIn(".amdhsa_workgroup_processor_mode 1", asm, name)


if __name__ == "__main__":
    unittest.main()
