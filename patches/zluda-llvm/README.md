# LLVM source patches for the Windows compiler

Base: ZLUDA's LLVM submodule `ff4dc1f7c9e1c64d4d69e40f4ed30c2280a96dfd`
(LLVM 22). `build-zluda-llvm.ps1` applies these patches idempotently before
building. ZLUDA's parent source patches remain in `patches/zluda`.

`0001-backport-gfx1154.patch` adapts official LLVM
[`7a0829e41228`](https://github.com/llvm/llvm-project/commit/7a0829e41228513299c5108685b0bc127463c6a1)
to the older parser/ELF structures. It adds actual gfx1154 processor features,
ISA 11.5.4, parser identity and ELF machine 0x57. It does not map gfx1154 onto
another CPU, patch binary headers or alter existing processors' features.
LLVM's gfx1154 uses `FeatureISAVersion11_5_Common` and the GFX11 speed model.
LLVM code remains Apache-2.0 WITH LLVM-exception (see the dependency license).

The prior offline compiler aborts on a public PTX pattern for gfx1154 with
unrecognized-processor diagnostics and exit `0xC0000409`. After rebuilding
LLVM, rebuild/relink ZLUDA and its offline emitter; changing native Clang
objects alone cannot fix the translated PTX path. Run
`scripts/windows/test-zluda-target-compilation.ps1` to compile the public
Driver API pattern and FP16 MMA fixtures (strict/reference and WMMA modes)
for every target without GPU work. All 33 compilations pass with correct ELF
targets and wave32. Strict gfx12 F16 reference retains its prior FP64 helper;
WMMA mode here is a compilation check, not a default arithmetic change.
This checks code generation and ELF identity, not hardware execution/numerics.
