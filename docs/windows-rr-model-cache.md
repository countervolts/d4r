# Local Windows RR model cache

## Native Python builder without Git Bash

`tools/windows/build_rr_model.py` now runs the extraction, template generation,
weight packing/widening, glslang compilation and exact-conversion passes directly
on Windows. It uses the same existing model scripts and shader recipe as the
Linux shell, with pinned RR DLL/glslang hashes and exact gfx11 arithmetic.
Python, NumPy and glslang must already be provisioned; automatic dependency
download and install.bat integration remain unfinished.

```powershell
python tools/windows/build_rr_model.py `
  --dll build/install-bat-cache/nvidia/nvngx_dlssd.dll `
  --preset D --output build/my-native-rr-d
```

Use a fresh ignored build/ directory. `--reuse` verifies the DLL, compiler,
Python, NumPy version, source hashes and every top-level model product before
returning. Missing, modified or wrong-preset caches fail without rebuilding in
place. Each child step has a 600-second limit; a timed-out Windows process tree
is terminated. The caller's environment is copied rather than changed. No Bash,
PowerShell build wrapper, game access, driver change or policy bypass is used.
Derived shader/weight products stay local and must not be bundled.

Actual preset-D build evidence: `build/native-rr-d-python-native-v2/`.
65 of 66 common top-level artifacts match the prior validated Windows model
byte-for-byte, including every shader, weight and argument template. The sole
manifest difference selects the Linux recipe's 64-wide output subgroup instead
of the prior Windows model's 32. Six native NGX hardware-depth frames still
match the old standalone reference exactly, with optional guides, enhanced/
inherited access, format changes, stencil preservation and no SR dependencies.
No speed improvement is claimed for that subgroup setting.

Cache reuse took approximately 0.26 seconds. Wrong-preset reuse and a modified
weight in a separate copied fixture were rejected. The valid model remained
unchanged. Evidence: `build/rr-native-python-builder-validation.json` and
`build/rr-ngx-python-native-model-v1/summary.json`. Preset E validation is recorded
below. Earlier v1/v2 artifacts
are preserved; v1 also differed in manifest line endings, corrected in v2.

## Preset E build and verified local deployment

The native Python E build completed all 48 steps, including its 4096-byte noise
prefix. As with D, 65/66 common top-level artifacts match the prior E model;
the manifest selects the Linux output subgroup width. Six NGX hardware-depth
frames match the prior E standalone engine reference exactly, with optional
guides, format switching, enhanced/inherited access, stencil preservation and
no configured SR dependencies. E cache reuse also passes.
Evidence: `build/rr-native-python-e-comparison.json` and
`build/rr-ngx-python-native-e-model-v1/summary.json`.

`tools/windows/deploy_rr_model.py` verifies the native Python cache's product
hashes and complete D/E manifest, copies a private model into a versioned runtime
`engine/rr-<preset>-<cache hash>` directory, and verifies every copied file.
Existing models are preserved. Repeated deployment reuses identical products;
changed products fail instead of replacing an existing model. No compiler or
NVIDIA DLL is copied. With `--enable`, it backs up d4r.ini outside the game/runtime
parent, then selects native RR and the relative model path. Comments, BOM,
line endings, SR settings and unrelated sections are preserved.

```powershell
python tools/windows/deploy_rr_model.py --model build/my-native-rr-d `
  --runtime C:/Games/Example/d4r --enable
```

Only ignored mock-runtime paths were used in validation. An actual E deployment
verified 66 products, preserved an exact INI backup, and repeated without further
configuration changes/backups. A tampered model was rejected with every runtime
file unchanged. Three CPU tests cover BOM/CRLF/comments, absent/duplicate RR
sections and idempotence. The deployed model then passed six exact RR frames
through the packaged architecture loader using the preserved, deployment-written
relative INI path. Evidence: `build/rr-model-deploy-validation.json`,
`build/rr-model-deploy-negative-validation.json`, and
`build/rr-ngx-deployed-e-ini-v1/summary.json`.

This automates local model deployment and selection. Dependency provisioning and
install.bat integration remain unfinished; the tools are not added to v4 ZIP.
No game was accessed, source committed, release published or system policy changed.

## Portable dependency bundle

`tools/windows/prepare_rr_tools.py` downloads and verifies pinned archives for
[Python 3.12.10 embeddable x64](https://www.python.org/downloads/release/python-31210/),
[NumPy 2.4.6 cp312 Windows x64](https://pypi.org/project/numpy/2.4.6/), and
glslang 16.6.0. It copies the tracked model-build sources, the kernel allowlist,
and license texts into a fresh local bundle. Python runs isolated with explicit
bundle-relative imports; system Python, pip and Git Bash are not needed to build
or deploy models from the prepared bundle. No system Python installation, PATH,
driver, security policy or game folder is changed.

```powershell
python tools/windows/prepare_rr_tools.py --output build/my-rr-tools `
  --archive build/my-rr-tools.zip
build/my-rr-tools/python/python.exe `
  build/my-rr-tools/recipe/tools/windows/build_rr_model.py `
  --dll build/install-bat-cache/nvidia/nvngx_dlssd.dll --preset D `
  --output build/my-rr-tools/recipe/build/rr-d
```

Prepared local archive: `build/rr-portable-tools-v2.zip`, 40,154,515 bytes;
SHA256 `297289a57a82142eb952479c99112b1c352a6753eb15bc6b2431bf686aed9b22`.
All 1,111 recorded files verify. The archive is created from the pristine tool
manifest before model generation and contains no NVIDIA DLL, extracted PTX,
SPIR-V model shader or generated model. Building inside the extracted tool
directory later creates private products there; those products must not be
added to an archive. The recorded clean tools ZIP remains unchanged.

Actual isolated-bundle D build completed all 48 steps. All 66 top-level model
artifacts match the already executed native Python D model byte-for-byte.
Bundled NumPy loaded from the bundle's own vendor directory, and cache reuse
passes. Bundled Python also deployed the verified 66-product model into a fresh
ignored mock runtime, selected it through INI and preserved an exact external
configuration backup. Evidence: `build/rr-portable-tools-validation.json` and
`build/rr-portable-deployment-validation.json`.

The first bundle lacked `native_kernels.txt`; its build failed before weight
extraction and remains preserved separately. V2 includes that allowlist. Both
cache implementations now fingerprint source `.txt` inputs too. Existing cache
stamps made before this change may reject reuse; their artifacts are preserved,
and rebuilding uses a fresh output directory. No cache was silently restamped.

Dependency preparation and portable model build/deployment are verified.
Connecting the tool bundle to install.bat and the main runtime ZIP remains open,
along with game integration, FSR execution and performance parity.

## Existing PowerShell/Git Bash workflow

The native D/E builder can reuse a verified model directory instead of rebuilding
all extracted weights and shaders. This is local build automation; it does not
capture game inputs or implement the Windows RR game adapter.

```powershell
./scripts/windows/build-native-rr-model.ps1 `
  -DlssdDll build/install-bat-cache/nvidia/nvngx_dlssd.dll `
  -Preset D -OutputDirectory build/native-rr-d-cache-verified -ReuseCache
```

Use the path to your own installer-staged, pinned DLL. On a new output directory,
the builder compiles the model and writes `windows-rr-cache.json`. On an existing
directory, `-ReuseCache` checks the manifest and returns only if every input and
product hash still matches. Missing, modified or stale caches fail; select a new
directory to rebuild. Legacy directories without a cache manifest are not
silently trusted or overwritten.

The cache records the DLL, Python, Git Bash and glslang hashes; NumPy version;
source hashes; preset; gfx1101 architecture; exact arithmetic; and hashes of all
top-level generated model artifacts. It also checks the RR manifest preset,
arithmetic and complete 12-stage shader/weight references. All derived model data
stays under ignored `build/`. The cache is not a public package artifact.

Validation on the real locally built preset D model:

- Full build completed and stamped 66 artifacts.
- Verified reuse took approximately 0.64 seconds and did not rewrite the shader.
- Flipping a byte in a weight file caused rejection; the original file was
  restored and its SHA256 rechecked before a final successful cache hit.
- Requesting E against the D model was rejected.
- The builder restored the caller's `D4R_RR_GFX11` and `D4R_RR_RELAXED` settings.

Evidence: `build/rr-cache-validation.json`. These checks establish cache behavior,
not RR quality, game support, or a performance improvement in inference.
