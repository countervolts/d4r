## Windows

Extract the Windows release ZIP beside the game's main executable, run `install.bat`,
then launch the game normally and select DLSS. The installer downloads and verifies
the required dependencies; Python and a compiler are included where needed.
Super Resolution setup needs no GPU selection or Ray Reconstruction model build.
See [Windows installation and settings](packaging/windows/WINDOWS_README.txt).

To compile the Windows K/M kernels for every registered RDNA 3, 3.5 and 4 target,
run `build-kernels.bat` from the source checkout. On a new build machine, first run
`powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\setup-windows-tools.ps1 -RuntimeProfile therock`
(requires Python). Outputs go to `dist/windows-kernels-<gfx target>`;
`build/windows-kernels/matrix-build.json` records hashes and target checks.
No GPU or NVIDIA DLL is required for this kernel build. These kernel outputs are
build artifacts; use the release ZIP to install into a game.
See [GPU coverage and full release builds](docs/windows-gpu-support.md).

> **Not affiliated with NVIDIA or AMD.**

## Results

RX 7700 XT on the linux build, windows should get similar performance (plus ts lowkey outdated)

[DLSS Ultra Performance demo video](https://cdn.ayois.gay/dlss).

More up to date videos (0.1.4), still linux build

**Ready or Not**

| Model | Quality | Performance |
|---|---|---|
| DLSS 4.5 | [video](https://cdn.ayois.gay/sk-ant-api03-OPG1vqqsmDkNduG5fXJBelDI-1APyL9_ytJrnOSekZxoh-2NL8NqNuqL3pCI24kJQsudaB896GYpu5TFv93tabAgK1H2sAA) | [video](https://cdn.ayois.gay/sk-ant-api03-d-4XS1it5mtMkm-QRz0Qnkfu9FAjz_dFJURvsh_3gPZTnC1sljpWc28F6fyjUWIC6xlZm-FckWGjaP-KuIipYmroa3dQ8AA) |
| DLSS 4 | [video](https://cdn.ayois.gay/sk-ant-api03-uXTWm4hN5jZksiSm4fvgIOr0dw43hGJ0TpZLe2ZghIbGb7aR9CGGlTV-1paBYO7G6lzIf83tOrCdaKdJerJ4re6epRWDEAA) | [video](https://cdn.ayois.gay/sk-ant-api03-fcZCkctdFOv513z1u6rPBrRseY47Qnw43Gnv5mZf_GekHw-yUoWYcBixjp2qqqbBIf7sVmht0Y676U96n2cwmFIsRgWkHAA) |
| DLSS 3 | [video](https://cdn.ayois.gay/sk-ant-api03-_vuywLcsMzZiJxcNgucBeeXFzWi44mes5A1777bg_4XQ2X0MH22J9M7JFpTS0_q6yaGWMXf3FvL8MDiv_dc7UANxrCmlGAA) | [video](https://cdn.ayois.gay/sk-ant-api03-WzaIy34WIesQeOOTVQNXZwpPPE8khmbUynvMw9B-Luj8GFcZ_sJvGXfg6aW00TD0Rr3nWu43ELJGx2K6qs_HL6FkLdmb1AA) |

**Townfall**

| Model | Quality | Ultra Performance |
|---|---|---|
| DLSS 4.5 | [video](https://cdn.ayois.gay/sk-ant-api03-p8uL0hOUklvA-BSeEafS8LQU-gayzbqAE3NacWAmckLoXOSf2sXGf7OeG1LmvOqeGFRh_ceCWi4pLBl9yxJlvYwQWH3Y1AA) | [video](https://cdn.ayois.gay/sk-ant-api03-Mz0BzYTrLjaYSuQXtOEHTw4ydYVbPvHmMjflm8UhFp976ftXqMExD8BCMYlo93JLEGwVL3rMWoZR0CH_deJJU7jX8_s3GAA) |
| DLSS 4 | [video](https://cdn.ayois.gay/sk-ant-api03-WgNoUPTwH0vt55Ag7RaWprXXINU5OOOEtuDXOvIjHgfAScSiIPIJGVaERoF-EA3JO7LS8xkohUlb6AFsW8AVZ2vMB14JwAA) | [video](https://cdn.ayois.gay/sk-ant-api03-skXZVD6G4DUdAfQw0RLz_UJW8GOfBlpZx1N4ezaF_UIXm7OIY1ciQysfRu4zRsxSXA8plZJi4NSzkXkV7Oxmg9ytg9zaaAA) |

## Star History

<a href="https://www.star-history.com/?type=date&repos=countervolts%2Fd4r">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=countervolts/d4r&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=countervolts/d4r&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=countervolts/d4r&type=date&legend=top-left" />
 </picture>
</a>

## License

GNU General Public License, version 3 only (GPL-3.0-only; see [LICENSE](LICENSE)). The patches in `patches/` are offered under the licenses of the projects they modify: ZLUDA (Apache-2.0 or MIT) and vkd3d-proton (LGPL-2.1). See [NOTICE](NOTICE).