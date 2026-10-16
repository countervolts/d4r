# M11 historical Windows scripts

The source archive's native-Windows M11 scripts are under
[`../../windows/m11/tools/`](../../windows/m11/tools/) and its shim build scripts
under [`../../windows/m11/src/`](../../windows/m11/src/).

They deliberately retain their original **internal** layout because many derive
paths from `$PSScriptRoot`. This pointer does not imply they are interchangeable
with the mainline `scripts/windows/` build and release tooling. See
[`../../docs/windows-m11.md`](../../docs/windows-m11.md).
