"""Separate corresponding-source asset for the lightweight Windows test ZIP.

Reads committed trees only, including the exact patched GPL OptiScaler tree and
its pinned submodules. Never copies a working directory or NVIDIA binaries.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import zipfile


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    repo = args.repo.resolve()
    info = json.loads((repo / "dist/optiscaler-windows-d4r/build-info.json").read_text(encoding="utf-8-sig"))
    opti = repo / "external/OptiScaler"
    if info["sourceDirty"] or git(opti, "status", "--porcelain").strip():
        raise RuntimeError("OptiScaler must match its clean binary source tree")
    if git(opti, "rev-parse", "HEAD").decode().strip() != info["sourceCommit"]:
        raise RuntimeError("OptiScaler source does not match the binary")
    if git(repo, "status", "--porcelain").strip():
        raise RuntimeError("Commit the packaging sources before publishing")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise RuntimeError("Use a new output archive")
    provenance = {"d4rCommit": git(repo, "rev-parse", "HEAD").decode().strip(),
                  "optiScalerBinary": info["dllSha256"], "sourceTrees": [], "excludedNvidiaBinaries": []}

    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as output:
        def tree(root, revision, prefix):
            provenance["sourceTrees"].append({"path": prefix, "commit": revision})
            raw = git(root, "archive", "--format=zip", revision)
            with zipfile.ZipFile(io.BytesIO(raw)) as source:
                for item in source.infolist():
                    if item.is_dir():
                        continue
                    name = prefix + "/" + item.filename
                    lowered = name.lower()
                    nvidia_binary = ("/nvapi/" in lowered or "nvngx" in Path(lowered).name) and Path(lowered).suffix in {
                        ".dll", ".exe", ".lib", ".bin", ".cubin", ".fatbin", ".pdb"}
                    if nvidia_binary:
                        provenance["excludedNvidiaBinaries"].append(name)
                        continue
                    output.writestr(name, source.read(item))
            for entry in git(root, "ls-tree", "-rz", revision).split(b"\0"):
                if not entry:
                    continue
                header, path = entry.split(b"\t", 1)
                mode, _, commit = header.split()
                if mode != b"160000":
                    continue
                child = path.decode("utf-8")
                child_root = root / child
                child_commit = commit.decode()
                if git(child_root, "rev-parse", "HEAD").decode().strip() != child_commit:
                    raise RuntimeError(f"Submodule does not match committed source: {child}")
                if git(child_root, "status", "--porcelain").strip():
                    raise RuntimeError(f"Dirty source submodule: {child}")
                tree(child_root, child_commit, prefix + "/" + child)

        tree(repo, provenance["d4rCommit"], "d4r")
        tree(opti, info["sourceCommit"], "OptiScaler")
        output.writestr("source-provenance.json", json.dumps(provenance, indent=2) + "\n")
        output.writestr("READ-ME-SOURCES.txt", """CORRESPONDING SOURCES -- NOT NEEDED TO TEST THE GAME

d4r/ contains the committed Windows port, build scripts, dependency patches
and licenses. OptiScaler/ contains the exact modified GPL-3.0 source tree and
its pinned source submodules corresponding to the distributed OptiScaler.dll.
See source-provenance.json for exact commits and the binary SHA256.

Build instructions: d4r/docs/windows-rdna4-port.md and
d4r/patches/optiscaler/README.md. The normal build script retrieves the pinned
upstream checkout, submodules and applies the exported source patches.
NVIDIA import-library binaries are intentionally excluded from this archive;
the pinned public NVIDIA/nvapi SDK used by the builder supplies them locally.
No proprietary NVIDIA runtime DLL or NVIDIA-derived private kernel is included.

The separately bundled ZLUDA/LLVM changes are in d4r/patches/zluda and
d4r/patches/zluda-llvm, with upstream source commits and build instructions.
""")
    with args.output.open("rb") as stream:
        sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    print(json.dumps({"archive": str(args.output), "bytes": args.output.stat().st_size,
                      "sha256": sha256,
                      "sourceTrees": len(provenance["sourceTrees"])}))


if __name__ == "__main__":
    main()
