#!/usr/bin/env python3
"""Read-only PE inventory for locally supplied NGX/DLSS DLLs.

This reports metadata, imports/exports, resource metadata, section sizes/entropy,
and targeted strings. It never writes or extracts DLL payloads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import sys
from collections import Counter
from pathlib import Path
from typing import Any


class PEError(ValueError):
    pass


def unpack(fmt: str, data: bytes, offset: int) -> tuple[Any, ...]:
    size = struct.calcsize(fmt)
    if offset < 0 or offset + size > len(data):
        raise PEError(f"truncated PE data at file offset 0x{offset:x}")
    return struct.unpack_from(fmt, data, offset)


def c_string(data: bytes, offset: int, limit: int = 4096) -> str:
    if offset < 0 or offset >= len(data):
        return ""
    end = data.find(b"\0", offset, min(len(data), offset + limit))
    if end < 0:
        end = min(len(data), offset + limit)
    return data[offset:end].decode("ascii", errors="replace")


def entropy(block: bytes) -> float:
    if not block:
        return 0.0
    counts = Counter(block)
    size = len(block)
    return -sum((n / size) * math.log2(n / size) for n in counts.values())


def aligned(value: int, alignment: int = 4) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


def read_utf16z(data: bytes, offset: int, limit: int) -> tuple[str, int]:
    end = offset
    max_end = min(len(data), offset + limit)
    chars = bytearray()
    while end + 1 < max_end:
        word = struct.unpack_from("<H", data, end)[0]
        end += 2
        if word == 0:
            break
        chars.extend(struct.pack("<H", word))
    return chars.decode("utf-16le", errors="replace"), end


def parse_version_info(blob: bytes) -> dict[str, Any]:
    result: dict[str, Any] = {"strings": {}}

    def visit(start: int, parent_end: int) -> None:
        if start + 6 > parent_end or start + 6 > len(blob):
            return
        length, value_length, value_type = unpack("<HHH", blob, start)
        if length < 6:
            return
        end = min(start + length, parent_end, len(blob))
        key, after_key = read_utf16z(blob, start + 6, max(2, end - (start + 6)))
        value_start = aligned(after_key)

        if value_type == 1:
            value_bytes = value_length * 2
        else:
            value_bytes = value_length
        value_end = min(value_start + value_bytes, end)

        if key == "VS_VERSION_INFO" and value_bytes >= 52 and value_start + 52 <= end:
            fixed = unpack("<13I", blob, value_start)
            if fixed[0] == 0xFEEF04BD:
                file_ms, file_ls, prod_ms, prod_ls = fixed[2:6]
                result["fixed_file_version"] = [file_ms >> 16, file_ms & 0xFFFF, file_ls >> 16, file_ls & 0xFFFF]
                result["fixed_product_version"] = [prod_ms >> 16, prod_ms & 0xFFFF, prod_ls >> 16, prod_ls & 0xFFFF]

        if value_type == 1 and value_length and key not in {"StringFileInfo", "VarFileInfo"}:
            value = blob[value_start:value_end].decode("utf-16le", errors="replace").rstrip("\0")
            result["strings"].setdefault(key, value)

        child = aligned(value_end)
        while child + 6 <= end:
            child_len = unpack("<H", blob, child)[0]
            if child_len < 6 or child + child_len > end:
                break
            visit(child, end)
            child = aligned(child + child_len)

    visit(0, len(blob))
    if not result["strings"]:
        result.pop("strings")
    return result


class PEImage:
    DIRECTORY_NAMES = {
        0: "export",
        1: "import",
        2: "resource",
        3: "exception",
        4: "security_certificate_file_offset",
        5: "base_relocation",
        6: "debug",
        9: "tls",
        10: "load_config",
        12: "iat",
        13: "delay_import",
    }
    RESOURCE_TYPES = {
        1: "CURSOR",
        2: "BITMAP",
        3: "ICON",
        4: "MENU",
        5: "DIALOG",
        6: "STRING",
        9: "ACCELERATOR",
        10: "RCDATA",
        14: "GROUP_ICON",
        16: "VERSIONINFO",
        24: "MANIFEST",
    }
    STRING_PATTERN = re.compile(
        r"(?:ngx|dlss|cuda|nvapi|tensor|weight|shader|cubin|fatbin|\.pth|d3d12|dxgi|vulkan|nvml|driver|nvsdk)",
        re.IGNORECASE,
    )
    MAGIC_MARKERS = {
        "ELF magic candidate": b"\x7fELF",
        "SPIR-V little-endian magic": b"\x03\x02\x23\x07",
        "DXBC container marker": b"DXBC",
        "DXIL marker": b"DXIL",
        "Zstandard frame magic": b"\x28\xb5\x2f\xfd",
        "LZ4 frame magic": b"\x04\x22\x4d\x18",
        "gzip header": b"\x1f\x8b",
        "ZIP local-file header": b"PK\x03\x04",
        "CUDA fatbin symbol": b"__nv_relfatbin",
        "CUDA fatbin wrapper symbol": b"__cudaFatCubin",
    }

    def __init__(self, path: Path, string_limit: int = 100) -> None:
        self.path = path
        self.data = path.read_bytes()
        self.string_limit = string_limit
        self.sections: list[dict[str, Any]] = []
        self.directories: list[tuple[int, int]] = []
        self.pointer_size = 8
        self.machine = 0
        self._parse_headers()

    def _parse_headers(self) -> None:
        if self.data[:2] != b"MZ":
            raise PEError("missing MZ header")
        pe_offset = unpack("<I", self.data, 0x3C)[0]
        if self.data[pe_offset : pe_offset + 4] != b"PE\0\0":
            raise PEError("missing PE signature")

        coff = pe_offset + 4
        self.machine, section_count, timestamp, _, _, optional_size, characteristics = unpack("<HHIIIHH", self.data, coff)
        optional = coff + 20
        magic = unpack("<H", self.data, optional)[0]
        if magic == 0x20B:
            self.pointer_size = 8
            directory_offset = optional + 112
            image_base = unpack("<Q", self.data, optional + 24)[0]
        elif magic == 0x10B:
            self.pointer_size = 4
            directory_offset = optional + 96
            image_base = unpack("<I", self.data, optional + 28)[0]
        else:
            raise PEError(f"unsupported optional-header magic 0x{magic:x}")

        directory_count = unpack("<I", self.data, directory_offset - 4)[0]
        for index in range(min(directory_count, 16)):
            self.directories.append(unpack("<II", self.data, directory_offset + index * 8))

        section_table = optional + optional_size
        for index in range(section_count):
            off = section_table + index * 40
            name, virtual_size, virtual_address, raw_size, raw_offset = unpack("<8sIIII", self.data, off)
            _, _, _, _, flags = unpack("<IIIII", self.data, off + 20)
            section_name = name.split(b"\0", 1)[0].decode("ascii", errors="replace")
            raw = self.data[raw_offset : min(raw_offset + raw_size, len(self.data))]
            self.sections.append(
                {
                    "name": section_name,
                    "virtual_address": virtual_address,
                    "virtual_size": virtual_size,
                    "raw_offset": raw_offset,
                    "raw_size": raw_size,
                    "characteristics": f"0x{flags:08x}",
                    "entropy": round(entropy(raw), 4),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            )

        self.pe_offset = pe_offset
        self.timestamp = timestamp
        self.characteristics = characteristics
        self.image_base = image_base

    def rva_to_offset(self, rva: int) -> int:
        if rva < 0x1000:
            return rva
        for section in self.sections:
            start = section["virtual_address"]
            size = max(section["virtual_size"], section["raw_size"])
            if start <= rva < start + size:
                offset = section["raw_offset"] + rva - start
                if offset < len(self.data):
                    return offset
        raise PEError(f"RVA 0x{rva:x} is not backed by a file section")

    def directory(self, index: int) -> tuple[int, int]:
        return self.directories[index] if index < len(self.directories) else (0, 0)

    def imports(self) -> list[dict[str, Any]]:
        rva, size = self.directory(1)
        if not rva or not size:
            return []
        off = self.rva_to_offset(rva)
        item_size = 20
        result = []
        for pos in range(off, min(off + size, len(self.data)) - item_size + 1, item_size):
            original, stamp, chain, name_rva, first_thunk = unpack("<IIIII", self.data, pos)
            if not (original or stamp or chain or name_rva or first_thunk):
                break
            name = c_string(self.data, self.rva_to_offset(name_rva))
            thunk_rva = original or first_thunk
            thunk_off = self.rva_to_offset(thunk_rva)
            imports: list[str] = []
            max_entries = min((len(self.data) - thunk_off) // self.pointer_size, 65536)
            for index in range(max_entries):
                thunk = unpack("<Q" if self.pointer_size == 8 else "<I", self.data, thunk_off + index * self.pointer_size)[0]
                if thunk == 0:
                    break
                ordinal_mask = 1 << (self.pointer_size * 8 - 1)
                if thunk & ordinal_mask:
                    imports.append(f"#{thunk & 0xFFFF}")
                else:
                    name_off = self.rva_to_offset(thunk)
                    imports.append(c_string(self.data, name_off + 2))
            result.append({"library": name, "symbols": imports})
        return result

    def exports(self) -> list[dict[str, Any]]:
        rva, size = self.directory(0)
        if not rva or size < 40:
            return []
        off = self.rva_to_offset(rva)
        fields = unpack("<IIHHIIIIIII", self.data, off)
        _, ordinal_base, function_count, name_count, funcs_rva, names_rva, ordinals_rva = fields[4:11]
        funcs_off = self.rva_to_offset(funcs_rva) if function_count else 0
        names_off = self.rva_to_offset(names_rva) if name_count else 0
        ords_off = self.rva_to_offset(ordinals_rva) if name_count else 0
        functions: list[dict[str, Any]] = []
        for i in range(min(name_count, 100000)):
            name_rva = unpack("<I", self.data, names_off + 4 * i)[0]
            ordinal_index = unpack("<H", self.data, ords_off + 2 * i)[0]
            name = c_string(self.data, self.rva_to_offset(name_rva))
            ordinal = ordinal_base + ordinal_index
            target_rva = unpack("<I", self.data, funcs_off + 4 * ordinal_index)[0] if ordinal_index < function_count else 0
            functions.append({"name": name, "ordinal": ordinal, "rva": f"0x{target_rva:x}"})
        return functions

    def resources(self) -> list[dict[str, Any]]:
        rva, size = self.directory(2)
        if not rva or not size:
            return []
        root = self.rva_to_offset(rva)
        items: list[dict[str, Any]] = []
        visited: set[int] = set()

        def parse_name(name_field: int, is_type: bool) -> str | int:
            if not (name_field & 0x80000000):
                resource_id = name_field & 0xFFFF
                return self.RESOURCE_TYPES.get(resource_id, resource_id) if is_type else resource_id
            rel = name_field & 0x7FFFFFFF
            name_offset = root + rel
            units = unpack("<H", self.data, name_offset)[0]
            return self.data[name_offset + 2 : name_offset + 2 + units * 2].decode("utf-16le", errors="replace")

        def walk(rel: int, path: list[str | int], depth: int = 0) -> None:
            if depth > 8 or rel in visited:
                return
            visited.add(rel)
            directory_off = root + rel
            named, ids = unpack("<HH", self.data, directory_off + 12)
            count = named + ids
            if count > 10000:
                return
            for index in range(count):
                name_field, target = unpack("<II", self.data, directory_off + 16 + index * 8)
                name = parse_name(name_field, not path)
                new_path = path + [name]
                if target & 0x80000000:
                    walk(target & 0x7FFFFFFF, new_path, depth + 1)
                    continue
                data_entry = root + target
                data_rva, data_size, codepage, _ = unpack("<IIII", self.data, data_entry)
                try:
                    data_off = self.rva_to_offset(data_rva)
                except PEError:
                    data_off = -1
                item: dict[str, Any] = {
                    "path": new_path,
                    "rva": f"0x{data_rva:x}",
                    "file_offset": f"0x{data_off:x}" if data_off >= 0 else None,
                    "size": data_size,
                    "codepage": codepage,
                }
                if (16 in new_path or "VERSIONINFO" in new_path) and data_off >= 0 and data_size <= 4 * 1024 * 1024:
                    item["version_info"] = parse_version_info(self.data[data_off : data_off + data_size])
                items.append(item)

        walk(0, [])
        return items

    def targeted_strings(self) -> list[dict[str, Any]]:
        hits: list[dict[str, Any]] = []
        seen: set[str] = set()
        patterns = (
            (re.compile(rb"[\x20-\x7e]{6,}"), "ascii"),
            (re.compile(rb"(?:[\x20-\x7e]\x00){6,}"), "utf16le"),
        )
        for pattern, encoding in patterns:
            for match in pattern.finditer(self.data):
                raw = match.group(0)
                value = raw.decode("ascii", errors="replace") if encoding == "ascii" else raw[::2].decode("ascii", errors="replace")
                value = value.strip()
                if value and value not in seen and self.STRING_PATTERN.search(value):
                    seen.add(value)
                    hits.append({"offset": f"0x{match.start():x}", "encoding": encoding, "value": value[:1000]})
                    if len(hits) >= self.string_limit:
                        return hits
        return hits

    def section_for_file_offset(self, offset: int) -> str | None:
        for section in self.sections:
            start = section["raw_offset"]
            if start <= offset < start + section["raw_size"]:
                return section["name"]
        return None

    def binary_markers(self) -> list[dict[str, Any]]:
        findings = []
        for label, marker in self.MAGIC_MARKERS.items():
            offsets = []
            cursor = 0
            count = 0
            valid_elf_count = 0
            elf_types: Counter[tuple[int, int, int, int]] = Counter()
            elf_candidates: Counter[tuple[int, int, int, int, int, int]] = Counter()
            while True:
                offset = self.data.find(marker, cursor)
                if offset < 0:
                    break
                count += 1
                if len(offsets) < 20:
                    offsets.append({"file_offset": f"0x{offset:x}", "section": self.section_for_file_offset(offset)})
                if label.startswith("ELF image") and offset + 64 <= len(self.data):
                    header = self.data[offset : offset + 64]
                    elf_class, elf_data = header[4], header[5]
                    if header[6] == 1 and elf_class in (1, 2) and elf_data in (1, 2):
                        endian = "<" if elf_data == 1 else ">"
                        elf_type = struct.unpack_from(endian + "H", header, 16)[0]
                        machine = struct.unpack_from(endian + "H", header, 18)[0]
                        elf_version = struct.unpack_from(endian + "I", header, 20)[0]
                        expected_header_size = 64 if elf_class == 2 else 52
                        header_size = struct.unpack_from(endian + "H", header, 52 if elf_class == 2 else 40)[0]
                        elf_candidates[(elf_class, elf_data, elf_type, machine, elf_version, header_size)] += 1
                        if elf_version == 1 and header_size == expected_header_size:
                            valid_elf_count += 1
                            elf_types[(elf_class, elf_data, elf_type, machine)] += 1
                cursor = offset + 1
            if count:
                finding: dict[str, Any] = {"marker": label, "count": count, "sample_offsets": offsets}
                if label.startswith("ELF"):
                    finding["valid_elf_header_count"] = valid_elf_count
                    finding["elf_header_types"] = [
                        {
                            "elf_class": "ELF64" if key[0] == 2 else "ELF32",
                            "endianness": "little" if key[1] == 1 else "big",
                            "type": key[2],
                            "machine": key[3],
                            "machine_name": "EM_CUDA" if key[3] == 190 else "unknown",
                            "count": value,
                        }
                        for key, value in elf_types.items()
                    ]
                    finding["candidate_header_fields"] = [
                        {
                            "elf_class": "ELF64" if key[0] == 2 else "ELF32",
                            "endianness": "little" if key[1] == 1 else "big",
                            "type": key[2],
                            "machine": key[3],
                            "machine_name": "EM_CUDA" if key[3] == 190 else "unknown",
                            "e_version": key[4],
                            "e_ehsize": key[5],
                            "count": value,
                        }
                        for key, value in elf_candidates.most_common(10)
                    ]
                findings.append(finding)

        for label, pattern in (
            ("PTX .version directive candidate", re.compile(rb"(?<![A-Za-z0-9_.])\.version\s+[0-9]+(?:\.[0-9]+)?")),
            ("PTX .target sm_ directive candidate", re.compile(rb"(?<![A-Za-z0-9_.])\.target\s+sm_[0-9]+")),
            (
                "PTX .entry directive candidate",
                re.compile(rb"(?<![A-Za-z0-9_.])\.entry\s+([A-Za-z_$][A-Za-z0-9_.$]*)\s*\("),
            ),
        ):
            matches = list(pattern.finditer(self.data))
            if matches:
                directives = {
                    (
                        m.group(1)
                        if label.startswith("PTX .entry")
                        else m.group(0).lstrip(b"\x00\r\n \t")
                    ).decode("ascii")
                    for m in matches
                }
                findings.append(
                    {
                        "marker": label,
                        "count": len(matches),
                        "unique_directives": sorted(directives),
                        "sample_offsets": [
                            {"file_offset": f"0x{m.start():x}", "section": self.section_for_file_offset(m.start())}
                            for m in matches[:20]
                        ],
                    }
                )
        return findings

    def report(self) -> dict[str, Any]:
        directories = {}
        for index, (rva, size) in enumerate(self.directories):
            if rva or size:
                directories[self.DIRECTORY_NAMES.get(index, str(index))] = {"rva_or_file_offset": f"0x{rva:x}", "size": size}

        hash_digest = hashlib.sha256(self.data).hexdigest()
        return {
            "path": str(self.path),
            "size_bytes": len(self.data),
            "sha256": hash_digest,
            "format": {
                "machine": f"0x{self.machine:04x}",
                "pe_type": "PE32+" if self.pointer_size == 8 else "PE32",
                "image_base": f"0x{self.image_base:x}",
                "coff_timestamp": self.timestamp,
                "coff_characteristics": f"0x{self.characteristics:04x}",
                "data_directories": directories,
            },
            "sections": self.sections,
            "imports": self.imports(),
            "exports": self.exports(),
            "resources": self.resources(),
            "targeted_strings": self.targeted_strings(),
            "candidate_binary_markers": self.binary_markers(),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="write JSON report to this local path")
    parser.add_argument("--strings-limit", type=int, default=100, help="maximum targeted strings per DLL")
    parser.add_argument("dll", nargs="+", type=Path)
    args = parser.parse_args()

    reports = []
    failed = False
    for path in args.dll:
        try:
            reports.append(PEImage(path, max(0, args.strings_limit)).report())
        except (OSError, PEError, struct.error) as exc:
            failed = True
            reports.append({"path": str(path), "error": str(exc)})

    encoded = json.dumps(reports, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
        print(f"Wrote read-only metadata report for {len(reports)} input(s) to {args.output}")
    else:
        sys.stdout.write(encoded)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
