// nvxrt utilities: SHA-256, LZ4 block decode, CUDA fatbin/cubin parsing, PTX normalisation.
// No NVIDIA code or data here; formats follow Mesa's MIT nv_fatbin.c and public descriptions.
#pragma once
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace nvxrt {

std::string sha256_hex(const void *data, size_t len);

// LZ4 block format (as used for compressed fatbin entries). Returns false on malformed input.
bool lz4_block_decode(const uint8_t *src, size_t srclen, std::vector<uint8_t> &dst, size_t dstlen);

struct FatbinEntry {
    uint16_t kind = 0;   // 1 = PTX, 2 = ELF (SASS)
    uint32_t sm = 0;
    std::vector<uint8_t> data; // decompressed payload
};

// data may be a fatbin (magic 0xBA55ED50), a bare CUDA ELF, or PTX text.
// Fills entries; returns false if nothing usable was recognised.
bool parse_cuda_module(const void *data, size_t size, std::vector<FatbinEntry> &entries);

// Strip comments, CR, collapse whitespace: the identity used for the kernel cache
// (Windows and Linux builds of the DLL differ only in CRLF).
std::string normalize_ptx(const std::string &ptx);

// Entry names (.entry NAME) declared in a PTX module.
std::vector<std::string> ptx_entries(const std::string &ptx);

// Kernel names in a CUDA ELF (.text.NAME sections).
std::vector<std::string> elf_kernel_names(const uint8_t *data, size_t size);

std::string hex(const void *p, size_t n);

} // namespace nvxrt
