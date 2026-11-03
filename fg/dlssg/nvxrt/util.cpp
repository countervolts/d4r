#include "util.h"

#include <cctype>
#include <regex>

namespace nvxrt {

// ---------------------------------------------------------------------------------------------- SHA-256
namespace {
const uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98,
    0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8,
    0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819,
    0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
    0xc67178f2};
inline uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
void sha_block(uint32_t h[8], const uint8_t *p)
{
    uint32_t w[64];
    for (int i = 0; i < 16; i++)
        w[i] = (uint32_t)p[4 * i] << 24 | (uint32_t)p[4 * i + 1] << 16 | (uint32_t)p[4 * i + 2] << 8 | p[4 * i + 3];
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
    for (int i = 0; i < 64; i++) {
        uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        uint32_t ch = (e & f) ^ (~e & g);
        uint32_t t1 = hh + S1 + ch + K[i] + w[i];
        uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2 = S0 + mj;
        hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
}
} // namespace

std::string sha256_hex(const void *data, size_t len)
{
    uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    const uint8_t *p = (const uint8_t *)data;
    size_t i = 0;
    for (; i + 64 <= len; i += 64)
        sha_block(h, p + i);
    uint8_t tail[128] = {};
    size_t rem = len - i;
    memcpy(tail, p + i, rem);
    tail[rem] = 0x80;
    size_t tl = rem + 1 + 8 <= 64 ? 64 : 128;
    uint64_t bits = (uint64_t)len * 8;
    for (int k = 0; k < 8; k++)
        tail[tl - 1 - k] = (uint8_t)(bits >> (8 * k));
    sha_block(h, tail);
    if (tl == 128)
        sha_block(h, tail + 64);
    char out[65];
    for (int k = 0; k < 8; k++)
        snprintf(out + 8 * k, 9, "%08x", h[k]);
    return std::string(out, 64);
}

std::string hex(const void *p, size_t n)
{
    static const char *d = "0123456789abcdef";
    std::string s;
    s.reserve(n * 2);
    for (size_t i = 0; i < n; i++) {
        uint8_t b = ((const uint8_t *)p)[i];
        s += d[b >> 4];
        s += d[b & 15];
    }
    return s;
}

// ---------------------------------------------------------------------------------------------- LZ4
bool lz4_block_decode(const uint8_t *src, size_t srclen, std::vector<uint8_t> &dst, size_t dstlen)
{
    dst.clear();
    dst.reserve(dstlen);
    size_t i = 0;
    while (i < srclen) {
        uint8_t token = src[i++];
        size_t lit = token >> 4;
        if (lit == 15) {
            uint8_t b;
            do {
                if (i >= srclen)
                    return false;
                b = src[i++];
                lit += b;
            } while (b == 255);
        }
        if (i + lit > srclen)
            return false;
        dst.insert(dst.end(), src + i, src + i + lit);
        i += lit;
        if (i >= srclen || dst.size() >= dstlen)
            break;
        if (i + 2 > srclen)
            return false;
        size_t off = src[i] | (src[i + 1] << 8);
        i += 2;
        size_t m = token & 15;
        if (m == 15) {
            uint8_t b;
            do {
                if (i >= srclen)
                    return false;
                b = src[i++];
                m += b;
            } while (b == 255);
        }
        m += 4;
        if (off == 0 || off > dst.size())
            return false;
        size_t start = dst.size() - off;
        for (size_t k = 0; k < m; k++)
            dst.push_back(dst[start + k]);
    }
    if (dst.size() > dstlen)
        dst.resize(dstlen);
    return true;
}

// ---------------------------------------------------------------------------------------------- fatbin
namespace {
template <class T> T rd(const uint8_t *p) { T v; memcpy(&v, p, sizeof(T)); return v; }
}

bool parse_cuda_module(const void *data, size_t size, std::vector<FatbinEntry> &entries)
{
    const uint8_t *p = (const uint8_t *)data;
    entries.clear();
    if (size >= 16 && rd<uint32_t>(p) == 0xBA55ED50u) {
        uint16_t hs = rd<uint16_t>(p + 6);
        uint64_t sz = rd<uint64_t>(p + 8);
        size_t end = std::min<size_t>(size, hs + sz);
        size_t o = hs;
        while (o + 64 <= end) {
            uint16_t kind = rd<uint16_t>(p + o);
            uint32_t eh = rd<uint32_t>(p + o + 4);
            uint64_t esz = rd<uint64_t>(p + o + 8);
            if (eh == 0)
                break;
            uint32_t arch = rd<uint32_t>(p + o + 28);
            uint64_t flags = rd<uint64_t>(p + o + 40);
            uint64_t dec = eh >= 64 ? rd<uint64_t>(p + o + 56) : 0;
            if (o + eh + esz > end)
                break;
            FatbinEntry e;
            e.kind = kind;
            e.sm = arch;
            const uint8_t *payload = p + o + eh;
            if ((flags & 0x2000) && dec) {
                if (!lz4_block_decode(payload, esz, e.data, dec))
                    e.data.clear();
            } else {
                e.data.assign(payload, payload + esz);
            }
            if (kind == 1) {
                while (!e.data.empty() && e.data.back() == 0)
                    e.data.pop_back();
            }
            if (!e.data.empty())
                entries.push_back(std::move(e));
            o += eh + esz;
        }
        return !entries.empty();
    }
    if (size >= 4 && memcmp(p, "\x7f" "ELF", 4) == 0) {
        FatbinEntry e;
        e.kind = 2;
        uint32_t flags = size >= 52 ? rd<uint32_t>(p + 48) : 0;
        e.sm = flags & 0xff; // low byte of e_flags holds the SM (e.g. 0x59 = 89)
        e.data.assign(p, p + size);
        entries.push_back(std::move(e));
        return true;
    }
    // PTX text?
    std::string s((const char *)p, strnlen((const char *)p, size));
    if (s.find(".version") != std::string::npos && s.find(".entry") != std::string::npos) {
        FatbinEntry e;
        e.kind = 1;
        std::smatch m;
        static const std::regex tre("\\.target\\s+sm_(\\d+)");
        if (std::regex_search(s, m, tre))
            e.sm = (uint32_t)std::stoul(m[1].str());
        e.data.assign(s.begin(), s.end());
        entries.push_back(std::move(e));
        return true;
    }
    return false;
}

std::string normalize_ptx(const std::string &ptx)
{
    std::string out;
    out.reserve(ptx.size());
    size_t i = 0, n = ptx.size();
    bool pending_space = false;
    while (i < n) {
        char c = ptx[i];
        if (c == '/' && i + 1 < n && ptx[i + 1] == '/') {
            while (i < n && ptx[i] != '\n')
                i++;
            pending_space = true;
            continue;
        }
        if (c == '/' && i + 1 < n && ptx[i + 1] == '*') {
            i += 2;
            while (i + 1 < n && !(ptx[i] == '*' && ptx[i + 1] == '/'))
                i++;
            i += 2;
            pending_space = true;
            continue;
        }
        if (c == '\r' || c == '\n' || c == ' ' || c == '\t' || c == '\0') {
            pending_space = true;
            i++;
            continue;
        }
        if (pending_space && !out.empty())
            out += ' ';
        pending_space = false;
        out += c;
        i++;
    }
    return out;
}

std::vector<std::string> ptx_entries(const std::string &ptx)
{
    std::vector<std::string> names;
    static const std::regex re("\\.entry\\s+([A-Za-z_$][\\w$]*)");
    for (std::sregex_iterator it(ptx.begin(), ptx.end(), re), e; it != e; ++it)
        names.push_back((*it)[1].str());
    return names;
}

std::vector<std::string> elf_kernel_names(const uint8_t *b, size_t size)
{
    std::vector<std::string> names;
    if (size < 64 || memcmp(b, "\x7f" "ELF", 4) != 0 || b[4] != 2)
        return names;
    uint64_t shoff = rd<uint64_t>(b + 40);
    uint16_t shentsize = rd<uint16_t>(b + 58), shnum = rd<uint16_t>(b + 60), shstrndx = rd<uint16_t>(b + 62);
    if (shoff + (uint64_t)shnum * shentsize > size || shstrndx >= shnum)
        return names;
    const uint8_t *sh = b + shoff + (uint64_t)shstrndx * shentsize;
    uint64_t stroff = rd<uint64_t>(sh + 24);
    for (int k = 0; k < shnum; k++) {
        const uint8_t *s = b + shoff + (uint64_t)k * shentsize;
        uint32_t nm = rd<uint32_t>(s);
        if (stroff + nm >= size)
            continue;
        std::string name((const char *)b + stroff + nm, strnlen((const char *)b + stroff + nm, size - stroff - nm));
        if (name.rfind(".text.", 0) == 0)
            names.push_back(name.substr(6));
    }
    return names;
}

} // namespace nvxrt
