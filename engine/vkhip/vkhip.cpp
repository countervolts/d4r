// vkhip: see vkhip.h.
#include "vkhip.h"
#include "donor_spv.h"
#include <algorithm>
#include <bit>
#include <cstdio>

namespace vkhip {
namespace {

template <class T> T rd(const std::vector<uint8_t>& b, size_t at) {
    T v{};
    if (at + sizeof(T) <= b.size()) std::memcpy(&v, b.data() + at, sizeof(T));
    return v;
}
template <class T> void wr(std::vector<uint8_t>& b, size_t at, T v) { std::memcpy(b.data() + at, &v, sizeof(T)); }

struct Section { std::string name; uint32_t type = 0; uint64_t flags = 0, addr = 0, offset = 0, size = 0; uint32_t link = 0; };

constexpr uint32_t SHT_NOBITS = 8, SHT_SYMTAB = 2, SHT_DYNSYM = 11;
constexpr uint64_t SHF_WRITE = 1, SHF_ALLOC = 2;

// gfx11 encodings (gfx12 encodes these the same, as s_add_co_u32 / s_add_co_ci_u32)
constexpr uint32_t GETPC_MASK = 0xff80ffffu, GETPC = 0xbe804700u;  // s_getpc_b64 s[n:n+1]
constexpr uint32_t S_SEXT_I32_I16 = 0xbe800f00u;  // | sdst << 16 | ssrc0: gfx12 sign-extends the 48-bit PC after s_getpc
constexpr uint32_t S_WAIT_ALU_MASK = 0xffff0000u, S_WAIT_ALU = 0xbf880000u;  // gfx12 s_wait_alu (SALU dependency wait)
constexpr uint32_t SOP2_OP_MASK = 0xff800000u, S_ADD_U32 = 0x80000000u, S_ADDC_U32 = 0x82000000u;
constexpr uint32_t S_MOV_B32_LIT = 0xbe8000ffu;  // | sdst << 16
constexpr uint32_t S_NOP = 0xbf800000u, S_BRANCH = 0xbfa00000u, S_CODE_END = 0xbf9f0000u;
// gfx10.3 (RDNA2) numbers three of them differently: s_getpc_b64 SOP1 0x1f, s_mov_b32 SOP1 3, s_branch SOPP 2 (the SOP2
// adds, s_nop and s_code_end are the same)
constexpr uint32_t GFX10_GETPC = 0xbe801f00u, GFX10_S_MOV_B32_LIT = 0xbe8003ffu, GFX10_S_BRANCH = 0xbf820000u;
bool gfx10(uint32_t mach) { return (mach >= 0x36 && mach <= 0x39) || mach == 0x3d || mach == 0x3e || mach == 0x45; }

// s_add_u32/s_addc_u32 sN, sN, literal (either operand order)
bool sop2Literal(uint32_t w, uint32_t op, uint32_t reg) {
    if ((w & SOP2_OP_MASK) != op || ((w >> 16) & 0x7f) != reg) return false;
    const uint32_t s0 = w & 0xff, s1 = (w >> 8) & 0xff;
    return (s0 == reg && s1 == 0xff) || (s0 == 0xff && s1 == reg);
}

} // namespace

uint32_t gfx10Mach(uint32_t gfxTargetVersion) {
    switch (gfxTargetVersion) {
    case 100300: return 0x36;  // gfx1030
    case 100301: return 0x37;  // gfx1031
    case 100302: return 0x38;  // gfx1032
    case 100303: return 0x39;  // gfx1033
    case 100304: return 0x3e;  // gfx1034
    case 100305: return 0x3d;  // gfx1035
    case 100306: return 0x45;  // gfx1036
    default: return 0;
    }
}

uint32_t gfx11Mach(uint32_t gfxTargetVersion) {
    switch (gfxTargetVersion) {
    case 110000: return 0x41;  // gfx1100
    case 110001: return 0x46;  // gfx1101
    case 110002: return 0x47;  // gfx1102
    case 110003: return 0x44;  // gfx1103
    default: return 0;
    }
}

uint32_t gfx12Mach(uint32_t gfxTargetVersion) {
    switch (gfxTargetVersion) {
    case 120000: return 0x48;  // gfx1200
    case 120001: return 0x4e;  // gfx1201
    default: return 0;
    }
}

bool Module::parse(const std::vector<uint8_t>& f) {
    mach = f.size() >= 64 ? rd<uint32_t>(f, 48) & 255 : 0;
    if (f.size() < 64 || std::memcmp(f.data(), "\x7f" "ELF", 4) || f[4] != 2 || f[5] != 1 ||
        rd<uint16_t>(f, 18) != 224 || (mach != 0x41 && mach != 0x46 && mach != 0x47 && mach != 0x44 &&
                                       mach != 0x48 && mach != 0x4e && !gfx10(mach))) {
        error = "requires a little-endian gfx1030-gfx1036, gfx1100-gfx1103 or gfx1200/gfx1201 AMDGPU ELF64 code object";
        return false;
    }
    const bool gfx12 = mach == 0x48 || mach == 0x4e;
    const uint64_t shoff = rd<uint64_t>(f, 0x28);
    const uint16_t shentsize = rd<uint16_t>(f, 0x3a), shnum = rd<uint16_t>(f, 0x3c), shstrndx = rd<uint16_t>(f, 0x3e);
    if (shentsize != 64 || shoff > f.size() || uint64_t(shnum) * 64 > f.size() - shoff || shstrndx >= shnum) { error = "bad section table"; return false; }
    std::vector<Section> s(shnum);
    for (uint32_t i = 0; i < shnum; ++i) {
        const size_t h = shoff + 64 * i;
        s[i].type = rd<uint32_t>(f, h + 4); s[i].flags = rd<uint64_t>(f, h + 8); s[i].addr = rd<uint64_t>(f, h + 16);
        s[i].offset = rd<uint64_t>(f, h + 24); s[i].size = rd<uint64_t>(f, h + 32); s[i].link = rd<uint32_t>(f, h + 40);
        if (s[i].link >= shnum || (s[i].type != SHT_NOBITS &&
            (s[i].offset > f.size() || s[i].size > f.size() - s[i].offset))) {
            error = "section outside code object"; return false;
        }
    }
    auto str = [&](uint32_t table, uint32_t at) {
        const uint64_t o = s[table].offset + at;
        std::string r;
        while (o + r.size() < f.size() && f[o + r.size()]) r += char(f[o + r.size()]);
        return r;
    };
    for (uint32_t i = 0; i < shnum; ++i) s[i].name = str(shstrndx, rd<uint32_t>(f, shoff + 64 * i));
    int text = -1, symtab = -1;
    dataLo = ~0ull; dataHi = 0;
    for (uint32_t i = 0; i < shnum; ++i) {
        if (s[i].name == ".text") text = int(i);
        if (s[i].type == SHT_SYMTAB || (s[i].type == SHT_DYNSYM && symtab < 0)) symtab = int(i);
        // .rodata holds __constant__ tables (read PC-relative like the globals) besides the kernel descriptors
        const bool data = s[i].name.rfind(".data", 0) == 0 || s[i].name.rfind(".bss", 0) == 0;
        const bool rodata = s[i].name.rfind(".rodata", 0) == 0;
        if (((data && (s[i].flags & SHF_WRITE)) || rodata) && (s[i].flags & SHF_ALLOC) && s[i].size) {
            dataLo = std::min(dataLo, s[i].addr); dataHi = std::max(dataHi, s[i].addr + s[i].size);
        }
    }
    if (text < 0 || symtab < 0) { error = "no .text or symbol table"; return false; }
    if (s[text].offset + s[text].size > f.size() || s[text].size % 4) { error = "bad .text"; return false; }
    textVa = s[text].addr;
    this->text.assign(f.begin() + s[text].offset, f.begin() + s[text].offset + s[text].size);
    if (dataHi > dataLo) {
        if (dataHi - dataLo > 256 * 1024 * 1024) { error = "code object data allocation exceeds limit"; return false; }
        dataInit.assign(dataHi - dataLo, 0);
        for (const auto& x : s) {
            if (x.type == SHT_NOBITS || !x.size || x.addr < dataLo || x.addr + x.size > dataHi) continue;
            if (!(x.name.rfind(".data", 0) == 0) && !(x.name.rfind(".rodata", 0) == 0)) continue;
            std::memcpy(dataInit.data() + (x.addr - dataLo), f.data() + x.offset, x.size);
        }
    } else {
        dataLo = dataHi = 0;
    }
    // symbols: kernel descriptors (<name>.kd) and data objects
    const auto& st = s[symtab];
    for (uint64_t at = st.offset; at + 24 <= st.offset + st.size; at += 24) {
        const std::string name = str(st.link, rd<uint32_t>(f, at));
        const uint8_t info = f[at + 4];
        const uint16_t shndx = rd<uint16_t>(f, at + 6);
        const uint64_t value = rd<uint64_t>(f, at + 8), size = rd<uint64_t>(f, at + 16);
        if (name.empty() || shndx == 0 || shndx >= shnum) continue;
        if (name.size() > 3 && name.compare(name.size() - 3, 3, ".kd") == 0 && size == 64) {
            const uint64_t o = s[shndx].offset + (value - s[shndx].addr);
            if (o + 64 > f.size()) continue;
            Kernel k;
            k.name = name.substr(0, name.size() - 3);
            k.lds = rd<uint32_t>(f, o); k.scratch = rd<uint32_t>(f, o + 4); k.kernarg = rd<uint32_t>(f, o + 8);
            k.entry = value + uint64_t(rd<int64_t>(f, o + 16));
            k.rsrc3 = rd<uint32_t>(f, o + 44); k.rsrc1 = rd<uint32_t>(f, o + 48); k.rsrc2 = rd<uint32_t>(f, o + 52);
            k.props = rd<uint16_t>(f, o + 56); k.preload = rd<uint16_t>(f, o + 58);
            kernels[k.name] = k;
        } else if ((info & 0xf) == 1 /* STT_OBJECT */) {
            globals[name] = Global{value, size};
        }
    }
    // PC-relative global addressing
    const uint32_t* w = reinterpret_cast<const uint32_t*>(this->text.data());
    const size_t n = this->text.size() / 4;
    for (size_t i = 0; i + 4 < n; ++i) {
        if ((w[i] & GETPC_MASK) != (gfx10(mach) ? GFX10_GETPC : GETPC)) continue;
        const uint32_t reg = (w[i] >> 16) & 0x7f;
        // gfx12: s_getpc_b64 s[n:n+1]; s_sext_i32_i16 s[n+1], s[n+1]; then the same add pair, with s_wait_alu
        // between any of them (the span is then longer than gfx11's five words)
        size_t j = i + 1;
        auto waits = [&] { while (gfx12 && j < n && (w[j] & S_WAIT_ALU_MASK) == S_WAIT_ALU) ++j; };
        waits();
        if (gfx12 && j < n && w[j] == (S_SEXT_I32_I16 | ((reg + 1) << 16) | (reg + 1))) { ++j; waits(); }
        const size_t add = j;
        bool ok = j + 1 < n && sop2Literal(w[j], S_ADD_U32, reg);
        j += 2; waits();
        const size_t addc = j;
        ok = ok && j + 1 < n && sop2Literal(w[j], S_ADDC_U32, reg + 1);
        if (!ok) {
            error = "unhandled s_getpc_b64 at text+" + std::to_string(4 * i);
            return false;
        }
        const int64_t offset = int64_t((uint64_t(w[addc + 1]) << 32) | w[add + 1]);
        const uint64_t target = textVa + 4 * (i + 1) + uint64_t(offset);
        // code first: with .rodata in the data image its range can span .text
        if (target >= textVa && target < textVa + this->text.size()) {
            ++textGetpc;
        } else if (target >= dataLo && target < dataHi) {
            sites.push_back(Site{uint32_t(i), reg, target, uint32_t(addc + 2 - i)});
        } else {
            error = "s_getpc_b64 at text+" + std::to_string(4 * i) + " addresses neither code nor data";
            return false;
        }
        i = addc + 1;
    }
    for (auto& [name, k] : kernels) {
        if (k.entry < textVa || k.entry >= textVa + this->text.size() || (k.entry - textVa) % 4) { error = name + ": entry outside .text"; return false; }
        // Only the kernarg pointer (s[0:1]) is enabled and nothing is preloaded: any further user SGPRs are
        // LLVM's UserSGPRInit16Bug padding (gfx1100/gfx1102), unread, and the kernel's rsrc2 places its
        // workgroup ids after them.
        // gfx10 has no architected flat scratch: LLVM always enables the private segment buffer (s[0:3]) before the
        // kernarg pointer, which then arrives in s[4:5]. Without scratch the kernel never reads s[0:3].
        const uint32_t user = (k.rsrc2 >> 1) & 0x1f;
        const uint32_t ids = uint32_t(std::popcount((k.rsrc2 >> 7) & 0xfu));  // tgid x/y/z, tg size (not the wave offset)
        const bool segmentBuffer = gfx10(mach) && (k.props & 0x7f) == ((1u << 0) | (1u << 3)) && !k.scratch;
        k.kernargSgpr = segmentBuffer ? 4 : 0;
        if (((k.props & 0x7f) != (1u << 3) && !segmentBuffer) || (k.preload & 0x7f) || user < k.kernargSgpr + 2 ||
            (user > k.kernargSgpr + 2 && user + ids != 16)) {
            error = name + ": not a kernarg-only kernel"; return false;
        }
        if (!(k.props & (1u << 10))) { error = name + ": not wave32"; return false; }
    }
    return true;
}

std::vector<uint32_t> Module::patchedText(uint64_t dataAddress) const {
    std::vector<uint32_t> w(text.size() / 4);
    std::memcpy(w.data(), text.data(), text.size());
    for (const auto& site : sites) {
        const uint64_t a = dataAddress + (site.target - dataLo);
        const uint32_t mov = gfx10(mach) ? GFX10_S_MOV_B32_LIT : S_MOV_B32_LIT;
        w[site.word + 0] = mov | (site.sdst << 16);
        w[site.word + 1] = uint32_t(a);
        w[site.word + 2] = mov | ((site.sdst + 1) << 16);
        w[site.word + 3] = uint32_t(a >> 32);
        for (uint32_t k = 4; k < site.words; ++k) w[site.word + k] = S_NOP;
    }
    return w;
}

bool Device::load(VkDevice d, PFN_vkGetDeviceProcAddr proc, bool binaries) {
    device = d;
    createCache = reinterpret_cast<PFN_vkCreatePipelineCache>(proc(d, "vkCreatePipelineCache"));
    destroyCache = reinterpret_cast<PFN_vkDestroyPipelineCache>(proc(d, "vkDestroyPipelineCache"));
    cacheData = reinterpret_cast<PFN_vkGetPipelineCacheData>(proc(d, "vkGetPipelineCacheData"));
    createCompute = reinterpret_cast<PFN_vkCreateComputePipelines>(proc(d, "vkCreateComputePipelines"));
    destroyPipeline = reinterpret_cast<PFN_vkDestroyPipeline>(proc(d, "vkDestroyPipeline"));
    createModule = reinterpret_cast<PFN_vkCreateShaderModule>(proc(d, "vkCreateShaderModule"));
    destroyModule = reinterpret_cast<PFN_vkDestroyShaderModule>(proc(d, "vkDestroyShaderModule"));
    pipelineBinary = binaries;
    if (binaries) {
        createBinaries = reinterpret_cast<PFN_vkCreatePipelineBinariesKHR>(proc(d, "vkCreatePipelineBinariesKHR"));
        binaryData = reinterpret_cast<PFN_vkGetPipelineBinaryDataKHR>(proc(d, "vkGetPipelineBinaryDataKHR"));
        destroyBinary = reinterpret_cast<PFN_vkDestroyPipelineBinaryKHR>(proc(d, "vkDestroyPipelineBinaryKHR"));
        if (!createBinaries || !binaryData || !destroyBinary) return false;
    }
    return createCache && destroyCache && cacheData && createCompute && destroyPipeline && createModule && destroyModule;
}

namespace {

// radv_shader_binary: u32 type | ac_shader_config (56 bytes) | radv_shader_info | u32 total_size, then
// radv_shader_binary_legacy's code/exec/ir/disasm/stats/debug-info sizes and the data (stats, code, ir, disasm,
// debug info). Returns the offset of the legacy data (the end of the fixed header) or 0.
size_t legacyHeader(const std::vector<uint8_t>& b, size_t& totalAt, size_t& sizes) {
    if (b.size() < 128 || rd<uint32_t>(b, 0) != 0 /* RADV_BINARY_TYPE_LEGACY */) return 0;
    for (size_t p = 64; p + 32 <= b.size(); p += 4) {
        const uint32_t total = rd<uint32_t>(b, p);
        if (total != b.size()) continue;
        // radv_shader_binary is 8-byte aligned: the legacy sizes follow total_size directly or after 4 bytes
        for (size_t pad : {size_t(0), size_t(4)}) {
            const size_t q = p + 4 + pad;
            if (q + 24 > b.size()) continue;
            const uint32_t code = rd<uint32_t>(b, q), exec = rd<uint32_t>(b, q + 4), ir = rd<uint32_t>(b, q + 8),
                           disasm = rd<uint32_t>(b, q + 12), stats = rd<uint32_t>(b, q + 16), dbg = rd<uint32_t>(b, q + 20);
            if (!code || code % 4 || exec > code) continue;
            if (uint64_t(q) + 24 + code + ir + disasm + stats + dbg != total) continue;
            totalAt = p; sizes = q;
            return q + 24;
        }
    }
    return 0;
}

constexpr size_t CFG = 4;  // ac_shader_config offset
enum CfgField { NUM_SGPRS = 0, NUM_VGPRS = 4, SHARED_VGPRS = 8, SPILLED_SGPRS = 12, SPILLED_VGPRS = 16, LDS_SIZE = 20,
                FLOAT_MODE = 32, SCRATCH = 36, WGP_MODE = 40, RSRC1 = 44, RSRC2 = 48, RSRC3 = 52 };
constexpr uint32_t RSRC2_LDS_MASK = 0x1ffu << 15;

// The code of a patched shader: s_branch to the entry, padding to 256 bytes, .text (its 256-byte alignment kept),
// then the s_code_end tail the hardware's instruction prefetch may read. Empty when the entry is out of range.
std::vector<uint32_t> injectedCode(const Module& module, const Kernel& kernel, uint64_t dataAddress) {
    std::vector<uint32_t> code(64, S_NOP);
    const int64_t simm = int64_t(64 + (kernel.entry - module.textVa) / 4) - 1;
    if (simm > 32767) return {};
    code[0] = (gfx10(module.mach) ? GFX10_S_BRANCH : S_BRANCH) | uint32_t(simm & 0xffff);
    const auto text = module.patchedText(dataAddress);
    code.insert(code.end(), text.begin(), text.end());
    code.insert(code.end(), 64, S_CODE_END);
    return code;
}

// --- AMD's PAL drivers: the donor's pipeline binary (an AMDPAL ELF) ----------------------------------------------

// A msgpack value: the AMDGPU metadata note is one map (amdpal.pipelines, amdpal.version).
struct Mp {
    enum Type { Nil, Bool, Uint, Int, F32, F64, Str, Bin, Arr, Map } type = Nil;
    uint64_t u = 0; int64_t i = 0; double f = 0;
    std::string s;  // Str and Bin
    std::vector<Mp> a;
    std::vector<std::pair<Mp, Mp>> m;
    static Mp uint(uint64_t v) { Mp r; r.type = Uint; r.u = v; return r; }
    static Mp boolean(bool v) { Mp r; r.type = Bool; r.u = v; return r; }
    Mp* at(const char* key) {
        if (type != Map) return nullptr;
        for (auto& [k, v] : m) if (k.type == Str && k.s == key) return &v;
        return nullptr;
    }
    // a map's entry, appended when missing
    Mp& set(const char* key, Mp value) {
        if (Mp* v = at(key)) return *v = std::move(value);
        Mp k; k.type = Str; k.s = key;
        m.emplace_back(std::move(k), std::move(value));
        return m.back().second;
    }
};

bool mpRead(const uint8_t*& p, const uint8_t* end, Mp& v, int depth = 0) {
    if (p >= end || depth > 32) return false;
    const uint8_t c = *p++;
    auto be = [&](unsigned n, uint64_t& out) {
        if (size_t(end - p) < n) return false;
        out = 0;
        for (unsigned k = 0; k < n; ++k) out = out << 8 | *p++;
        return true;
    };
    auto bytes = [&](uint64_t n, Mp::Type t) {
        if (uint64_t(end - p) < n) return false;
        v.type = t; v.s.assign(reinterpret_cast<const char*>(p), size_t(n)); p += n;
        return true;
    };
    auto items = [&](uint64_t n, bool map) {
        if (n > uint64_t(end - p)) return false;  // every item takes at least one byte
        v.type = map ? Mp::Map : Mp::Arr;
        for (uint64_t k = 0; k < n; ++k) {
            if (map) {
                v.m.emplace_back();
                if (!mpRead(p, end, v.m.back().first, depth + 1) || !mpRead(p, end, v.m.back().second, depth + 1)) return false;
            } else {
                v.a.emplace_back();
                if (!mpRead(p, end, v.a.back(), depth + 1)) return false;
            }
        }
        return true;
    };
    uint64_t n = 0;
    if (c <= 0x7f) { v.type = Mp::Uint; v.u = c; return true; }
    if (c >= 0xe0) { v.type = Mp::Int; v.i = int8_t(c); return true; }
    if ((c & 0xf0) == 0x80) return items(c & 15, true);
    if ((c & 0xf0) == 0x90) return items(c & 15, false);
    if ((c & 0xe0) == 0xa0) return bytes(c & 31, Mp::Str);
    switch (c) {
    case 0xc0: v.type = Mp::Nil; return true;
    case 0xc2: case 0xc3: v.type = Mp::Bool; v.u = c == 0xc3; return true;
    case 0xc4: case 0xc5: case 0xc6: return be(1u << (c - 0xc4), n) && bytes(n, Mp::Bin);
    case 0xca: { if (!be(4, n)) return false; float x; uint32_t w = uint32_t(n); std::memcpy(&x, &w, 4); v.type = Mp::F32; v.f = x; return true; }
    case 0xcb: { if (!be(8, n)) return false; std::memcpy(&v.f, &n, 8); v.type = Mp::F64; return true; }
    case 0xcc: case 0xcd: case 0xce: case 0xcf: v.type = Mp::Uint; return be(1u << (c - 0xcc), v.u);
    case 0xd0: case 0xd1: case 0xd2: case 0xd3: {
        const unsigned bits = 8u << (c - 0xd0);
        if (!be(bits / 8, n)) return false;
        v.type = Mp::Int; v.i = bits == 64 ? int64_t(n) : int64_t(n << (64 - bits)) >> (64 - bits);
        return true;
    }
    case 0xd9: case 0xda: case 0xdb: return be(1u << (c - 0xd9), n) && bytes(n, Mp::Str);
    case 0xdc: case 0xdd: return be(c == 0xdc ? 2 : 4, n) && items(n, false);
    case 0xde: case 0xdf: return be(c == 0xde ? 2 : 4, n) && items(n, true);
    default: return false;  // ext types: not in AMDGPU metadata
    }
}

void mpWrite(std::vector<uint8_t>& o, const Mp& v) {
    auto be = [&](uint8_t tag, unsigned n, uint64_t x) {
        o.push_back(tag);
        for (unsigned k = n; k-- > 0;) o.push_back(uint8_t(x >> (8 * k)));
    };
    auto length = [&](uint64_t n, uint8_t fix, unsigned fixBits, uint8_t tag8, uint8_t tag16, uint8_t tag32) {
        if (fixBits && n < (uint64_t(1) << fixBits)) o.push_back(uint8_t(fix | n));
        else if (tag8 && n <= 0xff) be(tag8, 1, n);
        else if (n <= 0xffff) be(tag16, 2, n);
        else be(tag32, 4, n);
    };
    switch (v.type) {
    case Mp::Nil: o.push_back(0xc0); break;
    case Mp::Bool: o.push_back(v.u ? 0xc3 : 0xc2); break;
    case Mp::Uint:
        if (v.u <= 0x7f) o.push_back(uint8_t(v.u));
        else if (v.u <= 0xff) be(0xcc, 1, v.u);
        else if (v.u <= 0xffff) be(0xcd, 2, v.u);
        else if (v.u <= 0xffffffffu) be(0xce, 4, v.u);
        else be(0xcf, 8, v.u);
        break;
    case Mp::Int:
        if (v.i >= 0) { Mp u = Mp::uint(uint64_t(v.i)); mpWrite(o, u); }
        else if (v.i >= -32) o.push_back(uint8_t(v.i));
        else if (v.i >= -128) be(0xd0, 1, uint64_t(v.i) & 0xff);
        else if (v.i >= -32768) be(0xd1, 2, uint64_t(v.i) & 0xffff);
        else if (v.i >= INT32_MIN) be(0xd2, 4, uint64_t(v.i) & 0xffffffffu);
        else be(0xd3, 8, uint64_t(v.i));
        break;
    case Mp::F32: { const float x = float(v.f); uint32_t w; std::memcpy(&w, &x, 4); be(0xca, 4, w); break; }
    case Mp::F64: { uint64_t w; std::memcpy(&w, &v.f, 8); be(0xcb, 8, w); break; }
    case Mp::Str: length(v.s.size(), 0xa0, 5, 0xd9, 0xda, 0xdb); o.insert(o.end(), v.s.begin(), v.s.end()); break;
    case Mp::Bin: length(v.s.size(), 0, 0, 0xc4, 0xc5, 0xc6); o.insert(o.end(), v.s.begin(), v.s.end()); break;
    case Mp::Arr: length(v.a.size(), 0x90, 4, 0, 0xdc, 0xdd); for (const auto& x : v.a) mpWrite(o, x); break;
    case Mp::Map: length(v.m.size(), 0x80, 4, 0, 0xde, 0xdf); for (const auto& [k, x] : v.m) { mpWrite(o, k); mpWrite(o, x); } break;
    }
}

constexpr uint8_t ELFOSABI_AMDGPU_PAL = 0x41;
constexpr uint32_t NT_AMDGPU_METADATA = 32;
constexpr uint64_t PAL_GLOBAL_TABLE = 0x10000000;  // user data map values from here on are PAL's own entries

// The donor's AMDPAL ELF rebuilt around the kernel: .text replaced, the compute stage's metadata set from the kernel
// descriptor, the push constant moved to s[0:1]. Empty and `why` set when the donor is not what this expects.
std::vector<uint8_t> palBinary(const std::vector<uint8_t>& f, const Module& module, const Kernel& kernel,
                               uint64_t dataAddress, std::string& why) {
    if (f.size() < 64 || std::memcmp(f.data(), "\x7f" "ELF", 4) || f[4] != 2 || f[5] != 1 ||
        f[7] != ELFOSABI_AMDGPU_PAL || rd<uint16_t>(f, 18) != 224) {
        why = "the driver's pipeline binary is not an AMDPAL ELF (not AMD's driver?)"; return {};
    }
    // the donor was compiled for the GPU the device runs on
    if ((rd<uint32_t>(f, 48) & 255) != module.mach) {
        char text[96];
        std::snprintf(text, sizeof(text), "code objects for EF_AMDGPU_MACH %#x, the GPU is %#x (rebuild the engine models)",
                      module.mach, rd<uint32_t>(f, 48) & 255);
        why = text; return {};
    }
    const uint64_t shoff = rd<uint64_t>(f, 0x28);
    const uint16_t shnum = rd<uint16_t>(f, 0x3c);
    if (rd<uint16_t>(f, 0x3a) != 64 || shoff > f.size() || uint64_t(shnum) * 64 > f.size() - shoff) { why = "bad donor section table"; return {}; }
    // the notes (ISA name, metadata) and the symbols other than the entry are the donor's; nothing else is kept
    std::vector<uint8_t> notes;
    Mp meta;
    bool haveMeta = false;
    for (uint32_t i = 0; i < shnum; ++i) {
        const size_t h = shoff + 64 * i;
        const uint64_t offset = rd<uint64_t>(f, h + 24), size = rd<uint64_t>(f, h + 32);
        const uint32_t type = rd<uint32_t>(f, h + 4);
        if (type == 4 /* SHT_RELA */ || type == 9 /* SHT_REL */) { why = "donor binary has relocations"; return {}; }
        if (type != 7 /* SHT_NOTE */) continue;
        if (offset > f.size() || size > f.size() - offset) { why = "bad donor note"; return {}; }
        for (uint64_t at = offset; at + 12 <= offset + size;) {
            const uint32_t nameSize = rd<uint32_t>(f, at), descSize = rd<uint32_t>(f, at + 4), noteType = rd<uint32_t>(f, at + 8);
            const uint64_t nameAt = at + 12, descAt = nameAt + ((nameSize + 3) & ~3u), next = descAt + ((descSize + 3) & ~3u);
            if (next > offset + size) { why = "bad donor note"; return {}; }
            if (noteType == NT_AMDGPU_METADATA && nameSize == 7 && !std::memcmp(f.data() + nameAt, "AMDGPU", 7)) {
                const uint8_t* p = f.data() + descAt;
                if (!mpRead(p, p + descSize, meta)) { why = "unreadable donor metadata"; return {}; }
                haveMeta = true;
            } else {
                notes.insert(notes.end(), f.begin() + at, f.begin() + next);
            }
            at = next;
        }
    }
    Mp* pipelines = haveMeta ? meta.at("amdpal.pipelines") : nullptr;
    Mp* pipeline = pipelines && pipelines->type == Mp::Arr && pipelines->a.size() == 1 ? &pipelines->a[0] : nullptr;
    Mp* stages = pipeline ? pipeline->at(".hardware_stages") : nullptr;
    Mp* cs = stages ? stages->at(".cs") : nullptr;
    Mp* regs = pipeline ? pipeline->at(".compute_registers") : nullptr;
    Mp* map = cs ? cs->at(".user_data_reg_map") : nullptr;
    Mp* lds = cs ? cs->at(".lds_size") : nullptr;
    Mp* symbol = cs ? cs->at(".entry_point_symbol") : nullptr;
    if (!cs || !regs || regs->type != Mp::Map || !map || map->type != Mp::Arr || !lds || lds->type != Mp::Uint ||
        !symbol || symbol->type != Mp::Str) {
        why = "unrecognized donor metadata (driver changed?)"; return {};
    }
    // the push constant: the donor's only API user data, two consecutive entries
    size_t sgpr = 0;
    while (sgpr < map->a.size() && !(map->a[sgpr].type == Mp::Uint && map->a[sgpr].u < PAL_GLOBAL_TABLE)) ++sgpr;
    if (sgpr + 1 >= map->a.size() || map->a[sgpr + 1].type != Mp::Uint || map->a[sgpr + 1].u != map->a[sgpr].u + 1) {
        why = "donor push constant not in two consecutive user SGPRs"; return {};
    }
    const uint32_t user = (kernel.rsrc2 >> 1) & 0x1f;
    if (map->a.size() < user) { why = "donor user data map too short"; return {}; }
    const uint64_t push = map->a[sgpr].u;
    // the push constant in the kernel's kernarg SGPRs (s[0:1], or s[4:5] behind gfx10's private segment buffer)
    for (size_t k = 0; k < map->a.size(); ++k)
        map->a[k] = Mp::uint(k >= kernel.kernargSgpr && k < kernel.kernargSgpr + 2 ? push + k - kernel.kernargSgpr : 0xffffffffu);
    if (lds->u < kernel.lds) { why = "donor LDS smaller than the kernel's"; return {}; }
    // Scratch on gfx12, experimental (D4R_VKHIP_SCRATCH=1): flat scratch is architected there, so the pipeline's
    // scratch_en / scratch_memory_size (bytes per lane, the kernel descriptor's private segment size; LLPC writes the
    // same unit) let the driver set the kernel's scratch up as for a spilling LLPC shader (the kernel descriptor still
    // enables only the kernarg pointer). The gfx1201 K layers spill (enc0 132 B, dec0 104, dec1 40, dec2 44) and run
    // much faster this way (RX 9070 XT, 26.10.1, 720p -> 1440p: K 1.65 -> 1.11 ms), but not reliably: at 1080p -> 4K,
    // 5-25% of 600-frame runs diverged slightly (a glitch the history spreads), whatever the scratch size (1x-32x),
    // barriers or grid trimming, while the same kernels through HIP (d3dhip) and M's scratch-free layers through this
    // path were deterministic. So off by default; gfx11 stays refused.
    const char* scratchEnv = std::getenv("D4R_VKHIP_SCRATCH");
    const bool scratchOk = kernel.scratch && scratchEnv && scratchEnv[0] == '1' &&
                           (module.mach == gfx12Mach(120000) || module.mach == gfx12Mach(120001));
    if (kernel.scratch && !scratchOk) { why = "kernel uses scratch (not supported through pipeline binaries)"; return {}; }
    // gfx10.3 has these modes in the same rsrc1 bits as gfx11
    const bool gfx11 = gfx11Mach(110000) == module.mach || gfx11Mach(110001) == module.mach ||
                       gfx11Mach(110002) == module.mach || gfx11Mach(110003) == module.mach || gfx10(module.mach);
    const uint32_t r1 = kernel.rsrc1, r2 = kernel.rsrc2;
    cs->set(".user_sgprs", Mp::uint(user));
    cs->set(".vgpr_count", Mp::uint(((r1 & 0x3f) + 1) * 8));
    cs->set(".sgpr_count", Mp::uint(106));
    cs->set(".float_mode", Mp::uint((r1 >> 12) & 0xff));
    cs->set(".debug_mode", Mp::boolean((r1 >> 22) & 1));
    cs->set(".wgp_mode", Mp::boolean((r1 >> 29) & 1));
    cs->set(".mem_ordered", Mp::boolean((r1 >> 30) & 1));
    cs->set(".forward_progress", Mp::boolean((r1 >> 31) & 1));
    if (gfx11) {
        // gfx12 dropped these modes
        cs->set(".dx10_clamp", Mp::boolean((r1 >> 21) & 1));
        cs->set(".ieee_mode", Mp::boolean((r1 >> 23) & 1));
        cs->set(".fp16_overflow", Mp::boolean((r1 >> 26) & 1));
    }
    cs->set(".excp_en", Mp::uint((r2 >> 24) & 0x7f));
    cs->set(".scratch_en", Mp::boolean(kernel.scratch != 0));
    cs->set(".scratch_memory_size", Mp::uint(kernel.scratch));
    cs->set(".trap_present", Mp::boolean(false));
    cs->set(".image_op", Mp::boolean(false));
    regs->set(".tgid_x_en", Mp::boolean((r2 >> 7) & 1));
    regs->set(".tgid_y_en", Mp::boolean((r2 >> 8) & 1));
    regs->set(".tgid_z_en", Mp::boolean((r2 >> 9) & 1));
    regs->set(".tg_size_en", Mp::boolean((r2 >> 10) & 1));
    regs->set(".tidig_comp_cnt", Mp::uint((r2 >> 11) & 3));
    const auto code = injectedCode(module, kernel, dataAddress);
    if (code.empty()) { why = "entry beyond s_branch range"; return {}; }
    std::vector<uint8_t> desc;
    mpWrite(desc, meta);
    const std::string entry = symbol->s;

    // ELF: header, .text at 256, .note, .symtab, .strtab (section and symbol names), section headers
    std::vector<uint8_t> o(f.begin(), f.begin() + 64);
    auto align = [&](size_t a) { while (o.size() % a) o.push_back(0); return o.size(); };
    align(256);
    const size_t textAt = o.size(), textSize = code.size() * 4;
    o.resize(textAt + textSize);
    std::memcpy(o.data() + textAt, code.data(), textSize);
    const size_t noteAt = align(4);
    o.insert(o.end(), notes.begin(), notes.end());
    const uint32_t metaHeader[3] = {7, uint32_t(desc.size()), NT_AMDGPU_METADATA};
    o.insert(o.end(), reinterpret_cast<const uint8_t*>(metaHeader), reinterpret_cast<const uint8_t*>(metaHeader) + 12);
    o.insert(o.end(), {'A', 'M', 'D', 'G', 'P', 'U', 0, 0});
    o.insert(o.end(), desc.begin(), desc.end());
    align(4);
    const size_t noteSize = o.size() - noteAt;
    std::string strtab(1, '\0');
    auto name = [&](const std::string& s) { const uint32_t at = uint32_t(strtab.size()); strtab += s; strtab += '\0'; return at; };
    const uint32_t nStrtab = name(".strtab"), nText = name(".text"), nNote = name(".note"), nSymtab = name(".symtab"),
                   nEntry = name(entry);
    const size_t symAt = align(8);
    o.resize(symAt + 48, 0);  // the null symbol, then the entry: GLOBAL FUNC in .text (section 2) at 0
    wr<uint32_t>(o, symAt + 24, nEntry); o[symAt + 28] = 0x12; wr<uint16_t>(o, symAt + 30, 2);
    wr<uint64_t>(o, symAt + 40, textSize);
    const size_t strAt = o.size();
    o.insert(o.end(), strtab.begin(), strtab.end());
    const size_t sectionsAt = align(8);
    o.resize(sectionsAt + 5 * 64, 0);
    auto section = [&](int i, uint32_t nameAt, uint32_t type, uint64_t flags, size_t at, size_t size, uint32_t link,
                       uint32_t info, uint64_t alignment, uint64_t entsize) {
        const size_t h = sectionsAt + 64 * size_t(i);
        wr<uint32_t>(o, h, nameAt); wr<uint32_t>(o, h + 4, type); wr<uint64_t>(o, h + 8, flags);
        wr<uint64_t>(o, h + 24, at); wr<uint64_t>(o, h + 32, size); wr<uint32_t>(o, h + 40, link);
        wr<uint32_t>(o, h + 44, info); wr<uint64_t>(o, h + 48, alignment); wr<uint64_t>(o, h + 56, entsize);
    };
    section(1, nStrtab, 3 /* SHT_STRTAB */, 0, strAt, strtab.size(), 0, 0, 1, 0);
    section(2, nText, 1 /* SHT_PROGBITS */, SHF_ALLOC | 4 /* SHF_EXECINSTR */, textAt, textSize, 0, 0, 256, 0);
    section(3, nNote, 7 /* SHT_NOTE */, 0, noteAt, noteSize, 0, 0, 4, 0);
    section(4, nSymtab, SHT_SYMTAB, 0, symAt, 48, 1, 1, 8, 24);
    wr<uint64_t>(o, 0x18, 0);           // e_entry
    wr<uint64_t>(o, 0x20, 0);           // e_phoff
    wr<uint64_t>(o, 0x28, sectionsAt);  // e_shoff
    wr<uint16_t>(o, 0x36, 0); wr<uint16_t>(o, 0x38, 0);  // no program headers
    wr<uint16_t>(o, 0x3a, 64); wr<uint16_t>(o, 0x3c, 5); wr<uint16_t>(o, 0x3e, 1);
    return o;
}

// createPipeline through VK_KHR_pipeline_binary: capture the donor's binary, rebuild it, create the pipeline from it
VkPipeline palPipeline(const Device& vk, VkComputePipelineCreateInfo ci, const Module& module, const Kernel& kernel,
                       uint64_t dataAddress, uint32_t id, std::string& error) {
    auto fail = [&](const std::string& why) { error = kernel.name + ": " + why; };
    VkPipelineCreateFlags2CreateInfoKHR capture{VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO_KHR};
    capture.flags = VK_PIPELINE_CREATE_2_CAPTURE_DATA_BIT_KHR;
    ci.pNext = &capture;
    VkPipeline donor = VK_NULL_HANDLE, result = VK_NULL_HANDLE;
    VkPipelineBinaryKHR donorBinary = VK_NULL_HANDLE, binary = VK_NULL_HANDLE;
    do {
        if (vk.createCompute(vk.device, VK_NULL_HANDLE, 1, &ci, nullptr, &donor) != VK_SUCCESS) { fail("donor pipeline"); break; }
        VkPipelineBinaryCreateInfoKHR bci{VK_STRUCTURE_TYPE_PIPELINE_BINARY_CREATE_INFO_KHR};
        bci.pipeline = donor;
        VkPipelineBinaryHandlesInfoKHR handles{VK_STRUCTURE_TYPE_PIPELINE_BINARY_HANDLES_INFO_KHR};
        if (vk.createBinaries(vk.device, &bci, nullptr, &handles) != VK_SUCCESS || handles.pipelineBinaryCount != 1) {
            fail("expected one donor pipeline binary, found " + std::to_string(handles.pipelineBinaryCount)); break;
        }
        handles.pPipelineBinaries = &donorBinary;
        if (vk.createBinaries(vk.device, &bci, nullptr, &handles) != VK_SUCCESS || !donorBinary) { fail("donor pipeline binary"); break; }
        VkPipelineBinaryDataInfoKHR di{VK_STRUCTURE_TYPE_PIPELINE_BINARY_DATA_INFO_KHR};
        di.pipelineBinary = donorBinary;
        VkPipelineBinaryKeyKHR key{VK_STRUCTURE_TYPE_PIPELINE_BINARY_KEY_KHR};
        size_t size = 0;
        if (vk.binaryData(vk.device, &di, &key, &size, nullptr) != VK_SUCCESS || !size) { fail("donor binary size"); break; }
        std::vector<uint8_t> data(size);
        if (vk.binaryData(vk.device, &di, &key, &size, data.data()) != VK_SUCCESS) { fail("donor binary data"); break; }
        data.resize(size);
        std::string why;
        const auto patched = palBinary(data, module, kernel, dataAddress, why);
        if (patched.empty()) { fail(why); break; }
        // a key of its own (FNV-1a of the binary and the id), so the driver never mistakes it for the donor
        VkPipelineBinaryKeyKHR patchedKey{VK_STRUCTURE_TYPE_PIPELINE_BINARY_KEY_KHR};
        patchedKey.keySize = 16;
        for (uint32_t half = 0; half < 2; ++half) {
            uint64_t h = 0xcbf29ce484222325ull ^ (uint64_t(id) << 32 | half);
            for (uint8_t b : patched) h = (h ^ b) * 0x100000001b3ull;
            std::memcpy(patchedKey.key + 8 * half, &h, 8);
        }
        VkPipelineBinaryDataKHR blob{patched.size(), const_cast<uint8_t*>(patched.data())};
        VkPipelineBinaryKeysAndDataKHR keysAndData{1, &patchedKey, &blob};
        VkPipelineBinaryCreateInfoKHR pci{VK_STRUCTURE_TYPE_PIPELINE_BINARY_CREATE_INFO_KHR};
        pci.pKeysAndDataInfo = &keysAndData;
        VkPipelineBinaryHandlesInfoKHR created{VK_STRUCTURE_TYPE_PIPELINE_BINARY_HANDLES_INFO_KHR};
        created.pipelineBinaryCount = 1; created.pPipelineBinaries = &binary;
        if (vk.createBinaries(vk.device, &pci, nullptr, &created) != VK_SUCCESS || !binary) { fail("the driver refused the patched binary"); break; }
        VkPipelineBinaryInfoKHR use{VK_STRUCTURE_TYPE_PIPELINE_BINARY_INFO_KHR};
        use.binaryCount = 1; use.pPipelineBinaries = &binary;
        ci.pNext = &use;
        if (vk.createCompute(vk.device, VK_NULL_HANDLE, 1, &ci, nullptr, &result) != VK_SUCCESS) { result = VK_NULL_HANDLE; fail("pipeline"); break; }
    } while (false);
    if (binary) vk.destroyBinary(vk.device, binary, nullptr);
    if (donorBinary) vk.destroyBinary(vk.device, donorBinary, nullptr);
    if (donor) vk.destroyPipeline(vk.device, donor, nullptr);
    return result;
}

} // namespace

VkPipeline createPipeline(const Device& vk, VkPipelineLayout layout, const Module& module, const Kernel& kernel,
                          Launch launch, uint64_t dataAddress, uint32_t id, std::string& error) {
    VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smi.codeSize = sizeof(donor_spv); smi.pCode = donor_spv;
    VkShaderModule shader = VK_NULL_HANDLE;
    if (vk.createModule(vk.device, &smi, nullptr, &shader) != VK_SUCCESS) { error = "donor shader module"; return VK_NULL_HANDLE; }
    const uint32_t spec[5] = {launch.bx, launch.by, launch.bz, std::max(1u, (kernel.lds + 3) / 4), id};
    VkSpecializationMapEntry entries[5];
    for (uint32_t i = 0; i < 5; ++i) entries[i] = {i, 4 * i, 4};
    VkSpecializationInfo si{5, entries, sizeof(spec), spec};
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo wave{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
    wave.requiredSubgroupSize = 32;
    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, &wave, 0, VK_SHADER_STAGE_COMPUTE_BIT, shader, "main", &si};
    ci.layout = layout;
    if (vk.pipelineBinary) {
        VkPipeline result = palPipeline(vk, ci, module, kernel, dataAddress, id, error);
        vk.destroyModule(vk.device, shader, nullptr);
        return result;
    }
    // RADV's cached shader takes the kernarg pointer in s[0:1] only (no gfx10 private segment buffer layout)
    if (kernel.kernargSgpr) {
        vk.destroyModule(vk.device, shader, nullptr);
        error = kernel.name + ": gfx10 code objects need AMD's driver (VK_KHR_pipeline_binary)";
        return VK_NULL_HANDLE;
    }
    VkPipeline result = VK_NULL_HANDLE;
    VkPipelineCache donorCache = VK_NULL_HANDLE, patchedCache = VK_NULL_HANDLE;
    VkPipeline donor = VK_NULL_HANDLE;
    std::vector<uint8_t> data, out;
    size_t size = 0;
    auto fail = [&](const std::string& why) { error = kernel.name + ": " + why; };
    do {
        VkPipelineCacheCreateInfo cci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
        if (vk.createCache(vk.device, &cci, nullptr, &donorCache) != VK_SUCCESS) { fail("donor cache"); break; }
        if (vk.createCompute(vk.device, donorCache, 1, &ci, nullptr, &donor) != VK_SUCCESS) { fail("donor pipeline"); break; }
        if (vk.cacheData(vk.device, donorCache, &size, nullptr) != VK_SUCCESS || !size) { fail("cache size"); break; }
        data.resize(size);
        if (vk.cacheData(vk.device, donorCache, &size, data.data()) != VK_SUCCESS) { fail("cache data"); break; }
        data.resize(size);
        // vk_pipeline_cache: header (header_size first), u32 count, then {u32 type, u32 key size, u32 data size,
        // key, align 8, data} per object
        const uint32_t headerSize = rd<uint32_t>(data, 0);
        if (headerSize < 16 || headerSize + 4 > data.size()) { fail("cache header"); break; }
        const uint32_t count = rd<uint32_t>(data, headerSize);
        out.assign(data.begin(), data.begin() + headerSize + 4);
        size_t at = headerSize + 4;
        int shaders = 0;
        bool bad = false;
        for (uint32_t e = 0; e < count && !bad; ++e) {
            if (at + 12 > data.size()) { bad = true; break; }
            const uint32_t type = rd<uint32_t>(data, at), keySize = rd<uint32_t>(data, at + 4), dataSize = rd<uint32_t>(data, at + 8);
            size_t keyAt = at + 12, dataAt = (keyAt + keySize + 7) & ~size_t(7);
            if (dataAt + dataSize > data.size()) { bad = true; break; }
            std::vector<uint8_t> object(data.begin() + dataAt, data.begin() + dataAt + dataSize);
            size_t q = 0, totalAt = 0;
            const size_t h = legacyHeader(object, totalAt, q);
            if (h) {
                // the donor's config: LDS and user SGPRs must be what the kernel expects
                const uint32_t donorRsrc2 = rd<uint32_t>(object, CFG + RSRC2);
                if (((donorRsrc2 >> 1) & 0x1f) != 2) { fail("donor user SGPRs != 2 (push constant not inlined at s[0:1])"); bad = true; break; }
                // Fingerprint of the layout this code was written against (Mesa 26.2): a config whose fields do not
                // agree with each other and with what the donor declares means the struct moved; refuse rather than
                // patch the wrong words.
                {
                    const uint32_t sgprs = rd<uint32_t>(object, CFG + NUM_SGPRS), vgprs = rd<uint32_t>(object, CFG + NUM_VGPRS);
                    const uint32_t lds = rd<uint32_t>(object, CFG + LDS_SIZE), scratch = rd<uint32_t>(object, CFG + SCRATCH);
                    const uint32_t rsrc1 = rd<uint32_t>(object, CFG + RSRC1), fmode = rd<uint32_t>(object, CFG + FLOAT_MODE);
                    const uint32_t declared = std::max(1u, (kernel.lds + 3) / 4) * 4;
                    const bool sane = sgprs && sgprs <= 128 && vgprs && vgprs <= 256 && ((rsrc1 & 0x3f) + 1) * 8 >= vgprs &&
                                      ((rsrc1 & 0x3f) + 1) * 8 < vgprs + 8 && lds >= declared && lds < declared + 1024 &&
                                      ((donorRsrc2 & RSRC2_LDS_MASK) != 0) == (lds != 0) && scratch == 0 && fmode < 256 &&
                                      object[CFG + WGP_MODE] <= 1 && ((rsrc1 >> 12) & 0xff) == fmode;
                    if (!sane) {
                        char fields[160];
                        std::snprintf(fields, sizeof(fields), " [sgprs %u vgprs %u rsrc1 %#x rsrc2 %#x lds %u/%u scratch %u fmode %u wgp %u]",
                                      sgprs, vgprs, rsrc1, donorRsrc2, lds, declared, scratch, fmode, object[CFG + WGP_MODE]);
                        fail(std::string("unrecognized RADV shader binary layout (Mesa changed?)") + fields); bad = true; break;
                    }
                }
                std::vector<uint8_t> patched(object.begin(), object.begin() + h);
                const auto code = injectedCode(module, kernel, dataAddress);
                if (code.empty()) { fail("entry beyond s_branch range"); bad = true; break; }
                const uint32_t codeBytes = uint32_t(code.size() * 4);
                wr<uint32_t>(patched, CFG + NUM_SGPRS, 106);
                wr<uint32_t>(patched, CFG + NUM_VGPRS, ((kernel.rsrc1 & 0x3f) + 1) * 8);
                wr<uint32_t>(patched, CFG + SHARED_VGPRS, 0);
                wr<uint32_t>(patched, CFG + SPILLED_SGPRS, 0);
                wr<uint32_t>(patched, CFG + SPILLED_VGPRS, 0);
                wr<uint32_t>(patched, CFG + FLOAT_MODE, (kernel.rsrc1 >> 12) & 0xff);
                wr<uint32_t>(patched, CFG + SCRATCH, (kernel.scratch * 32 + 255) / 256 * 256);
                patched[CFG + WGP_MODE] = uint8_t((kernel.rsrc1 >> 29) & 1);
                wr<uint32_t>(patched, CFG + RSRC1, kernel.rsrc1);
                wr<uint32_t>(patched, CFG + RSRC2, (kernel.rsrc2 & ~RSRC2_LDS_MASK) | (donorRsrc2 & RSRC2_LDS_MASK));
                wr<uint32_t>(patched, CFG + RSRC3, kernel.rsrc3);
                // total_size precedes the legacy sizes (code, exec, ir, disasm, stats, debug info)
                wr<uint32_t>(patched, totalAt, uint32_t(h + codeBytes));
                wr<uint32_t>(patched, q, codeBytes);
                wr<uint32_t>(patched, q + 4, codeBytes);
                for (int k = 2; k <= 5; ++k) wr<uint32_t>(patched, q + 4 * k, 0);
                patched.resize(h + codeBytes);
                std::memcpy(patched.data() + h, code.data(), codeBytes);
                object.swap(patched);
                ++shaders;
            }
            const size_t base = out.size();
            out.resize(base + 12);
            wr<uint32_t>(out, base, type); wr<uint32_t>(out, base + 4, keySize); wr<uint32_t>(out, base + 8, uint32_t(object.size()));
            out.insert(out.end(), data.begin() + keyAt, data.begin() + keyAt + keySize);
            while (out.size() % 8) out.push_back(0);
            out.insert(out.end(), object.begin(), object.end());
            at = dataAt + dataSize;
        }
        if (bad) { if (error.empty()) fail("cache data layout"); break; }
        if (shaders != 1) { fail("expected one shader object in the donor cache, found " + std::to_string(shaders)); break; }
        cci.initialDataSize = out.size(); cci.pInitialData = out.data();
        if (vk.createCache(vk.device, &cci, nullptr, &patchedCache) != VK_SUCCESS) { fail("patched cache"); break; }
        if (vk.createCompute(vk.device, patchedCache, 1, &ci, nullptr, &result) != VK_SUCCESS) { result = VK_NULL_HANDLE; fail("pipeline"); break; }
    } while (false);
    if (donor) vk.destroyPipeline(vk.device, donor, nullptr);
    if (donorCache) vk.destroyCache(vk.device, donorCache, nullptr);
    if (patchedCache) vk.destroyCache(vk.device, patchedCache, nullptr);
    vk.destroyModule(vk.device, shader, nullptr);
    return result;
}

} // namespace vkhip
