// nvxrt-scan: find every CUDA fatbin embedded in a binary (e.g. the game's nvngx_dlssg.dll) and write the PTX module
// the nvxrt layer would pick for it into the kernel cache (<cache>/kernels/<hash>/module.ptx). The hash and the PTX
// choice are the layer's own (same util.cpp), so build_cache.py can translate everything before the game starts.
// usage: nvxrt-scan FILE CACHE_DIR   (prints "<hash> <sm> <entries...>" per module)
#include "util.h"
#include <cstdio>
#include <string>
#include <sys/stat.h>
#include <vector>

using namespace nvxrt;

static void mkdirs(const std::string &path)
{
    for (size_t i = 1; i <= path.size(); i++)
        if (i == path.size() || path[i] == '/')
            mkdir(path.substr(0, i).c_str(), 0755);
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: %s FILE CACHE_DIR\n", argv[0]);
        return 2;
    }
    FILE *f = fopen(argv[1], "rb");
    if (!f) {
        perror(argv[1]);
        return 1;
    }
    std::vector<uint8_t> b;
    fseek(f, 0, SEEK_END);
    b.resize(ftell(f));
    fseek(f, 0, SEEK_SET);
    if (fread(b.data(), 1, b.size(), f) != b.size()) {
        fclose(f);
        return 1;
    }
    fclose(f);
    std::string cache = argv[2];
    int modules = 0, written = 0;
    for (size_t o = 0; o + 16 <= b.size(); o++) {
        uint32_t magic;
        memcpy(&magic, &b[o], 4);
        if (magic != 0xBA55ED50u)
            continue;
        uint16_t hs;
        uint64_t sz;
        memcpy(&hs, &b[o + 6], 2);
        memcpy(&sz, &b[o + 8], 8);
        if (hs < 16 || sz > b.size() - o)
            continue;
        std::vector<FatbinEntry> entries;
        if (!parse_cuda_module(&b[o], hs + sz, entries))
            continue;
        const FatbinEntry *best = nullptr;
        for (auto &e : entries)
            if (e.kind == 1 && e.sm <= 120 && (!best || e.sm > best->sm))
                best = &e;
        if (!best)
            continue;
        modules++;
        std::string ptx(best->data.begin(), best->data.end());
        std::string norm = normalize_ptx(ptx);
        std::string hash = sha256_hex(norm.data(), norm.size());
        std::string dir = cache + "/kernels/" + hash;
        std::string path = dir + "/module.ptx";
        struct stat st;
        if (stat(path.c_str(), &st) != 0) {
            mkdirs(dir);
            FILE *w = fopen((path + ".tmp").c_str(), "wb");
            if (w) {
                fwrite(ptx.data(), 1, ptx.size(), w);
                fclose(w);
                rename((path + ".tmp").c_str(), path.c_str());
                written++;
            }
        }
        printf("%s sm_%u", hash.c_str(), best->sm);
        for (auto &n : ptx_entries(ptx))
            printf(" %s", n.c_str());
        printf("\n");
        o += hs + sz - 1;
    }
    fprintf(stderr, "nvxrt-scan: %d PTX modules, %d new\n", modules, written);
    return modules ? 0 : 1;
}
