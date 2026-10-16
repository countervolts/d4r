#include "gpu_target.h"
#include <cstdio>

int main(int argc, char** argv) {
    using namespace d4r::diag;
    try {
        unsigned rejected = 0, objects = 0;
        for (const auto& info : gpu_targets) {
            const char* target = info.architecture;
            require_gpu_target(target, target);
            require_gpu_target(std::string(target) + ":xnack-:sramecc-", target);
            for (const char* invalid : {"gfx1030", "gfx1104", "gfx1155", "gfx1170", "gfx12000", "gfx1202", "gfx1250", ""}) {
                try { require_gpu_target(invalid, target); }
                catch (const std::exception&) { ++rejected; continue; }
                throw std::runtime_error("Invalid GPU target accepted");
            }
        }
        for (int i = 1; i < argc; ++i) {
            require_code_object_target(argv[i], D4R_TARGET_ARCH);
            const auto other = std::string(D4R_TARGET_ARCH) == "gfx1201" ? "gfx1200" : "gfx1201";
            try { require_code_object_target(argv[i], other); }
            catch (const std::exception&) { ++rejected; ++objects; continue; }
            throw std::runtime_error("Wrong code-object target accepted");
        }
        if (!objects) throw std::runtime_error("Supply actual compiled code objects");
        std::printf("PASS GPU_TARGET_CONTRACT target=%s objects=%u rejected=%u hardware_executed=0\n",
            D4R_TARGET_ARCH, objects, rejected);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL GPU_TARGET_CONTRACT %s\n", e.what()); return 5;
    }
}
