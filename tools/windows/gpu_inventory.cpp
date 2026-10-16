#include "hip_api.h"
#include <fstream>
#include <sstream>
#include <iomanip>

static std::string json_string(const std::string& value) {
    std::ostringstream out; out << '"';
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') out << '\\' << char(c);
        else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(c);
        else out << char(c);
    }
    out << '"'; return out.str();
}
int main(int argc, char** argv) {
    using namespace d4r::diag;
    start();
    try {
        Args args(argc, argv);
        if (args.output_dir.empty()) throw std::runtime_error("--output-dir is required");
        for (const char* name : {"HSA_OVERRIDE_GFX_VERSION", "HSA_OVERRIDE_GFX_VERSION_0"})
            if (std::getenv(name)) throw std::runtime_error(std::string("Unset architecture override ") + name);
        HipApi hip(args.hip_root);
        hip.check(hip.hipInit(0), "hipInit(inventory)");
        int count=0, version=0, driver=0;
        hip.check(hip.hipRuntimeGetVersion(&version), "hipRuntimeGetVersion");
        hip.check(hip.hipDriverGetVersion(&driver), "hipDriverGetVersion");
        std::printf("GPU_INVENTORY_RUNTIME runtime=%d driver_api=%d headers=%d.%d.%d sizeof(properties)=%zu\n",
                    version,driver,HIP_VERSION_MAJOR,HIP_VERSION_MINOR,HIP_VERSION_PATCH,sizeof(hipDeviceProp_t));
        hip.check(hip.hipGetDeviceCount(&count), "hipGetDeviceCount");
        std::ostringstream json;
        json << "{\"schema\":1,\"runtimeVersion\":" << version << ",\"driverApiVersion\":" << driver << ",\"compiledTarget\":"
             << json_string(D4R_TARGET_ARCH) << ",\"gpuWorkExecuted\":false,\"devices\":[";
        for (int i=0; i<count; ++i) {
            hipDeviceProp_t p{}; hip.check(hip.hipGetDeviceProperties(&p,i), "hipGetDevicePropertiesR0600");
            const auto arch=gpu_architecture(p.gcnArchName);
            std::printf("GPU_INVENTORY ordinal=%d name=%s gcnArchName=%s wave=%d build_eligible=%d\n",
                        i,p.name,p.gcnArchName,p.warpSize,supported_gpu_target(arch));
            if (i) json << ',';
            json << "{\"ordinal\":" << i << ",\"name\":" << json_string(p.name)
                 << ",\"gcnArchName\":" << json_string(p.gcnArchName) << ",\"architecture\":" << json_string(arch)
                 << ",\"wavefrontSize\":" << p.warpSize << ",\"vramBytes\":" << p.totalGlobalMem
                 << ",\"buildEligible\":" << (supported_gpu_target(arch) ? "true" : "false") << '}';
        }
        json << "]}\n";
        std::filesystem::create_directories(wide(args.output_dir));
        std::ofstream file(std::filesystem::path(wide(args.output_dir))/L"gpu-inventory.json",std::ios::binary);
        file << json.str(); file.close();
        if (!file) throw std::runtime_error("Cannot write GPU inventory");
        if (!count) throw std::runtime_error("HIP reports no devices");
        loaded_modules();
        std::printf("PASS GPU_INVENTORY devices=%d gpu_work_executed=0\n",count);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr,"FAIL GPU_INVENTORY %s\n",error.what()); loaded_modules(); return 4;
    }
}
