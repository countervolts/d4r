#include "hip_api.h"

int main(int argc, char** argv)
{
    using namespace d4r::diag;
    start();
    try {
        Args args(argc, argv);
        if (args.module.empty()) throw std::runtime_error("--module is required");
        HipApi hip(args.hip_root);
        hipDeviceProp_t properties{};
        hip.select_gfx1201(args.device, properties);
        hipModule_t module = nullptr;
        hip.check(hip.hipModuleLoad(&module, args.module.c_str()), "hipModuleLoad(K enc1)");
        struct Cleanup {
            HipApi& hip; hipModule_t module;
            ~Cleanup() { if (module) (void)hip.hipModuleUnload(module); }
        } cleanup{hip, module};
        for (const char* name : {"dltss_pwin_enc1_layer_prep", "dltss_pwin_enc1_layer"}) {
            hipFunction_t function = nullptr;
            hip.check(hip.hipModuleGetFunction(&function, module, name), name);
            if (!function) throw std::runtime_error(std::string("Null K function: ") + name);
        }
        std::printf("PASS K_MODULE architecture=gfx1201 functions=prep,transformer transformer_executed=0\n");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL K_MODULE %s\n", error.what());
        loaded_modules();
        return 4;
    }
}
