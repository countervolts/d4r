#include "hip_api.h"
#include <fstream>

namespace {
struct PwinParams {
    int width, height;
    const uint8_t* input;
    const uint8_t* skip;
    uint8_t* merged;
    uint8_t* full;
    uint64_t pad40, pad48;
    int sx, sy;
    const uint8_t* weights;
    uint8_t rest[104];
};
static_assert(sizeof(PwinParams) == 176, "K parameter ABI");
static_assert(offsetof(PwinParams, weights) == 64, "K weight pointer ABI");

uint16_t half_quarter(int value)
{
    const uint16_t positive[] = {0x0000u, 0x3400u, 0x3800u, 0x3a00u, 0x3c00u};
    if (value < -4 || value > 4) throw std::runtime_error("bad half fixture");
    return static_cast<uint16_t>(positive[std::abs(value)] | (value < 0 ? 0x8000u : 0u));
}

struct Allocation {
    d4r::diag::HipApi& hip;
    void* pointer = nullptr;
    Allocation(d4r::diag::HipApi& api, size_t bytes) : hip(api) {
        hip.check(hip.hipMalloc(&pointer, bytes), "hipMalloc(K fixture)");
    }
    ~Allocation() { if (pointer) (void)hip.hipFree(pointer); }
};
}

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
        if (args.kernel_name != "enc1" && args.kernel_name != "enc2")
            throw std::runtime_error("--kernel-name must be enc1 or enc2");
        const bool enc2 = args.kernel_name == "enc2";
        const std::string entry = "dltss_pwin_" + args.kernel_name + "_layer";
        hipModule_t module = nullptr;
        hip.check(hip.hipModuleLoad(&module, args.module.c_str()), "hipModuleLoad(K)");
        struct Cleanup {
            HipApi& hip; hipModule_t module;
            ~Cleanup() { if (module) (void)hip.hipModuleUnload(module); }
        } cleanup{hip, module};
        hipFunction_t prep = nullptr, transformer = nullptr;
        hip.check(hip.hipModuleGetFunction(&prep, module, (entry + "_prep").c_str()), "K prep");
        hip.check(hip.hipModuleGetFunction(&transformer, module, entry.c_str()), "K transformer");
        if (!prep || !transformer) throw std::runtime_error("Null K entry point");

        // Identity fixture: all projection, MLP and patch-merge weights are zero.
        // G1/G2 are one. The full transformer must return its input through its
        // residual path and the merged output must be zero. Every stage launches.
        constexpr size_t full_elements = 8u * 8u * 64u;
        const size_t output_channels = enc2 ? 96u : 64u;
        const size_t merged_elements = 4u * 4u * output_channels;
        const size_t weight_bytes = enc2 ? 166080u : 149504u;
        const size_t prep_items = (enc2 ? 288u : 256u) * 16u + 1024u;
        std::vector<uint8_t> weights(weight_bytes, 0);
        auto set_half = [&](size_t byte, uint16_t bits) {
            weights.at(byte) = static_cast<uint8_t>(bits);
            weights.at(byte + 1) = static_cast<uint8_t>(bits >> 8);
        };
        for (size_t c = 0; c < 64u; ++c) {
            set_half(2u * c, 0x3c00u);                  // G1
            set_half(49408u + 2u * c, 0x3c00u);       // G2
        }
        std::vector<uint16_t> input(full_elements), full(full_elements, 0x7e00u);
        std::vector<uint16_t> merged(merged_elements, 0x7e00u);
        for (size_t i = 0; i < input.size(); ++i)
            input[i] = half_quarter(static_cast<int>(i % 9u) - 4);
        Allocation dev_input(hip, input.size() * sizeof(uint16_t));
        Allocation dev_weights(hip, weights.size());
        Allocation dev_full(hip, full.size() * sizeof(uint16_t));
        Allocation dev_merged(hip, merged.size() * sizeof(uint16_t));
        hip.check(hip.hipMemcpy(dev_input.pointer, input.data(), input.size() * sizeof(uint16_t),
            hipMemcpyHostToDevice), "hipMemcpy(K input)");
        hip.check(hip.hipMemcpy(dev_weights.pointer, weights.data(), weights.size(),
            hipMemcpyHostToDevice), "hipMemcpy(K weights)");
        PwinParams params{};
        params.width = params.height = 8;
        params.input = static_cast<const uint8_t*>(dev_input.pointer);
        params.full = static_cast<uint8_t*>(dev_full.pointer);
        params.merged = static_cast<uint8_t*>(dev_merged.pointer);
        params.weights = static_cast<const uint8_t*>(dev_weights.pointer);
        void* arguments[] = {&params};
        hip.check(hip.hipModuleLaunchKernel(prep, static_cast<unsigned>((prep_items + 127u) / 128u), 1, 1,
            128, 1, 1, 0, nullptr, arguments, nullptr), "hipModuleLaunchKernel(K prep)");
        hip.check(hip.hipDeviceSynchronize(), "hipDeviceSynchronize(K prep)");
        for (unsigned iteration = 0; iteration < args.iterations; ++iteration) {
            std::fill(full.begin(), full.end(), 0x7e00u);
            std::fill(merged.begin(), merged.end(), 0x7e00u);
            hip.check(hip.hipMemcpy(dev_full.pointer, full.data(), full.size() * sizeof(uint16_t),
                hipMemcpyHostToDevice), "hipMemcpy(K full sentinel)");
            hip.check(hip.hipMemcpy(dev_merged.pointer, merged.data(), merged.size() * sizeof(uint16_t),
                hipMemcpyHostToDevice), "hipMemcpy(K merged sentinel)");
            hip.check(hip.hipModuleLaunchKernel(transformer, 1, 1, 1, 32, 1, 4, 0,
                nullptr, arguments, nullptr), "hipModuleLaunchKernel(K transformer)");
            hip.check(hip.hipDeviceSynchronize(), "hipDeviceSynchronize(K transformer)");
            hip.check(hip.hipMemcpy(full.data(), dev_full.pointer, full.size() * sizeof(uint16_t),
                hipMemcpyDeviceToHost), "hipMemcpy(K full result)");
            hip.check(hip.hipMemcpy(merged.data(), dev_merged.pointer, merged.size() * sizeof(uint16_t),
                hipMemcpyDeviceToHost), "hipMemcpy(K merged result)");
            for (size_t i = 0; i < full.size(); ++i)
                if (full[i] != input[i]) {
                    std::fprintf(stderr, "FAIL K_IDENTITY iteration=%u full_index=%zu expected=0x%04x actual=0x%04x\n",
                        iteration, i, input[i], full[i]);
                    return 5;
                }
            for (size_t i = 0; i < merged.size(); ++i)
                if (merged[i] != 0u) {
                    std::fprintf(stderr, "FAIL K_IDENTITY iteration=%u merged_index=%zu expected=0x0000 actual=0x%04x\n",
                        iteration, i, merged[i]);
                    return 5;
                }
        }
        std::printf("PASS K_IDENTITY architecture=gfx1201 kernel=%s iterations=%u full_elements=%zu merged_elements=%zu transformer_executed=1 nonzero_weights=0\n",
            args.kernel_name.c_str(), args.iterations, full_elements, merged_elements);

        // Exercise nonzero V projection, position-only attention and Wo GEMMs.
        // The generated data is saved for comparison with pwin_model.py.
        auto fragment_offset = [](unsigned k, unsigned n) {
            return 64u * (n & 7u) + 16u * ((k & 7u) >> 1) + 8u * (n >> 3) +
                   4u * (k >> 3) + 2u * (k & 1u);
        };
        auto set_matrix = [&](size_t base, unsigned k, unsigned n, unsigned nt_stride) {
            set_half(base + 512u * (k >> 4) + nt_stride * (n >> 4) +
                fragment_offset(k & 15u, n & 15u), 0x3c00u);
        };
        constexpr size_t qkv = 128u, head = 12288u, bias = 24704u, wo = 41216u;
        constexpr size_t w1 = 49664u, w2 = 82944u, pm = 115712u;
        for (unsigned k = 0; k < 32u; ++k) {
            if (enc2) {
                set_matrix(qkv, k, k, 1024u);                           // head 0 Q
                set_matrix(qkv + 2048u, k, k, 1024u);                   // head 0 K
                set_matrix(qkv + head + 2048u * 3u, k, k, 1024u);      // head 1 Q
                set_matrix(qkv + head + 2048u * 4u, k, k, 1024u);      // head 1 K
            }
            set_matrix(qkv + 2048u * 2u, k, k, 1024u);                   // head 0 V <- input 0:32
            set_matrix(qkv + head + 2048u * 5u, k, k, 1024u);          // head 1 V <- input 32:64
            set_matrix(wo, k, k, 1024u);                               // head 0 Wo -> output 0:32
            set_matrix(wo + 64u * 64u, k, k + 32u, 1024u);             // head 1 Wo -> output 32:64
            set_matrix(w1, k, k, 2048u);                               // first MLP group input -> hidden
            set_matrix(w2, k, k, 1024u);                               // first MLP group hidden -> output
        }
        for (unsigned k = 0; k < 64u; ++k)
            set_matrix(pm, k, k, 8192u);                               // patch merge selects top-left token
        for (unsigned h = 0; h < 2u; ++h)
            for (unsigned row = 0; row < 64u; ++row)
                for (unsigned col = 0; col < 64u; ++col) {
                    const unsigned r = row & 15u, c = col & 15u;
                    const size_t offset = bias + h * 8192u +
                        512u * (4u * (col >> 4) + (row >> 4)) +
                        64u * (r & 7u) + 16u * ((c & 7u) >> 1) +
                        2u * (c & 1u) + 4u * (r >> 3) + 8u * (c >> 3);
                    set_half(offset, 0x2400u);                         // 1/64 position average
                }
        hip.check(hip.hipMemcpy(dev_weights.pointer, weights.data(), weights.size(),
            hipMemcpyHostToDevice), "hipMemcpy(K nonzero weights)");
        hip.check(hip.hipModuleLaunchKernel(prep, static_cast<unsigned>((prep_items + 127u) / 128u), 1, 1,
            128, 1, 1, 0, nullptr, arguments, nullptr), "hipModuleLaunchKernel(K nonzero prep)");
        hip.check(hip.hipDeviceSynchronize(), "hipDeviceSynchronize(K nonzero prep)");
        std::fill(full.begin(), full.end(), 0x7e00u);
        std::fill(merged.begin(), merged.end(), 0x7e00u);
        hip.check(hip.hipMemcpy(dev_full.pointer, full.data(), full.size() * sizeof(uint16_t),
            hipMemcpyHostToDevice), "hipMemcpy(K nonzero full sentinel)");
        hip.check(hip.hipMemcpy(dev_merged.pointer, merged.data(), merged.size() * sizeof(uint16_t),
            hipMemcpyHostToDevice), "hipMemcpy(K nonzero merged sentinel)");
        hip.check(hip.hipModuleLaunchKernel(transformer, 1, 1, 1, 32, 1, 4, 0,
            nullptr, arguments, nullptr), "hipModuleLaunchKernel(K nonzero transformer)");
        hip.check(hip.hipDeviceSynchronize(), "hipDeviceSynchronize(K nonzero transformer)");
        hip.check(hip.hipMemcpy(full.data(), dev_full.pointer, full.size() * sizeof(uint16_t),
            hipMemcpyDeviceToHost), "hipMemcpy(K nonzero full result)");
        hip.check(hip.hipMemcpy(merged.data(), dev_merged.pointer, merged.size() * sizeof(uint16_t),
            hipMemcpyDeviceToHost), "hipMemcpy(K nonzero merged result)");
        size_t changed = 0;
        for (size_t i = 0; i < full.size(); ++i) {
            if ((full[i] & 0x7c00u) == 0x7c00u)
                throw std::runtime_error("K nonzero full output has NaN/Inf or unwritten sentinel at " + std::to_string(i));
            changed += full[i] != input[i];
        }
        for (unsigned mt = 0; mt < 16u; ++mt) {
            const unsigned my = mt >> 2, mx = mt & 3u;
            const size_t full_base = (8u * (2u * my) + 2u * mx) * 64u;
            for (unsigned channel = 0; channel < output_channels; ++channel) {
                const size_t i = mt * output_channels + channel;
                const uint16_t expected = channel < 64u ? full[full_base + channel] : 0u;
                if (merged[i] != expected)
                    throw std::runtime_error("K patch merge identity mismatch at " + std::to_string(i));
            }
        }
        if (!changed) throw std::runtime_error("K nonzero weights had no visible transformer effect");
        if (!args.fixture_dir.empty()) {
            std::filesystem::create_directories(wide(args.fixture_dir));
            auto save = [&](const char* name, const void* data, size_t bytes) {
                const auto path = std::filesystem::path(wide(args.fixture_dir)) / name;
                std::ofstream file(path, std::ios::binary);
                if (!file.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes)))
                    throw std::runtime_error(std::string("Failed writing K fixture ") + name);
            };
            save("input.bin", input.data(), input.size() * sizeof(uint16_t));
            save("weights.bin", weights.data(), weights.size());
            save("full.bin", full.data(), full.size() * sizeof(uint16_t));
            save("merged.bin", merged.data(), merged.size() * sizeof(uint16_t));
        }
        std::printf("PASS K_NONZERO architecture=gfx1201 kernel=%s changed_elements=%zu full_elements=%zu merged_elements=%zu fixture_saved=%d numerical_reference_pending=1\n",
            args.kernel_name.c_str(), changed, full.size(), merged.size(), !args.fixture_dir.empty());
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL K_MODULE %s\n", error.what());
        loaded_modules();
        return 4;
    }
}
