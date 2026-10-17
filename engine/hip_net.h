// d4r engine: the DLSS network on d4r's native HIP layer kernels instead of the Vulkan layer shader. HIP reaches
// the hardware's packed FP8 conversions and keeps a block's activations in registers, which a cooperative-matrix
// shader cannot, so the network runs about 1.7x faster here. The stages around it stay on Vulkan: the engine's
// front writes the tokens into memory shared with HIP, launch() runs the layers, the engine's back reads their
// output (ExternalNetwork in runtime.h).
//
// The model directory needs hipnet.bin and hip/*.hsaco (compile_m.py --hip for preset M, compile_k.py --hip for
// preset K) and the header + entry names select the preset:
//   D4RMHIP1 (10 layers) - M on the kernels/m layer kernels. width/height are the render dimensions; two
//     window-alignment phases, one per evaluation.
//   D4RKHIP1 (11 layers) - K on the kernels/k DltssPaddedWinLayer kernels, the bottleneck layer running as its
//     main kernel plus post1..3. width/height are the OUTPUT dimensions (the token grid is max(256, align 32 of
//     their quarter resolution, makeKShape) and the input/output buffers hold 16 / 40 f16 channels per token,
//     row-major, with no layout copy. Eight immutable window graphs, one per evaluation of makeKWindows' cycle.
// Either way every graph is uploaded once and one evaluation is a single hipGraphLaunch: no per-frame allocation,
// no per-frame parameter copy. The frame counter advances once per launch() and is never reset (a reset
// evaluation is still an evaluation), so the caller must pair one launch() with each network evaluation.
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
namespace d4r {
class HipNetwork {
public:
    // tokensFd, outputFd: opaque file descriptors (Linux) or Win32 NT handles (Windows; the caller keeps ownership) of the two shared Vulkan allocations (ownership passes to HIP);
    // allocationBytes: the size of each allocation; the buffers start at offset 0. width/height are preset
    // dependent, as above.
    HipNetwork(const std::string& modelDirectory, uint32_t width, uint32_t height, intptr_t tokensFd, intptr_t outputFd,
               uint64_t tokensAllocationBytes, uint64_t outputAllocationBytes);
    ~HipNetwork();
    HipNetwork(const HipNetwork&) = delete;
    HipNetwork& operator=(const HipNetwork&) = delete;
    void launch();      // queues the network; input tokens must already be complete
    bool done();        // true once the queued layers have finished
    void wait();        // blocks until they have
    float lastMilliseconds(); // GPU time of the last finished launch
    // Diagnostic: relaunches the network `rounds` x 8 times on the unchanged input tokens (the graphs cycle with
    // period 8) and returns an FNV-1a hash of the output buffer after each launch, rounds-major.
    std::vector<uint64_t> debugRepeatHashes(uint32_t rounds);
    // Diagnostic: median GPU microseconds of every launch node of graph `phase` run individually (not graph-fused),
    // over `trials` runs. Node order is network order, a layer's post kernels following its main kernel.
    std::vector<float> debugNodeMicros(uint32_t phase, uint32_t trials);
private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
} // namespace d4r
