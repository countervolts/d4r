#include "hip_api.h"
#include "cuda_image_api.h"

int main(int argc, char** argv) {
    using namespace d4r::diag;
    start();
    try {
        Args args(argc, argv);
        HipApi hip(args.hip_root);
        hipDeviceProp_t props{};
        hip.select_gfx1201(args.device, props);
        SearchDirectory search(std::filesystem::path(wide(args.cuda_dll)).parent_path());
        CudaApi cuda(args.cuda_dll);
        cuda.check(cuda.cuInit(0), "cuInit");
        CUdevice device = -1;
        int count = 0;
        cuda.check(cuda.cuDeviceGetCount(&count), "cuDeviceGetCount");
        for (int ordinal = 0; ordinal < count; ++ordinal) {
            CUdevice candidate = -1;
            int bus = -1, pciDevice = -1, domain = -1;
            cuda.check(cuda.cuDeviceGet(&candidate, ordinal), "cuDeviceGet");
            cuda.check(cuda.cuDeviceGetAttribute(&bus, 33, candidate), "cuDeviceGetAttribute(PCI_BUS_ID)");
            cuda.check(cuda.cuDeviceGetAttribute(&pciDevice, 34, candidate), "cuDeviceGetAttribute(PCI_DEVICE_ID)");
            cuda.check(cuda.cuDeviceGetAttribute(&domain, 50, candidate), "cuDeviceGetAttribute(PCI_DOMAIN_ID)");
            if (bus == props.pciBusID && pciDevice == props.pciDeviceID && domain == props.pciDomainID) device = candidate;
        }
        if (device < 0) throw std::runtime_error("CUDA image probe HIP/CUDA PCI identity mismatch");
        CUcontext context = nullptr;
        cuda.check(cuda.cuCtxCreate_v2(&context, 0, device), "cuCtxCreate_v2");
        struct Cleanup { CudaApi& api; CUcontext context; ~Cleanup() { (void)api.cuCtxDestroy_v2(context); } } cleanup{cuda, context};
        d4r::cuda::ImageApi images(cuda);
        for (unsigned iteration = 0; iteration < args.iterations; ++iteration) {
            for (unsigned format : {16u, 32u}) for (unsigned channels : {1u, 2u, 4u}) for (bool surface : {false, true}) {
                const unsigned halvesPerPixel = channels * (format == 16 ? 1 : 2);
                d4r::cuda::Image image(images, 17, 9, format, channels, surface);
                image.verify_wrong_object_kind();
                std::vector<uint16_t> input(17 * 9 * halvesPerPixel), output(input.size());
                for (size_t i = 0; i < input.size(); ++i) input[i] = uint16_t((i + iteration) % 8 * 0x100);
                image.upload(input.data());
                image.download(output.data());
                if (input != output) throw std::runtime_error("CUDA array storage roundtrip mismatch");
                const size_t rowBytes = 17 * halvesPerPixel * sizeof(uint16_t), pitch = (rowBytes + 255) & ~size_t(255);
                CUdeviceptr devicePixels = 0;
                cuda.check(cuda.cuMemAlloc_v2(&devicePixels, pitch * 9), "cuMemAlloc_v2(pitched image regression)");
                struct PixelsCleanup { CudaApi& cuda; CUdeviceptr pointer; ~PixelsCleanup() { (void)cuda.cuMemFree_v2(pointer); } } pixelsCleanup{cuda, devicePixels};
                std::vector<uint8_t> padded(pitch * 9, 0xcd);
                for (unsigned y = 0; y < 9; ++y) std::memcpy(padded.data() + y * pitch, input.data() + y * 17 * halvesPerPixel, rowBytes);
                cuda.check(cuda.cuMemcpyHtoD_v2(devicePixels, padded.data(), padded.size()), "cuMemcpyHtoD_v2(pitched pattern)");
                const bool deferred = std::getenv("D4R_DIAG_ASYNC_ARRAY_COPY") != nullptr;
                image.upload_device(devicePixels, pitch, deferred);
                if (deferred) cuda.check(cuda.cuCtxSynchronize(), "cuCtxSynchronize(input batch regression)");
                image.download(output.data());
                if (input != output) throw std::runtime_error("CUDA device-to-array pitched copy mismatch");
                std::fill(padded.begin(), padded.end(), 0xee);
                cuda.check(cuda.cuMemcpyHtoD_v2(devicePixels, padded.data(), padded.size()), "cuMemcpyHtoD_v2(output sentinel)");
                image.download_device(devicePixels, pitch);
                cuda.check(cuda.cuMemcpyDtoH_v2(padded.data(), devicePixels, padded.size()), "cuMemcpyDtoH_v2(device array output)");
                for (unsigned y = 0; y < 9; ++y) {
                    if (std::memcmp(padded.data() + y * pitch, input.data() + y * 17 * halvesPerPixel, rowBytes))
                        throw std::runtime_error("CUDA array-to-device pitched copy mismatch");
                    if (!std::all_of(padded.begin() + y * pitch + rowBytes, padded.begin() + (y + 1) * pitch, [](uint8_t value) { return value == 0xee; }))
                        throw std::runtime_error("CUDA array-to-device copy overwrote pitch padding");
                }
            }
        }
        cuda.check(cuda.cuCtxSynchronize(), "cuCtxSynchronize");
        std::printf("PASS CUDA_IMAGES architecture=gfx1201 iterations=%u descriptor_and_storage=1 pitched_device_copies=1 fp16_fp32=1 async_array_copy=%u\n",
            args.iterations, unsigned(std::getenv("D4R_DIAG_ASYNC_ARRAY_COPY") != nullptr));
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL CUDA_IMAGES %s\n", e.what());
        loaded_modules();
        return 4;
    }
}
