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
            for (unsigned channels : {1u, 2u, 4u}) for (bool surface : {false, true}) {
                d4r::cuda::Image image(images, 17, 9, 16, channels, surface);
                image.verify_wrong_object_kind();
                std::vector<uint16_t> input(17 * 9 * channels), output(input.size());
                for (size_t i = 0; i < input.size(); ++i) input[i] = uint16_t((i + iteration) % 8 * 0x100);
                image.upload(input.data());
                image.download(output.data());
                if (input != output) throw std::runtime_error("CUDA array storage roundtrip mismatch");
            }
        }
        cuda.check(cuda.cuCtxSynchronize(), "cuCtxSynchronize");
        std::printf("PASS CUDA_IMAGES architecture=gfx1201 iterations=%u descriptor_and_storage=1\n", args.iterations);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL CUDA_IMAGES %s\n", e.what());
        loaded_modules();
        return 4;
    }
}
