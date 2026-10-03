#include <hip/hip_runtime.h>
#include <cstdint>

typedef _Float16 half_t;
typedef _Float16 half4_t __attribute__((ext_vector_type(4)));

struct [[gnu::packed, gnu::aligned(8)]] DldnParams {
    uint8_t payload[512];
};

// Реальная структура параметров HKPN денойзера (480 байт)
struct [[gnu::packed, gnu::aligned(8)]] HKPNTailParams {
    uint64_t in_features_ptr;    // [0] Указатель на входящие признаки / денойз
    uint64_t out_pixels_ptr;     // [8] Указатель на выходной линейный буфер VRAM (st.to.global)
    uint64_t in_texture_sampler; // [16] Сэмплер текстуры (tex.level.2d.v4)
    uint32_t width;              // [24]
    uint32_t height;             // [28]
    uint32_t row_pitch;          // [32]
    uint8_t  rest[440];
};

#define DUMMY_KERNEL(name) \
    extern "C" __global__ void __launch_bounds__(256) name(DldnParams params) { __syncthreads(); }

// --- 1. Выходные KPN-фильтры: прямая запись st.to.global ---
extern "C" __global__ void 
__launch_bounds__(256) 
cuda_dldn_engine_hkpn_output_kernel_transformer_diamond_wallaby(HKPNTailParams params)
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    // Прямая запись пикселей денойзера в выходной буфер VRAM (st.to.global)
    if (params.out_pixels_ptr != 0)
    {
        half4_t* out = (half4_t*)params.out_pixels_ptr;
        const size_t pitch = (params.row_pitch > 0) ? (params.row_pitch / sizeof(half4_t)) : 1920;
        
        half4_t color = (half4_t){0.0f, 0.0f, 0.0f, 1.0f};
        if (params.in_features_ptr != 0)
        {
            const half4_t* in = (const half4_t*)params.in_features_ptr;
            color = in[y * pitch + x];
        }

        out[y * pitch + x] = color;
    }
}

extern "C" __global__ void 
__launch_bounds__(256) 
cuda_dldn_engine_hkpn_output_kernel_transformer(HKPNTailParams params)
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (params.out_pixels_ptr != 0)
    {
        half4_t* out = (half4_t*)params.out_pixels_ptr;
        const size_t pitch = (params.row_pitch > 0) ? (params.row_pitch / sizeof(half4_t)) : 1920;
        
        half4_t color = (half4_t){0.0f, 0.0f, 0.0f, 1.0f};
        if (params.in_features_ptr != 0)
        {
            const half4_t* in = (const half4_t*)params.in_features_ptr;
            color = in[y * pitch + x];
        }

        out[y * pitch + x] = color;
    }
}

extern "C" __global__ void 
__launch_bounds__(256) 
rr2_post_kernel(HKPNTailParams params)
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (params.out_pixels_ptr != 0 && params.in_features_ptr != 0)
    {
        half4_t* out = (half4_t*)params.out_pixels_ptr;
        const half4_t* in = (const half4_t*)params.in_features_ptr;
        const size_t pitch = (params.row_pitch > 0) ? (params.row_pitch / sizeof(half4_t)) : 1920;
        out[y * pitch + x] = in[y * pitch + x];
    }
}

// --- 2. Вспомогательные и скрытые слои Swin-трансформера ---
DUMMY_KERNEL(cuda_dldn_engine_hkpn_input_kernel)
DUMMY_KERNEL(cuda_dldn_engine_hkpn_input_kernel_dbg_preset_2)
DUMMY_KERNEL(cuda_dldn_engine_hkpn_input_kernel_preset_1)
DUMMY_KERNEL(cuda_dldn_engine_hkpn_output_kernel)
DUMMY_KERNEL(cuda_dldn_engine_multiscale_downsample_kernel)
DUMMY_KERNEL(cuda_dldn_engine_swin_dec0_diamond_wallaby_kernel)
DUMMY_KERNEL(cuda_dldn_engine_swin_dec0_kernel)
DUMMY_KERNEL(cuda_dldn_engine_swin_dec1_kernel)
DUMMY_KERNEL(cuda_dldn_engine_swin_dec2_kernel)
DUMMY_KERNEL(cuda_dldn_engine_swin_dec3_kernel)
DUMMY_KERNEL(cuda_dldn_engine_swin_dec4_kernel)
DUMMY_KERNEL(cuda_dldn_engine_swin_enc0_diamond_wallaby_kernel)
DUMMY_KERNEL(cuda_dldn_engine_swin_enc0_kernel)
DUMMY_KERNEL(cuda_dldn_engine_swin_enc1_kernel)
DUMMY_KERNEL(cuda_dldn_engine_swin_enc2_kernel)
DUMMY_KERNEL(cuda_dldn_engine_swin_enc3_kernel)
DUMMY_KERNEL(cuda_dldn_engine_swin_enc4_kernel)
DUMMY_KERNEL(cuda_dldn_engine_swin_enc5_kernel)
DUMMY_KERNEL(rr2_clear_range_kernel)
DUMMY_KERNEL(rr2_dec0_kernel_0)
DUMMY_KERNEL(rr2_dec0_kernel_1)
DUMMY_KERNEL(rr2_dec1_kernel_0)
DUMMY_KERNEL(rr2_dec1_kernel_1)
DUMMY_KERNEL(rr2_dec2_kernel_0)
DUMMY_KERNEL(rr2_dec2_kernel_1)
DUMMY_KERNEL(rr2_dec3_kernel_0)
DUMMY_KERNEL(rr2_dec3_kernel_1)
DUMMY_KERNEL(rr2_dec4_kernel_0)
DUMMY_KERNEL(rr2_dec4_kernel_1)
DUMMY_KERNEL(rr2_downsample_kernel_static_hdr)
DUMMY_KERNEL(rr2_downsample_kernel_static_hdr_no_transparency)
DUMMY_KERNEL(rr2_enc0_kernel)
DUMMY_KERNEL(rr2_enc1_kernel)
DUMMY_KERNEL(rr2_enc2_kernel)
DUMMY_KERNEL(rr2_enc3_kernel)
DUMMY_KERNEL(rr2_enc4_kernel)
DUMMY_KERNEL(rr2_enc5_kernel)
DUMMY_KERNEL(rr2_pooled_history2_kernel)
DUMMY_KERNEL(rr2_post_kernel_3_2)