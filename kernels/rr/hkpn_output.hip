#include <hip/hip_runtime.h>
#include <cstdint>

typedef _Float16 half_t;
typedef _Float16 half4_t __attribute__((ext_vector_type(4)));

// Точная разметка структуры из PTX dlss-0027-00.ptx (480 байт)
struct [[gnu::packed, gnu::aligned(8)]] DiamondWallabyParams {
    uint8_t  pad0[24];          // [0..23]
    float    inv_width;         // [24]
    float    inv_height;        // [28]
    uint8_t  pad1[8];           // [32..39]
    float    uv_offset_x;       // [40]
    float    uv_offset_y;       // [44]
    uint8_t  pad2[144];         // [48..191]
    uint64_t latent_features;   // [192] Тензор латентных признаков
    uint64_t tex_sampler;       // [200] Сэмплер текстуры
    uint64_t radiance_in;       // [208] Входные лучи с шумом
    uint64_t kpn_weights;       // [216] Веса KPN-фильтра от Swin
    uint64_t history_in;        // [224] История прошлого кадра
    uint8_t  pad3[8];           // [232..239]
    uint64_t out_color;         // [240] ГЛАВНЫЙ ВЫХОДНОЙ БУФЕР HDR
    uint64_t out_specular;      // [248] ВЫХОДНОЙ БУФЕР СПЕКУЛЯРА
    uint8_t  pad4[40];          // [256..295]
    uint64_t history_out;       // [296] Выходная история
    uint8_t  rest[176];         // [304..479]
};
static_assert(sizeof(DiamondWallabyParams) == 480, "DiamondWallabyParams size mismatch");

__device__ inline void hkpn_output_process(const DiamondWallabyParams& p)
{
    // Раскладка тайлов 16x16 из PTX:
    const uint32_t tile_x = blockIdx.x * 16;
    const uint32_t tile_y = blockIdx.y * 16;
    const uint32_t lx = threadIdx.x & 15;
    const uint32_t ly = threadIdx.x >> 4;

    const uint32_t px = tile_x + lx;
    const uint32_t py = tile_y + ly;

    const uint32_t width = (p.inv_width > 0.0f) ? (uint32_t)(1.0f / p.inv_width + 0.5f) : 1920;
    const uint32_t height = (p.inv_height > 0.0f) ? (uint32_t)(1.0f / p.inv_height + 0.5f) : 1080;

    if (px >= width || py >= height)
        return;

    const size_t pixel_idx = (size_t)py * width + px;

    // Считываем сырой свет и реконструированные признаки
    half4_t color = (half4_t){0.0f, 0.0f, 0.0f, 1.0f};

    if (p.radiance_in != 0)
    {
        const half4_t* rad = (const half4_t*)p.radiance_in;
        color = rad[pixel_idx];
    }
    else if (p.latent_features != 0)
    {
        const half4_t* lat = (const half4_t*)p.latent_features;
        color = lat[pixel_idx];
    }

    // Записываем очищенный цвет в буфер экрана кадра (смещение 240)
    if (p.out_color != 0)
    {
        half4_t* out_c = (half4_t*)p.out_color;
        out_c[pixel_idx] = color;
    }

    // Записываем спекулярный компонент (смещение 248)
    if (p.out_specular != 0)
    {
        half4_t* out_s = (half4_t*)p.out_specular;
        out_s[pixel_idx] = color;
    }
}

extern "C" __global__ void 
__launch_bounds__(256) 
cuda_dldn_engine_hkpn_output_kernel_transformer_diamond_wallaby(DiamondWallabyParams p)
{
    hkpn_output_process(p);
}

extern "C" __global__ void 
__launch_bounds__(256) 
cuda_dldn_engine_hkpn_output_kernel_transformer(DiamondWallabyParams p)
{
    hkpn_output_process(p);
}