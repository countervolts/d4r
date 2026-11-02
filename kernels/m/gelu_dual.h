// Experimental gfx12 wave32 GELU scheduling. Fixed register constraints make
// the bank/parity requirements explicit; callers must measure the move and
// occupancy cost before using this inside a larger kernel.
#pragma once

template <bool Dual>
__device__ __forceinline__ void swin_gelu_scheduled(float (&x)[8])
{
#define G_PAIR(OP, A, B) OP A "\n\t" OP B "\n\t"
#define G_DUAL(OP, A, B) "v_dual_" OP A " :: v_dual_" OP B "\n\t"
#define G_STEPS(P, PREFIX) \
    P(PREFIX "max_num_f32 ", "v8, -2.0, v0", "v9, -2.0, v1") \
    P(PREFIX "max_num_f32 ", "v10, -2.0, v2", "v11, -2.0, v3") \
    P(PREFIX "max_num_f32 ", "v12, -2.0, v4", "v13, -2.0, v5") \
    P(PREFIX "max_num_f32 ", "v14, -2.0, v6", "v15, -2.0, v7") \
    P(PREFIX "min_num_f32 ", "v8, 2.0, v8", "v9, 2.0, v9") \
    P(PREFIX "min_num_f32 ", "v10, 2.0, v10", "v11, 2.0, v11") \
    P(PREFIX "min_num_f32 ", "v12, 2.0, v12", "v13, 2.0, v13") \
    P(PREFIX "min_num_f32 ", "v14, 2.0, v14", "v15, 2.0, v15") \
    "v_and_b32 v16, 0x7fffffff, v8\n\t" \
    "v_and_b32 v17, 0x7fffffff, v9\n\t" \
    "v_and_b32 v18, 0x7fffffff, v10\n\t" \
    "v_and_b32 v19, 0x7fffffff, v11\n\t" \
    "v_and_b32 v20, 0x7fffffff, v12\n\t" \
    "v_and_b32 v21, 0x7fffffff, v13\n\t" \
    "v_and_b32 v22, 0x7fffffff, v14\n\t" \
    "v_and_b32 v23, 0x7fffffff, v15\n\t" \
    P(PREFIX "fmaak_f32 ", "v16, %8, v16, %9", "v17, %8, v17, %9") \
    P(PREFIX "fmaak_f32 ", "v18, %8, v18, %9", "v19, %8, v19, %9") \
    P(PREFIX "fmaak_f32 ", "v20, %8, v20, %9", "v21, %8, v21, %9") \
    P(PREFIX "fmaak_f32 ", "v22, %8, v22, %9", "v23, %8, v23, %9") \
    P(PREFIX "fmaak_f32 ", "v16, v8, v16, 0.5", "v17, v9, v17, 0.5") \
    P(PREFIX "fmaak_f32 ", "v18, v10, v18, 0.5", "v19, v11, v19, 0.5") \
    P(PREFIX "fmaak_f32 ", "v20, v12, v20, 0.5", "v21, v13, v21, 0.5") \
    P(PREFIX "fmaak_f32 ", "v22, v14, v22, 0.5", "v23, v15, v23, 0.5") \
    P(PREFIX "mul_f32 ", "v0, v0, v16", "v1, v1, v17") \
    P(PREFIX "mul_f32 ", "v2, v2, v18", "v3, v3, v19") \
    P(PREFIX "mul_f32 ", "v4, v4, v20", "v5, v5, v21") \
    P(PREFIX "mul_f32 ", "v6, v6, v22", "v7, v7, v23")
#define G_ARGS \
    : "+{v0}"(x[0]), "+{v1}"(x[1]), "+{v2}"(x[2]), "+{v3}"(x[3]), \
      "+{v4}"(x[4]), "+{v5}"(x[5]), "+{v6}"(x[6]), "+{v7}"(x[7]) \
    : "s"(-0.0810546875f), "i"(0x3ed30000) \
    : "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", \
      "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23"
    // Constants match swin_tok.h's half-rounded coefficients. The explicit
    // abs mask is valid for the clamp's finite results (NaNs remain NaNs in x).
    if constexpr (Dual) {
        asm volatile(G_STEPS(G_DUAL, "") G_ARGS);
    } else {
        asm volatile(G_STEPS(G_PAIR, "v_") G_ARGS);
    }
#undef G_ARGS
#undef G_STEPS
#undef G_DUAL
#undef G_PAIR
}
