// Round the components of a float32 cooperative accumulator to binary16 (round-to-nearest-even) and widen them
// back to f32. A register-only conversion, or a f32 -> f16 -> f32 component conversion, is folded away by RADV on
// gfx1201, and so is the packHalf2x16 round-trip (which is round-toward-zero anyway): the rounding only happens
// once the binary16 value lands in binary16 memory. Scalar f32 -> f16 -> f32 casts are folded the same way,
// notably for filtered recurrent features, hence the integer helper below.
// Cooperative matrices are composites of implementation-dependent components: every invocation owns and may
// subscript a fixed number of them (length(), 8 for a 16x16 subgroup-scope accumulator on a 32-lane wave).
float roundHalfValue(float v) {
    uint bits = floatBitsToUint(v), magnitude = bits & 0x7fffffffu;
    if (magnitude < 0x38800000u) return roundEven(v * 16777216.0) * (1.0 / 16777216.0);
    if (magnitude >= 0x477ff000u)
        return magnitude > 0x7f800000u ? v : uintBitsToFloat((bits & 0x80000000u) | 0x7f800000u);
    return uintBitsToFloat((bits + 4095u + ((bits >> 13u) & 1u)) & 0xffffe000u);
}
vec4 roundHalfValue(vec4 v) {
    return vec4(roundHalfValue(v.x), roundHalfValue(v.y), roundHalfValue(v.z), roundHalfValue(v.w));
}

// The cooperative-matrix store/read-back through a wave-owned row-major region: the cheapest form where the region
// already exists for the stage's own staging store (dec0 reuses EO). storage/offset/stride are in storage's
// element units, as in the engine's LD16/ST16.
#define ROUND_HALF_STORE(m, storage, offset, stride) { \
    coopMatStore(HACC(m), storage, offset, stride, RM); \
    subgroupMemoryBarrierShared(); subgroupBarrier(); \
    HACC rh_st_; coopMatLoad(rh_st_, storage, offset, stride, RM); \
    m = ACC(rh_st_); \
    subgroupMemoryBarrierShared(); subgroupBarrier(); \
}

// The hardware binary16 conversion through one private scratch slot per invocation, ~2x cheaper than a
// cooperative-matrix store/read-back to shared memory. storage must hold 8 binary16 values per invocation of the
// workgroup: shared float16_t RS[gl_WorkGroupSize.x * 8u]; (8 = components per invocation, 256 / subgroup size).
// The memory fences prevent compiler forwarding from removing the rounding. Slots are invocation-private,
// so no subgroup execution barrier is needed (unlike ROUND_HALF_STORE's cross-lane cooperative loads).
#define ROUND_HALF_SHARED(m, storage) { \
    for (int rh_i_ = 0, rh_n_ = (m).length(); rh_i_ < rh_n_; ++rh_i_) \
        storage[gl_LocalInvocationID.x * 8u + uint(rh_i_)] = float16_t((m)[rh_i_]); \
    subgroupMemoryBarrierShared(); \
    for (int rh_i_ = 0, rh_n_ = (m).length(); rh_i_ < rh_n_; ++rh_i_) \
        (m)[rh_i_] = float(storage[gl_LocalInvocationID.x * 8u + uint(rh_i_)]); \
    subgroupMemoryBarrierShared(); \
}
