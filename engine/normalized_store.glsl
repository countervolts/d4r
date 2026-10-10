// Match formatted CUDA stores before the Vulkan image conversion. The reference rounds halfway away from
// zero; feeding an exact normalized code to imageStore also avoids implementation-specific UNORM/SNORM rounding.
vec4 unorm8Value(vec4 v) {
    v = clamp(mix(v, vec4(0.0), isnan(v)), 0.0, 1.0);
    precise vec4 scaled = v * 255.0;
    return floor(scaled + 0.5) * (1.0 / 255.0);
}
vec4 snorm8Value(vec4 v) {
    v = clamp(mix(v, vec4(0.0), isnan(v)), -1.0, 1.0);
    precise vec4 scaled = abs(v) * 127.0;
    return sign(v) * floor(scaled + 0.5) * (1.0 / 127.0);
}
vec3 unorm10Value(vec3 v) {
    v = clamp(mix(v, vec3(0.0), isnan(v)), 0.0, 1.0);
    precise vec3 scaled = v * 1023.0;
    precise vec3 code = floor(scaled + 0.5);
    precise vec3 value = code * (1.0 / 1023.0);
    // CUDA's float-backed formatted surface stores code/1023 with correctly rounded division. RADV lowers
    // even precise constant division to reciprocal multiplication (24/1024 codes differ by one ulp).
    // Correct the quotient with its fused residual; all 1024 normalized codes then match bit-for-bit.
    return fma(fma(value, vec3(-1023.0), code), vec3(1.0 / 1023.0), value);
}
