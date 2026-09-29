// No host SDK, CRT, or device library dependency. Compiled only for gfx1201.
extern "C" __attribute__((global)) void d4r_probe_pattern(
    unsigned int* output, unsigned int count, unsigned int seed)
{
    const unsigned int i = __builtin_amdgcn_workgroup_id_x() * 128u +
                           __builtin_amdgcn_workitem_id_x();
    if (i < count)
        output[i] = (i * 1664525u + seed) ^ 0xa5a55a5au;
}

extern "C" __attribute__((global)) void d4r_probe_interop(
    unsigned int* shared, unsigned int count, unsigned int seed, unsigned int* errors)
{
    const unsigned int i = __builtin_amdgcn_workgroup_id_x() * 128u +
                           __builtin_amdgcn_workitem_id_x();
    if (i < count) {
        const unsigned int expected = (i * 1664525u + seed) ^ 0xa5a55a5au;
        if (shared[i] != expected) __atomic_fetch_add(errors, 1u, __ATOMIC_RELAXED);
        shared[i] ^= 0x3579bdf1u;
    }
}
