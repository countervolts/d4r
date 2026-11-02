int main() {
    DEVPROP pr; GETPROP(&pr, 0); int sms = pr.multiProcessorCount; printf("%s units=%d\n", pr.name, sms);
    float* o; MALLOC(&o, 4096); EVENT e0, e1; ECREATE(&e0); ECREATE(&e1); float ms;
    const int blocks = sms * 16, thr = 256, it = 40000;
    // source-level operations per iteration and chain (the compilers fuse them differently): fma 1, int 4 (xor, shift, add,
    // add), packed f16 fma 2 (one per half)
    struct { const char* n; void (*k)(float*, int, float, float); double per; } K[] = {
        {"fp32 fma", k_fma, 1}, {"int32 xor/shift/add", k_iadd, 4}, {"packed f16x2 fma", k_h2fma, 2}};
    for (auto& k : K) for (int r = 0; r < 3; ++r) {
        ERECORD(e0); k.k<<<blocks, thr>>>(o, it, 1.0001f, 0.0001f); ERECORD(e1); ESYNC(e1); ELAPSED(&ms, e0, e1);
        const double warps = double(blocks) * thr / 32, ops = warps * it * CH * k.per;   // per lane: x 32 below
        printf("%-22s %6.2f T ops/s\n", k.n, ops * 32 / ms / 1e9);
    }
}
