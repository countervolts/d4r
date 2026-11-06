// NVIDIA's f32 activations of the head (rrlite_dec0_4x4's output code): a sigmoid rounded to binary16, and a
// tanh that is a polynomial below 0.6 and 1 - 2 / (e^2x + 1) above
// (RADV folds a register f32 -> f16 -> f32 conversion away: round_half.glsl's integer rounding keeps it)
float sigmoidL(float v) { return roundHalfValue(1.0 / (exp2(v * -1.442695) + 1.0)); }
float tanhL(float t)
{
    const float m = abs(t), s = t * t;
    const float big = m >= 9.0109138 ? 1.0 : fma(1.0 / (exp2(m * 2.88539) + 1.0), -2.0, 1.0);
    return m >= 0.6 ? (t < 0.0 ? -big : big) :
        fma(fma(fma(fma(fma(0.01573968306183815, s, -0.05230396240949631), s, 0.13315297663211823), s, -0.33332768082618713), s, 0.0), t, t);
}
// the block output channel of each head input: NGX's pair-order columns 0, 1, 6 (gates), 2-5 (feature), 7-9 (mix),
// 10-12 (covariance a, b, rho) - preset M's columns without its four feature gates - in natural channel order
#ifndef LH_GATE0
#define LH_GATE0 0
#define LH_GATE1 1
#define LH_GATE2 12
#define LH_MIX0 13
#define LH_MIX1 16
#define LH_MIX2 17
#define LH_COVA 20
#define LH_COVB 21
#define LH_RHO 24
#define LH_FEAT0 4
#define LH_FEAT1 5
#define LH_FEAT2 8
#define LH_FEAT3 9
#endif
