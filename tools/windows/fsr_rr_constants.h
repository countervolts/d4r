#pragma once
#include <cstdint>

struct D4rFsrRrPrepareConstants
{
    float clipToView[16];
    float viewToPreviousView[16];
    float renderSize[4];
    float motion[4];
    uint32_t colorDepthBase[4];
    uint32_t motionBaseFlags[4];
    float depthParams[4];
    float detail[4];
};
static_assert(sizeof(D4rFsrRrPrepareConstants) == 224, "fsr_rr_prepare.comp push constants");

enum D4rFsrRrPrepareFlags : uint32_t
{
    kFsrRrHardwareDepth = 1u,
    kFsrRrRoughnessPacked = 2u,
    kFsrRrSpecularHit = 4u,
    kFsrRrSpecularHitInW = 8u,
    kFsrRrDiffuseHit = 16u,
    kFsrRrDiffuseHitInW = 32u,
    kFsrRrNormalsUnorm = 64u,
    kFsrRrBeforeParticles = 128u,
    kFsrRrInvalidNormalPassthrough = 256u,
};

