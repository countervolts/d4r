// Whole dec0 windows needed by K reconstruction and the next frame's bilinear token history.
// Tensor dimensions and the kernel's mirrored reads are unchanged. Padding outside this span
// is unspecified: callers comparing the complete padded head must request the original grid.
#pragma once
#include <stdint.h>

static inline uint32_t d4rKHeadWindowCount(uint32_t tokens, uint32_t shift, uint32_t outputPixels)
{
    // ceil(output/4) visible tokens, plus the history sample's neighbouring token and one
    // conservative guard. The last whole window supplies all of its own attention inputs.
    uint32_t needed = (outputPixels + 3u) / 4u + 2u;
    if (needed > tokens) needed = tokens;
    return (needed + shift + 7u) / 8u;
}
