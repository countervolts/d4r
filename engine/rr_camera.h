// Preset D/E camera transforms in NGX 310.7's single-precision operation order.
// Keeping the inversions and reprojection products matters: their near-zero residues feed specular motion history.
#pragma once
#include <array>
#include <cmath>

namespace d4r::rr_camera {
using Matrix = std::array<float, 16>;
inline Matrix multiply(const Matrix& a, const Matrix& b) {
    Matrix r{};
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
            float v = a[4 * i] * b[j];
            v += a[4 * i + 1] * b[4 + j];
            v += a[4 * i + 2] * b[8 + j];
            v += a[4 * i + 3] * b[12 + j];
            r[4 * i + j] = v;
        }
    return r;
}
inline Matrix inverse(const Matrix& m) {
    Matrix r{};
    float determinant = 0;
    for (int i = 0; i < 4; ++i) {
        determinant = 0;
        for (int j = 0; j < 4; ++j) {
            float t[9]; int n = 0;
            for (int row = 0; row < 4; ++row)
                if (row != i)
                    for (int col = 0; col < 4; ++col)
                        if (col != j) t[n++] = m[4 * row + col];
            float cofactor = (t[4] * t[8] - t[7] * t[5]) * t[0];
            cofactor -= (t[3] * t[8] - t[6] * t[5]) * t[1];
            cofactor += (t[3] * t[7] - t[6] * t[4]) * t[2];
            cofactor *= ((i + j) & 1) ? -1.0f : 1.0f;
            r[4 * j + i] = cofactor;
            determinant += cofactor * m[4 * i + j];
        }
    }
    // The final row's determinant and one division, followed by sixteen multiplies, match the reference.
    if (determinant == 0 || !std::isfinite(determinant))
        return Matrix{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    const float reciprocal = 1.0f / determinant;
    for (float& v : r) v *= reciprocal;
    return r;
}
struct Transforms { Matrix previousClip, inverseClip, inversePixels, previousPixels; };
inline Transforms transforms(const Matrix& view, const Matrix& clip, const Matrix& previousView,
                             const Matrix& previousClip, float width, float height) {
    Matrix camera = inverse(view), previousCamera = inverse(previousView);
    for (int i = 12; i < 15; ++i) {
        previousCamera[i] -= camera[i];
        camera[i] = 0.0f;
    }
    const Matrix current = multiply(inverse(camera), clip);
    const Matrix previous = multiply(inverse(previousCamera), previousClip);
    const Matrix inverted = inverse(current);
    const Matrix pixels{width * 0.5f, 0, 0, 0, 0, height * -0.5f, 0, 0,
                        0, 0, 1, 0, width * 0.5f, height * 0.5f, 0, 1};
    return {previous, inverted, inverse(multiply(current, pixels)),
            multiply(multiply(current, multiply(inverted, previous)), pixels)};
}
} // namespace d4r::rr_camera
