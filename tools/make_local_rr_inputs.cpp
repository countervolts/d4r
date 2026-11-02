// Recreate the official NVIDIA RR probe's input exactly, as d4r-rr's input directory.
//
// tools/bench_official_ngx_cuda.cpp renders one static noisy fsr_rr_scene frame
//   camera eye (-0.6, 0.6, -1.5), target (0, -0.1, 6.5), projection(fov=1, aspect=iw/ih, near=0.1, far=1000),
//   fsr_scene::render(scene, iw, ih, camera, camera /* same camera -> zero motion */, projection, frameIndex=0)
// and feeds: color RGBA16F, motion RG16F (scene.motionPixels), linear depth R32F with 1e7 where the trace missed,
// normals+packed roughness RGBA16F (scene.normalsRoughness), linear diffuse/specular albedo RGBA8 UNORM,
// specular hit distance R32F constant 4, exposure 1, view/projection matrices, then reuses that one frame with
// Reset=1 on frame 0 only and jitter X = +-0.25 by frame%2, Y = +-0.25 by frame%4<2.
//
// usage: make_rr_inputs OUT_DIR RENDER_W RENDER_H OUTPUT_W OUTPUT_H ENTRIES
// ENTRIES frames.txt lines: the first resets, the rest continue the history; every entry reuses frame 0's files.
#include "../../../tools/fsr_rr_scene.h"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static void write(const std::string& path, const void* data, size_t bytes) {
    std::ofstream out(path, std::ios::binary);
    if (!out || !out.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes)))
        { std::fprintf(stderr, "write %s failed\n", path.c_str()); std::exit(1); }
}

int main(int argc, char** argv) {
    if (argc != 7) { std::fprintf(stderr, "usage: make_rr_inputs OUT RW RH OW OH ENTRIES\n"); return 2; }
    const std::string outDir = argv[1];
    const int iw = std::atoi(argv[2]), ih = std::atoi(argv[3]), ow = std::atoi(argv[4]), oh = std::atoi(argv[5]);
    const int entries = std::atoi(argv[6]);
    fsr_scene::Frame scene;
    const fsr_scene::Camera camera{{-0.6f, 0.6f, -1.5f}, {0.f, -0.1f, 6.5f}};
    const auto projection = fsr_scene::projection_matrix(1.f, float(iw) / ih, 0.1f, 1000.f);
    const auto view = fsr_scene::view_matrix(camera.eye, camera.target);
    fsr_scene::render(scene, iw, ih, camera, camera, projection, 0);
    std::vector<float> depth = scene.depth;
    for (size_t i = 0; i < depth.size(); ++i) if (!scene.valid[i]) depth[i] = 1.e7f;   // as the probe: sky outside the bounds
    const size_t px = size_t(iw) * ih;
    write(outDir + "/0-color.bin", scene.color.data(), px * 8);
    write(outDir + "/0-motion.bin", scene.motionPixels.data(), px * 4);
    write(outDir + "/0-depth.bin", depth.data(), px * 4);
    write(outDir + "/0-normals.bin", scene.normalsRoughness.data(), px * 8);      // packed mode: w carries roughness
    write(outDir + "/0-roughness.bin", scene.normalsRoughness.data(), px * 8);    // the probe supplies no separate texture
    write(outDir + "/0-diffuse.bin", scene.diffuseAlbedoLinear.data(), px * 4);
    write(outDir + "/0-specular.bin", scene.specularAlbedoLinear.data(), px * 4);
    std::vector<float> hit(px, 4.f);
    write(outDir + "/0-hitdistance.bin", hit.data(), px * 4);
    std::ofstream frames(outDir + "/frames.txt");
    frames << "size " << iw << " " << ih << " " << ow << " " << oh << "\n";
    for (int f = 0; f < entries; ++f) {
        const float jx = f % 2 ? 0.25f : -0.25f, jy = f % 4 < 2 ? 0.25f : -0.25f;
        frames << "frame 0 " << jx << " " << jy << " 1 1 1 " << (f == 0 ? 1 : 0) << " 4";   // bit 2: packed roughness
        for (int i = 0; i < 16; ++i) frames << ' ' << view.m[i / 4][i % 4];
        for (int i = 0; i < 16; ++i) frames << ' ' << projection.m[i / 4][i % 4];
        frames << "\n";
    }
    std::printf("%s: one static %dx%d frame, %d frames.txt entries, color %zu B motion %zu B depth %zu B\n",
                outDir.c_str(), iw, ih, entries, px * 8, px * 4, px * 4);
    return 0;
}
