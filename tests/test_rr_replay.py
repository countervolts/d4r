"""Ray Reconstruction capture and replay: the exact guides travel, or the replay fails.

The shim's input dump records the guides a game bound, in the formats the game passed them,
and the harness replay feeds those back instead of its own synthetic planes. Both are checked
here against the real source, because the failure they exist to prevent is a plausible image
produced from fewer guides than the game rendered.
"""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def extract(source, start_marker, end_marker):
    start = source.index(start_marker)
    return source[start:source.index(end_marker, start)]


def build_and_run(body, name, with_directory=False):
    with tempfile.TemporaryDirectory() as temporary:
        runner = Path(temporary) / name
        subprocess.run(['c++', '-std=c++17', '-Wall', '-Wextra', '-Werror',
                        '-x', 'c++', '-', '-o', str(runner)], input=body,
                       text=True, capture_output=True, check=True)
        # The harness joins its frame prefix with a backslash, which is one file name on this
        # platform; the capture directory therefore holds files named "capture\frame-NNNNNN-...".
        capture = Path(temporary) / 'capture'
        capture.mkdir()
        command = [str(runner), str(capture)] if with_directory else [str(runner)]
        subprocess.run(command, check=True, timeout=10)


class AuxCaptureTests(unittest.TestCase):
    """dump_aux_guides must record the caller's bytes and the descriptor that reads them."""

    def test_capture_records_every_guide_in_the_callers_format(self):
        shim = (ROOT / 'tools/d4r_nvngx_shim.cpp').read_text()
        tables = extract(shim, 'struct AuxScalar', 'struct FrameParams')
        # The forward declaration in that slice has no definition here and nothing uses it.
        tables = tables.replace(
            'static unsigned int get_uint_or(void* parameters, const char* name, unsigned int fallback);\n', '')
        capture = extract(shim, 'static void dump_aux_guides', '// Worker. Points the denoiser')
        prelude = r'''
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <cassert>
using UINT = unsigned;
using DXGI_FORMAT = unsigned;
constexpr int MAX_PATH = 260;
constexpr int kSlots = 4;
enum class AuxLayout { Verbatim, Converted };
struct Staging
{
    uint8_t* mapped = nullptr;
    struct { size_t Offset; struct { size_t RowPitch; } Footprint; } layout = {};
};
struct AuxSurface
{
    Staging staging[kSlots];
    uint32_t texelBytes = 0;
    UINT width = 0, height = 0;
    DXGI_FORMAT format = 0;
    AuxLayout how = AuxLayout::Verbatim;
};
struct AuxBinding
{
    AuxSurface* surface = nullptr;
    std::vector<const char*> names;
    std::vector<const char*> sources;
};
struct InputSlot { std::vector<AuxBinding> auxBindings; };
struct Feature { InputSlot inputs[kSlots]; };
''' + tables + r'''
struct FrameParams
{
    int auxScalar[sizeof(kAuxScalars) / sizeof(kAuxScalars[0])] = {};
    unsigned int auxSubrect[sizeof(kAuxSubrectNames) / sizeof(kAuxSubrectNames[0])][2] = {};
    float auxMatrix[sizeof(kAuxMatrices) / sizeof(kAuxMatrices[0])][16] = {};
    bool auxMatrixSet[sizeof(kAuxMatrices) / sizeof(kAuxMatrices[0])] = {};
};
static void logf(const char*, ...) {}
static std::string join_aux_names(const std::vector<const char*>& names)
{
    std::string joined;
    for (const char* name : names)
        joined += joined.empty() ? name : std::string(", ") + name;
    return joined;
}
static bool input_dump_directory(uint32_t, char (&directory)[MAX_PATH])
{
    std::snprintf(directory, MAX_PATH, "capture");
    return true;
}
static uint8_t* capture_pinned(const char*, size_t) { return nullptr; }
struct Written { std::string path; std::vector<uint8_t> bytes; };
static std::vector<Written> g_written;
static void write_capture_file(uint32_t, const char*, const char* path, const void* data, size_t bytes)
{
    const uint8_t* first = static_cast<const uint8_t*>(data);
    g_written.push_back({path, std::vector<uint8_t>(first, first + bytes)});
}
'''
        program = prelude + capture + r'''
static const char* text_of(const std::string& path)
{
    for (const Written& written : g_written)
        if (written.path == path)
            return reinterpret_cast<const char*>(written.bytes.data());
    return nullptr;
}
int main()
{
    Feature feature;
    InputSlot& slot = feature.inputs[0];
    // A 3x2 BGRA8 guide: the shim converts this to float32 on upload, so the capture must hold
    // the caller's four bytes per texel, not the array's sixteen.
    constexpr size_t rowPitch = 3 * 4 + 8; // padded, as a readback heap is
    std::vector<uint8_t> bgra(rowPitch * 2), depth(3 * 2 * 4);
    for (size_t i = 0; i < bgra.size(); ++i) bgra[i] = static_cast<uint8_t>(i + 1);
    for (size_t i = 0; i < depth.size(); ++i) depth[i] = static_cast<uint8_t>(200 - i);
    AuxSurface surfaces[2];
    surfaces[0].staging[0].mapped = bgra.data();
    surfaces[0].staging[0].layout.Offset = 0;
    surfaces[0].staging[0].layout.Footprint.RowPitch = rowPitch;
    surfaces[0].texelBytes = 4;
    surfaces[0].width = 3;
    surfaces[0].height = 2;
    surfaces[0].format = 87; // DXGI_FORMAT_B8G8R8A8_UNORM
    surfaces[0].how = AuxLayout::Converted;
    surfaces[1].staging[0].mapped = depth.data();
    surfaces[1].staging[0].layout.Offset = 0;
    surfaces[1].staging[0].layout.Footprint.RowPitch = 3 * 4;
    surfaces[1].texelBytes = 4;
    surfaces[1].width = 3;
    surfaces[1].height = 2;
    surfaces[1].format = 87;
    surfaces[1].how = AuxLayout::Converted;
    slot.auxBindings.push_back({&surfaces[0], {"GBuffer.Normals"}, {"GBuffer.Normals"}});
    slot.auxBindings.push_back({&surfaces[1], {"DLSS.Input.DiffuseAlbedo"}, {"GBuffer.DiffuseAlbedo",
                                                                            "DLSS.Input.DiffuseAlbedo"}});
    FrameParams params;
    params.auxScalar[0] = 1;
    params.auxSubrect[0][0] = 5;
    params.auxSubrect[0][1] = 7;
    params.auxMatrixSet[0] = true;
    params.auxMatrix[0][0] = 1.5f;
    params.auxMatrix[0][5] = -2.25f;
    dump_aux_guides(feature, 0, 42, params);

    // Two planes and one descriptor, named for the frame.
    assert(g_written.size() == 3);
    const std::string plane0 = g_written[0].path, plane1 = g_written[1].path;
    assert(plane0.find("frame-000042-guide-00") != std::string::npos);
    assert(plane1.find("frame-000042-guide-01") != std::string::npos);
    assert(g_written[2].path.find("frame-000042-guides.txt") != std::string::npos);

    // Tightly packed caller bytes: the padded staging row must not leak into the capture.
    assert(g_written[0].bytes.size() == 3 * 2 * 4);
    for (UINT row = 0; row < 2; ++row)
        for (UINT x = 0; x < 3; ++x)
            assert(g_written[0].bytes[(row * 3 + x) * 4] == bgra[row * rowPitch + x * 4]);

    // The descriptor names both spellings, the format, the geometry, the caller's texel size and
    // whether the shim converts it.
    const char* text = text_of(g_written[2].path);
    assert(text != nullptr);
    assert(std::string(text).find("guides 2\n") == 0);
    assert(std::string(text).find("guide 1 GBuffer.DiffuseAlbedo,DLSS.Input.DiffuseAlbedo "
                                  "DLSS.Input.DiffuseAlbedo 87 3 2 4 1") != std::string::npos);
    assert(std::string(text).find("scalar DLSS.Denoise.Mode 1") != std::string::npos);
    assert(std::string(text).find("subrect DLSS.Input.DiffuseAlbedo.Subrect.Base.X 5") != std::string::npos);
    assert(std::string(text).find("subrect DLSS.Input.DiffuseAlbedo.Subrect.Base.Y 7") != std::string::npos);
    assert(std::string(text).find("matrix WorldToViewMatrix 1 1.5") != std::string::npos);
    assert(std::string(text).find("matrix ViewToClipMatrix 0") != std::string::npos);

    // A guide whose bytes cannot be read fails the whole capture: a descriptor without its plane
    // is an image made of fewer guides than the game gave, and nothing downstream would show it.
    g_written.clear();
    surfaces[1].staging[0].mapped = nullptr;
    dump_aux_guides(feature, 0, 43, params);
    assert(g_written.empty());
}
'''
        build_and_run(program, 'rr-guide-capture')


class ReplayGuideLoadTests(unittest.TestCase):
    """A Ray Reconstruction replay must load the recorded guides, or refuse the run."""

    def test_missing_or_malformed_guides_fail_instead_of_defaulting(self):
        harness = (ROOT / 'tools/d3d12_dlss_harness.cpp').read_text()
        # The guide loader, the file reader it needs and the name helpers; the frame-level
        # parameter loader is covered by ReplayParamsTests instead.
        replay = (extract(harness, 'struct ReplayGuide', 'static bool read_replay_params') +
                  extract(harness, 'static std::vector<std::string> split_names', 'static float halton'))
        program = r'''
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>
#include <cassert>
using DXGI_FORMAT = unsigned;
constexpr DXGI_FORMAT DXGI_FORMAT_UNKNOWN = 0;
struct ID3D12Resource;
constexpr int MAX_PATH = 260;
''' + replay + r'''
// join_names comes along with the loader and nothing here calls it.
[[maybe_unused]] static std::string unused_join(const std::vector<std::string>& names) { return join_names(names); }
static void write_bytes(const std::string& path, const std::vector<uint8_t>& bytes)
{
    FILE* file = std::fopen(path.c_str(), "wb");
    assert(file != nullptr);
    assert(std::fwrite(bytes.data(), 1, bytes.size(), file) == bytes.size());
    std::fclose(file);
}
static void write_text(const std::string& path, const std::string& text)
{
    write_bytes(path, std::vector<uint8_t>(text.begin(), text.end()));
}
int main(int, char** argv)
{
    const std::string directory = argv[1];
    ReplayFrame frame;

    // No descriptor at all: an RR replay without the game's guides is refused, not invented.
    assert(!load_replay_guides(directory.c_str(), 1, frame));

    // A descriptor whose guide plane is missing is refused too.
    write_text(directory + "\\frame-000002-guides.txt",
               "guides 1\nguide 0 GBuffer.Normals GBuffer.Normals 6 2 2 4 0\n");
    assert(!load_replay_guides(directory.c_str(), 2, frame));

    // The plane present, the count disagreeing with the file: refused rather than partly read.
    write_bytes(directory + "\\frame-000002-guide-00.bin", std::vector<uint8_t>(2 * 2 * 4, 7));
    write_text(directory + "\\frame-000002-guides.txt",
               "guides 2\nguide 0 GBuffer.Normals GBuffer.Normals 6 2 2 4 0\n");
    assert(!load_replay_guides(directory.c_str(), 2, frame));

    // A complete descriptor: the guides, the settings, both Subrect axes and the matrices.
    write_text(directory + "\\frame-000002-guides.txt",
               "guides 1\n"
               "guide 0 GBuffer.DiffuseAlbedo,DLSS.Input.DiffuseAlbedo DLSS.Input.DiffuseAlbedo 87 2 2 4 1\n"
               "scalar DLSS.Denoise.Mode 1\n"
               "subrect DLSS.Input.Normals.Subrect.Base.X 4\n"
               "subrect DLSS.Input.Normals.Subrect.Base.Y 9\n"
               "matrix WorldToViewMatrix 1 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16\n"
               "matrix ViewToClipMatrix 0\n");
    write_bytes(directory + "\\frame-000002-guide-00.bin", std::vector<uint8_t>(2 * 2 * 4, 3));
    assert(load_replay_guides(directory.c_str(), 2, frame));
    assert(frame.guides.size() == 1);
    assert(frame.guides[0].sources.size() == 2);
    assert(frame.guides[0].sources[0] == "GBuffer.DiffuseAlbedo");
    assert(frame.guides[0].sources[1] == "DLSS.Input.DiffuseAlbedo");
    assert(frame.guides[0].publish[0] == "DLSS.Input.DiffuseAlbedo");
    assert(frame.guides[0].format == 87u && frame.guides[0].width == 2 && frame.guides[0].height == 2);
    assert(frame.guides[0].texelBytes == 4 && frame.guides[0].converted);
    assert(frame.guides[0].bytes.size() == 2 * 2 * 4 && frame.guides[0].bytes[0] == 3);
    assert(frame.scalars.size() == 1 && frame.scalars[0].first == "DLSS.Denoise.Mode" &&
           frame.scalars[0].second == 1);
    assert(frame.subrects.size() == 2 && frame.subrects[0].second == 4 && frame.subrects[1].second == 9);
    assert(frame.subrects[1].first == "DLSS.Input.Normals.Subrect.Base.Y");
    assert(frame.matrices.size() == 2 && frame.matrixPresent.size() == 2);
    assert(frame.matrices[0].second[0] == 1.0f && frame.matrices[0].second[15] == 16.0f);
    assert(frame.matrixPresent[0].second);
    // A matrix the game did not register stays absent, so the replay clears it rather than
    // leaving the previous frame's camera in place.
    assert(!frame.matrixPresent[1].second);
}
'''
        with tempfile.TemporaryDirectory() as temporary:
            # The harness joins the frame prefix with a backslash; on this platform that is one
            # file name inside the directory, which is what the capture names are.
            capture = Path(temporary) / 'capture'
            capture.mkdir()
            runner = Path(temporary) / 'rr-replay-load'
            subprocess.run(['c++', '-std=c++17', '-Wall', '-Wextra', '-Werror',
                            '-x', 'c++', '-', '-o', str(runner)], input=program,
                           text=True, capture_output=True, check=True)
            subprocess.run([str(runner), str(capture)], check=True, timeout=10)


class ReplayParamsTests(unittest.TestCase):
    """The exposure plane, indicator axes, Output.Subrect origin and creation state.

    Each is something the replay must take from the capture rather than invent: a substituted
    exposure value, a default origin or a guessed quality mode is a comparison against a reference
    the game never asked for.
    """

    def build(self, body, name):
        harness = (ROOT / 'tools/d3d12_dlss_harness.cpp').read_text()
        source = (extract(harness, 'struct ReplayGuide', 'struct ReplayTexture') +
                  extract(harness, 'struct ReplayFrame', 'static bool read_file') +
                  extract(harness, 'static bool read_file', '// Splits a comma-separated name list'))
        prelude = r'''
#include <array>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>
#include <cassert>
using DXGI_FORMAT = unsigned;
constexpr DXGI_FORMAT DXGI_FORMAT_UNKNOWN = 0;
struct ID3D12Resource;
constexpr int MAX_PATH = 260;
'''
        common = r'''
static void write_bytes(const std::string& path, const std::vector<uint8_t>& bytes)
{
    FILE* file = std::fopen(path.c_str(), "wb");
    assert(file != nullptr);
    assert(std::fwrite(bytes.data(), 1, bytes.size(), file) == bytes.size());
    std::fclose(file);
}
static void write_text(const std::string& path, const std::string& text)
{
    write_bytes(path, std::vector<uint8_t>(text.begin(), text.end()));
}
static void write_planes(const std::string& directory, uint32_t frame, unsigned width, unsigned height)
{
    char path[MAX_PATH];
    std::snprintf(path, sizeof(path), "%s\\frame-%06u-", directory.c_str(), frame);
    write_bytes(std::string(path) + "color.rgba16f", std::vector<uint8_t>(width * height * 8, 1));
    write_bytes(std::string(path) + "depth.r32f", std::vector<uint8_t>(width * height * 4, 2));
    write_bytes(std::string(path) + "motion.rg16f", std::vector<uint8_t>(width * height * 4, 3));
}
'''
        build_and_run(prelude + source + common + body, name, with_directory=True)

    def test_exposure_geometry_and_creation_state_come_from_the_capture(self):
        self.build(r'''
int main(int, char** argv)
{
    const std::string directory = argv[1];
    ReplayFrame frame;

    // A 1x1 exposure scalar: the plane is four bytes, not the harness's own 1.0.
    write_planes(directory, 1, 4, 2);
    write_bytes(directory + "\\frame-000001-exposure.r32f", std::vector<uint8_t>{0, 0, 0x80, 0x3f});
    write_text(directory + "\\frame-000001-params.txt",
               "has_exposure 1\n"
               "exposure_size 1 1\n"
               "invert 1 1\n"
               "output_base 7 9\n"
               "create 1280 720 2560 1440 2 73 1 5\n");
    assert(load_replay_frame(directory.c_str(), 1, 4 * 2 * 8, 4 * 2 * 4, 4 * 2 * 4, frame));
    assert(frame.hasExposure == 1);
    assert(frame.exposureWidth == 1 && frame.exposureHeight == 1);
    assert(frame.exposure.size() == 4);
    uint32_t exposureBits = 0;
    std::memcpy(&exposureBits, frame.exposure.data(), sizeof(exposureBits));
    assert(exposureBits == 0x3f800000u); // 1.0f, the value the capture wrote
    assert(frame.invertX == 1 && frame.invertY == 1);
    assert(frame.outputBaseX == 7 && frame.outputBaseY == 9);
    assert(frame.creation.recorded);
    assert(frame.creation.width == 1280 && frame.creation.height == 720);
    assert(frame.creation.outWidth == 2560 && frame.creation.outHeight == 1440);
    assert(frame.creation.quality == 2 && frame.creation.flags == 73);
    assert(frame.creation.outputSubrects == 1 && frame.creation.preset == 5);

    // A per-pixel exposure buffer, in its own geometry.
    ReplayFrame wide;
    write_planes(directory, 2, 4, 2);
    write_bytes(directory + "\\frame-000002-exposure.r32f", std::vector<uint8_t>(4 * 2 * 4, 9));
    write_text(directory + "\\frame-000002-params.txt",
               "has_exposure 1\nexposure_size 4 2\ncreate 1280 720 2560 1440 2 73 1 5\n");
    assert(load_replay_frame(directory.c_str(), 2, 4 * 2 * 8, 4 * 2 * 4, 4 * 2 * 4, wide));
    assert(wide.exposureWidth == 4 && wide.exposureHeight == 2 && wide.exposure.size() == 4 * 2 * 4);

    // No exposure texture bound: nothing is read, and none is invented.
    ReplayFrame none;
    write_planes(directory, 3, 4, 2);
    write_text(directory + "\\frame-000003-params.txt",
               "has_exposure 0\nexposure_size 0 0\ncreate 1280 720 2560 1440 2 73 0 5\n");
    assert(load_replay_frame(directory.c_str(), 3, 4 * 2 * 8, 4 * 2 * 4, 4 * 2 * 4, none));
    assert(none.exposure.empty() && none.creation.outputSubrects == 0);
}
''', 'rr-replay-params')

    def test_unreplayable_exposure_and_truncated_params_fail(self):
        self.build(r'''
int main(int, char** argv)
{
    const std::string directory = argv[1];
    ReplayFrame frame;

    // A claimed exposure texture whose geometry the capture never recorded: the replay would have
    // to invent both the size and the contents.
    write_planes(directory, 1, 4, 2);
    write_bytes(directory + "\\frame-000001-exposure.r32f", std::vector<uint8_t>(4, 0));
    write_text(directory + "\\frame-000001-params.txt", "has_exposure 1\nexposure_size 0 0\n");
    assert(!load_replay_frame(directory.c_str(), 1, 4 * 2 * 8, 4 * 2 * 4, 4 * 2 * 4, frame));

    // Geometry recorded but the plane absent, then present at the wrong size.
    write_planes(directory, 2, 4, 2);
    write_text(directory + "\\frame-000002-params.txt", "has_exposure 1\nexposure_size 4 4\n");
    assert(!load_replay_frame(directory.c_str(), 2, 4 * 2 * 8, 4 * 2 * 4, 4 * 2 * 4, frame));
    write_bytes(directory + "\\frame-000002-exposure.r32f", std::vector<uint8_t>(4 * 4 * 2, 0));
    assert(!load_replay_frame(directory.c_str(), 2, 4 * 2 * 8, 4 * 2 * 4, 4 * 2 * 4, frame));

    // A truncated parameter is refused rather than left at its default.
    write_planes(directory, 3, 4, 2);
    write_text(directory + "\\frame-000003-params.txt", "has_exposure 0\noutput_base 4\n");
    assert(!load_replay_frame(directory.c_str(), 3, 4 * 2 * 8, 4 * 2 * 4, 4 * 2 * 4, frame));

    // A capture with no creation state is reported as such, so the run can refuse to guess.
    ReplayFrame legacy;
    write_planes(directory, 4, 4, 2);
    write_text(directory + "\\frame-000004-params.txt", "has_exposure 0\n");
    assert(load_replay_frame(directory.c_str(), 4, 4 * 2 * 8, 4 * 2 * 4, 4 * 2 * 4, legacy));
    assert(!legacy.creation.recorded);
}
''', 'rr-replay-params-fail')

if __name__ == '__main__':
    unittest.main()