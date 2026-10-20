"""Exercise the shim's actual app-ID retry block with a deterministic NGX provider."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which('c++'), 'needs C++ compiler')
class NgxIdentityRetryTests(unittest.TestCase):
    def test_capability_rejection_shutdown_and_identity_boundaries(self):
        shim = (ROOT / 'tools/d4r_nvngx_shim.cpp').read_text()
        block = shim[shim.index('    const bool hasProject = project != nullptr && !project->id.empty();'):]
        block = block[:block.index('    using LoadErrorFn')]
        source = r'''
#include <cassert>
#include <string>
#include <vector>
constexpr unsigned NGX_SUCCESS = 1;
struct Project { std::string id; };
struct Worker { template<class F> auto call(F f) { return f(); } };
static std::vector<char> events;
static int available, featureInit, needsDriver;
static unsigned capResult, projectResult;
static bool nullCaps, known;
static unsigned caps(void** p) { events.push_back('c'); *p = nullCaps ? nullptr : &available; return capResult; }
static unsigned destroy(void*) { events.push_back('d'); return 1; }
static unsigned shutdown() { events.push_back('s'); return 1; }
static unsigned initProject(const char*, int, const char*, const wchar_t*, unsigned, void*) {
    events.push_back('p'); return projectResult;
}
static unsigned d4r_ngx_get_int(void*, const char* name, int* out) {
    std::string key(name);
    *out = key == "SuperSampling.Available" ? available :
           key == "SuperSampling.FeatureInitResult" ? featureInit : needsDriver;
    return known ? 1 : 0;
}
template<class... T> void logf(const char*, T...) {}
struct State {
    Worker worker;
    struct {
        unsigned (*getCapabilityParameters)(void**) = caps;
        unsigned (*destroyParameters)(void*) = destroy;
        unsigned (*shutdown)() = ::shutdown;
        unsigned (*initProjectId)(const char*, int, const char*, const wchar_t*, unsigned, void*) = initProject;
    } ngx;
    std::wstring dataPath = L".";
    int featureInfo = 0;
} g;
static unsigned run(unsigned result, Project* project = nullptr) {
    unsigned long long id = 123;
    unsigned apiVersion = 0x15, version = 0x15;
    BLOCK
    return result;
}
int main() {
    auto reset = [] {
        events.clear(); available = 1; featureInit = 1; needsDriver = 0;
        capResult = projectResult = 1; nullCaps = false; known = true;
        g.ngx.initProjectId = initProject;
    };
    reset(); assert(run(1) == 1); assert((events == std::vector<char>{'c','d'}));
    reset(); assert(run(0xbad0000cU) == 1); assert((events == std::vector<char>{'p'}));
    reset(); available = 0; featureInit = int(0xbad00004U);
    assert(run(1) == 1); assert((events == std::vector<char>{'c','d','s','p'}));
    reset(); featureInit = int(0xbad00004U); projectResult = 0xbad00002U;
    assert(run(1) == 0xbad00002U); assert((events == std::vector<char>{'c','d','s','p'}));
    reset(); needsDriver = 1; available = 0;
    assert(run(1) == 1); assert((events == std::vector<char>{'c','d'}));
    reset(); known = false; available = 0;
    assert(run(1) == 1); assert((events == std::vector<char>{'c','d'}));
    reset(); capResult = 0xbad00002U;
    assert(run(1) == 1); assert((events == std::vector<char>{'c'}));
    reset(); nullCaps = true; assert(run(1) == 1); assert((events == std::vector<char>{'c'}));
    reset(); Project explicitProject{"game-owned"};
    assert(run(0xbad0000cU, &explicitProject) == 0xbad0000cU); assert(events.empty());
    reset(); g.ngx.initProjectId = nullptr;
    assert(run(0xbad0000cU) == 0xbad0000cU); assert(events.empty());
}
'''.replace('    BLOCK', block)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / 'retry.cpp').write_text(source)
            result = subprocess.run(['c++', '-std=c++20', '-Wall', '-Wextra', '-Werror',
                                     str(path / 'retry.cpp'), '-o', str(path / 'retry')], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            subprocess.run([str(path / 'retry')], check=True)


if __name__ == '__main__':
    unittest.main()
