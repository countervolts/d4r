"""GPU identity and model capability decisions without a GPU or Wine process."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class GpuCompatibilityTests(unittest.TestCase):
    def compile_run(self, source, cxx=False):
        with tempfile.TemporaryDirectory() as t:
            executable = Path(t) / 'check'
            result = subprocess.run(['c++' if cxx else 'cc', '-x', 'c++' if cxx else 'c',
                            '-std=c++17' if cxx else '-std=gnu11', '-Wall', '-Wextra', '-Werror',
                            '-pthread',
                            '-I', str(ROOT / 'tools'), '-o', str(executable), '-'], input=source,
                           capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            subprocess.run([str(executable)], check=True, timeout=5)

    def test_pci_identity_is_not_a_vendor_id_or_gpu_size_guess(self):
        self.compile_run(r'''
#include "d4r_gpu_identity.h"
#include <assert.h>
int main(void) {
    assert(d4r_same_pci("0000:03:00.0", "00000000:03:00.0"));
    assert(!d4r_same_pci("0000:03:00.0", "0000:04:00.0"));
    assert(!d4r_same_pci("0000:03:00.0", "0001:03:00.0"));
    assert(!d4r_same_pci("0000:03:00.0garbage", "0000:03:00.0"));
    assert(d4r_pci_matches_kfd("0000:03:00.0", 0, 0x300));
    assert(!d4r_pci_matches_kfd("0000:03:00.0", 0, 0x400));
    assert(!d4r_pci_matches_kfd("0000:03:00.0", 1, 0x300));
}
''')

    def test_bridge_maps_nonzero_compute_ordinal_and_only_its_luid(self):
        source = (ROOT / 'tools/wine_nvcuda_bridge.c').read_text()
        selection = source[source.index('static CUresult selected_cuda_device('):
                           source.index('static void create_default_zluda_context(')]
        luid = source[source.index('CUresult WINAPI cuDeviceGetLuid('):
                      source.index('CUresult WINAPI cuDeviceGetAttribute(')]
        self.compile_run(r'''
#include "d4r_gpu_identity.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#define WINAPI
typedef int CUresult;
typedef int CUdevice;
typedef int (*CUDEVICEGETCOUNT_FN)(int*);
typedef int (*CUDEVICEGET_FN)(int*, int);
typedef int (*CUDEVICEGETLUID_FN)(char*, unsigned*, int);
enum { CUDA_SUCCESS=0, CUDA_ERROR_NOT_INITIALIZED=3, CUDA_ERROR_NOT_SUPPORTED=801 };
static const char* render = "0000:03:00.0";
static int devices = 2;
static unsigned GetEnvironmentVariableA(const char* key, char* out, unsigned size) {
    (void)key; snprintf(out, size, "%s", render); return (unsigned)strlen(render);
}
static void tracef(const char* fmt, ...) { (void)fmt; }
static int count(int* n) { *n=devices; return 0; }
static int get(int* d,int n) { *d=100+n; return 0; }
static int pci(char* out,int size,int device) {
    snprintf(out, (size_t)size, "%s", device==101 ? "00000000:03:00.0" : "0000:04:00.0"); return 0;
}
static int original_luid(char* out,unsigned* mask,int device) { (void)device; memset(out,0,8);*mask=0;return 0; }
static void* find_zluda_symbol(const char* name) {
    if (!strcmp(name,"cuDeviceGetCount")) return (void*)count;
    if (!strcmp(name,"cuDeviceGet")) return (void*)get;
    if (!strcmp(name,"cuDeviceGetPCIBusId")) return (void*)pci;
    return (void*)original_luid;
}
static int missing(const char* name) { (void)name; return 999; }
static unsigned get_process_u32(const char* name,unsigned fallback) {
    (void)fallback;
    return strstr(name,"LOW") ? 0x3f2 : strstr(name,"HIGH") ? 0 : 1;
}
''' + selection + luid + r'''
int main(void) {
    assert(d4r_pci_matches_kfd("0000:03:00.0",0,0x300));
    int selected=-1; assert(selected_cuda_device(&selected)==0 && selected==101);
    char id[8]; unsigned mask=9;
    assert(cuDeviceGetLuid(id,&mask,100)==0 && mask==0);
    assert(cuDeviceGetLuid(id,&mask,101)==0 && mask==1);
    unsigned low=0;memcpy(&low,id,4);assert(low==0x3f2);
    render="0000:05:00.0";assert(selected_cuda_device(&selected)==801);
    render="";assert(selected_cuda_device(&selected)==801);
    devices=1;assert(selected_cuda_device(&selected)==0 && selected==100);
}
''')

    def test_rr_model_cannot_reach_unsupported_fp8_pipelines(self):
        self.compile_run(r'''
#include "d4r_engine_compat.h"
#include <cassert>
int main() {
    const char* fp8 = "D4RR0001\npreset 4\nstage enc0 x.spv\n";
    const char* wide = "D4RR0001\npreset 4\narithmetic gfx11\nstage enc0 x.spv\n";
    assert(!d4r_rr_model_compatible(fp8, true, false));
    assert(!d4r_rr_model_compatible(wide, false, false));
    assert(d4r_rr_model_compatible(wide, true, false));
    assert(d4r_rr_model_compatible(fp8, true, true));
    assert(!d4r_rr_model_compatible("D4RR0002\npreset 6\n", true, false));
    assert(!d4r_rr_model_compatible("D4RR0001\narithmetic unknown\n", true, true));
    VkCooperativeMatrixPropertiesKHR p{};
    p.MSize=p.NSize=p.KSize=16; p.scope=VK_SCOPE_SUBGROUP_KHR;
    p.AType=p.BType=VK_COMPONENT_TYPE_FLOAT16_KHR;
    p.CType=p.ResultType=VK_COMPONENT_TYPE_FLOAT32_KHR;
    assert(d4r_matrix_tuple(p, VK_COMPONENT_TYPE_FLOAT16_KHR));
    p.KSize=32; assert(!d4r_matrix_tuple(p, VK_COMPONENT_TYPE_FLOAT16_KHR));
}
''', cxx=True)

    def test_existing_context_is_preserved_and_bootstrap_binds_each_thread(self):
        text = (ROOT / 'tools/wine_nvcuda_bridge.c').read_text()
        ensure = text[text.index('static void ensure_context(void)'):text.index('/* D4R_NGX_STREAM_PRIORITY')]
        self.compile_run(r'''
#include <assert.h>
#include <pthread.h>
#include <stdint.h>
#include <string.h>
typedef int CUresult;
typedef void* CUcontext;
typedef CUresult (*CUCTX_CURRENT_FN)(CUcontext);
enum { CUDA_SUCCESS=0 };
static pthread_once_t context_once=PTHREAD_ONCE_INIT;
static int context_bootstrap_result=3, bootstrap_calls=0;
static _Thread_local int context_setup_result=3;
static _Thread_local void* active;
static void* cuda_context;
static int current(void** out) { *out=active;return 0; }
static int set(void* context) { active=context;return 0; }
static void* find_zluda_symbol(const char* key) {
    return !strcmp(key,"cuCtxGetCurrent") ? (void*)current : (void*)set;
}
static void create_default_zluda_context(void) {
    ++bootstrap_calls;cuda_context=(void*)(uintptr_t)0x456;context_bootstrap_result=0;
}
''' + ensure + r'''
static void* worker(void* ignored) {
    (void)ignored;ensure_context();assert(active==cuda_context && context_setup_result==0);return NULL;
}
int main(void) {
    active=(void*)(uintptr_t)0x123;ensure_context();assert(active==(void*)(uintptr_t)0x123 && bootstrap_calls==0);
    pthread_t a;assert(!pthread_create(&a,NULL,worker,NULL));pthread_join(a,NULL);
    assert(bootstrap_calls==1);ensure_context();assert(active==(void*)(uintptr_t)0x123);
    assert(!pthread_create(&a,NULL,worker,NULL));pthread_join(a,NULL);assert(bootstrap_calls==1);
}
''')
