// MinGW-side support for d4r_ngx_param_msvc.cpp: allocation hooks for the
// parameter objects and the MSVC floating-point marker symbol that clang-cl
// objects reference.
#include <cstdlib>

extern "C" {
int _fltused = 0;
}

extern "C" void* d4r_host_alloc(unsigned long long size)
{
    return std::calloc(1, static_cast<size_t>(size));
}

extern "C" void d4r_host_free(void* pointer)
{
    std::free(pointer);
}
