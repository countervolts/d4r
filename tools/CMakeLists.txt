# Native build of the diagnostic tools (replays, benchmarks, probes, D3D12/Vulkan harnesses) for Windows, where
# the scripts/build_*.sh recipes (g++ -ldl, MinGW cross compilers) do not apply. It also works on Linux.
#
# Windows, all tools (MSYS2 MinGW-w64, the toolchain the Linux scripts cross-compile with):
#   cmake -S tools -B build/tools -G Ninja -DCMAKE_CXX_COMPILER=C:/msys64/mingw64/bin/g++.exe ^
#         -DCMAKE_C_COMPILER=C:/msys64/mingw64/bin/gcc.exe
#   cmake --build build/tools
# Visual Studio also works but skips the tools written for MinGW's headers and _Float16 (the summary says which).
#
# Optional inputs (each enables the tools that need it):
#   D4R_ROCM_DIR        ROCm / AMD HIP SDK root (Windows: C:/ROCm/<version>; also HIP_PATH, ROCM_PATH)
#   NVSDK_NGX_INCLUDE   your copy of NVIDIA's nvsdk_ngx*.h (the ngx_*_probe tools)
#   D4R_CLANG_CL        clang-cl for the MSVC-ABI NGX parameter objects (default: the HIP SDK's, then PATH)
# The CUDA-driver tools load ZLUDA at run time: libcuda.so on Linux, nvcuda.dll on Windows. See
# docs/windows-tools.md.
cmake_minimum_required(VERSION 3.20)
project(d4r_tools LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")
# the probes check their results with assert(): keep it in every configuration
foreach(FLAGS IN ITEMS CMAKE_C_FLAGS_RELEASE CMAKE_CXX_FLAGS_RELEASE CMAKE_C_FLAGS_RELWITHDEBINFO
                       CMAKE_CXX_FLAGS_RELWITHDEBINFO CMAKE_C_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_MINSIZEREL)
    string(REGEX REPLACE "[-/]DNDEBUG" "" ${FLAGS} "${${FLAGS}}")
endforeach()
find_package(Vulkan)
find_package(Python3 COMPONENTS Interpreter)

if(MSVC)
    add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS _USE_MATH_DEFINES)
    add_compile_options(/W3 /utf-8 /permissive- /Zc:preprocessor /bigobj)
else()
    add_compile_options(-O2 -Wall -Wextra -Wno-missing-field-initializers)
endif()
if(MINGW)
    # self-contained executables, as scripts/build_*.sh links them (no MSYS2 DLLs needed at run time)
    add_link_options(-static -static-libgcc -static-libstdc++)
endif()

set(D4R_ROCM_DIR "$ENV{D4R_ROCM_DIR}" CACHE PATH "ROCm / HIP SDK root (optional)")
if(NOT D4R_ROCM_DIR)
    foreach(CANDIDATE "$ENV{HIP_PATH}" "$ENV{ROCM_PATH}" /opt/rocm)
        if(CANDIDATE AND EXISTS "${CANDIDATE}/include/hip/hip_runtime_api.h")
            set(D4R_ROCM_DIR "${CANDIDATE}")
            break()
        endif()
    endforeach()
endif()
if(NOT D4R_ROCM_DIR AND WIN32)
    file(GLOB D4R_ROCM_CANDIDATES "C:/ROCm/*" "C:/Program Files/AMD/ROCm/*")
    foreach(CANDIDATE IN LISTS D4R_ROCM_CANDIDATES)
        if(EXISTS "${CANDIDATE}/include/hip/hip_runtime_api.h")
            set(D4R_ROCM_DIR "${CANDIDATE}") # the newest, by name
        endif()
    endforeach()
endif()
if(D4R_ROCM_DIR)
    find_library(D4R_AMDHIP64 NAMES amdhip64 HINTS "${D4R_ROCM_DIR}/lib" NO_DEFAULT_PATH)
    find_program(D4R_HIPCC NAMES hipcc HINTS "${D4R_ROCM_DIR}/bin" NO_DEFAULT_PATH)
endif()
set(NVSDK_NGX_INCLUDE "$ENV{NVSDK_NGX_INCLUDE}" CACHE PATH "Directory with NVIDIA's nvsdk_ngx*.h (optional)")
find_program(D4R_CLANG_CL NAMES clang-cl HINTS "${D4R_ROCM_DIR}/lib/llvm/bin")

set(D4R_BUILT)
set(D4R_SKIPPED)
macro(d4r_tool NAME)
    cmake_parse_arguments(T "" "" "SOURCES;LIBS;INCLUDES;DEFINES" ${ARGN})
    add_executable(${NAME} ${T_SOURCES})
    target_include_directories(${NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" ${T_INCLUDES})
    target_link_libraries(${NAME} PRIVATE ${T_LIBS})
    target_compile_definitions(${NAME} PRIVATE ${T_DEFINES})
    if(NOT WIN32)
        target_link_libraries(${NAME} PRIVATE ${CMAKE_DL_LIBS})
    endif()
    list(APPEND D4R_BUILT ${NAME})
endmacro()
macro(d4r_skip NAME WHY)
    list(APPEND D4R_SKIPPED "${NAME} (${WHY})")
endmacro()

# --- CUDA driver API replays and benchmarks: run against ZLUDA (or MicroCUDA), loaded at run time ---
foreach(TOOL IN ITEMS replay_dlss_kernel replay_dlss_layer replay_dlss_surface_store replay_dlss_surface_half_store
                      replay_dlss_texture_sample replay_rr_launch bench_dlss_kernel bench_cuda_copies)
    d4r_tool(${TOOL} SOURCES ${TOOL}.cpp)
endforeach()
# kernels/tools/ptxload: loads PTX through the CUDA driver (ZLUDA) and resolves every entry
d4r_tool(ptxload SOURCES "${ROOT}/kernels/tools/ptxload.c")
if(MSVC AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    d4r_skip(replay_dlss_fp8_ops "uses _Float16: build with MinGW g++ or clang")
else()
    d4r_tool(replay_dlss_fp8_ops SOURCES replay_dlss_fp8_ops.cpp)
endif()

# --- HIP host API (the HIP SDK's headers and amdhip64) ---
if(D4R_AMDHIP64)
    set(HIP_INCLUDE "${D4R_ROCM_DIR}/include")
    d4r_tool(replay_dlss_histogram SOURCES replay_dlss_histogram.cpp LIBS "${D4R_AMDHIP64}"
             INCLUDES "${HIP_INCLUDE}" DEFINES __HIP_PLATFORM_AMD__=1)
    if(Vulkan_FOUND)
        d4r_tool(hip_vk_interop_probe SOURCES hip_vk_interop_probe.cpp LIBS "${D4R_AMDHIP64}" Vulkan::Vulkan
                 INCLUDES "${HIP_INCLUDE}" DEFINES __HIP_PLATFORM_AMD__=1)
    else()
        d4r_skip(hip_vk_interop_probe "no Vulkan SDK")
    endif()
    # MicroCUDA: a CUDA driver library over HIP (libd4r_microcuda.so / d4r_microcuda.dll, opt-in)
    add_library(d4r_microcuda SHARED microcuda/runtime.cpp)
    target_include_directories(d4r_microcuda PRIVATE "${HIP_INCLUDE}")
    target_link_libraries(d4r_microcuda PRIVATE "${D4R_AMDHIP64}")
    if(WIN32)
        set_target_properties(d4r_microcuda PROPERTIES PREFIX "" WINDOWS_EXPORT_ALL_SYMBOLS ON)
        target_link_libraries(d4r_microcuda PRIVATE bcrypt)
        if(MINGW)   # static libgcc's own exports would otherwise stop ld from exporting the cu* entry points
            target_link_options(d4r_microcuda PRIVATE -Wl,--export-all-symbols,--exclude-libs,ALL)
        endif()
    else()
        find_package(OpenSSL REQUIRED COMPONENTS Crypto)
        find_package(Threads REQUIRED)
        target_link_libraries(d4r_microcuda PRIVATE OpenSSL::Crypto Threads::Threads)
    endif()
    list(APPEND D4R_BUILT d4r_microcuda)
else()
    d4r_skip("replay_dlss_histogram, hip_vk_interop_probe, d4r_microcuda" "no HIP SDK; set D4R_ROCM_DIR")
endif()

# --- HIP with device code (or hipModuleLoad of code objects): compiled by hipcc ---
if(D4R_HIPCC)
    set(HIPCC_ENV)
    set(HIP_DEFINES)
    if(WIN32)
        # The HIP SDK's clang uses the newest Visual Studio's C++ headers, which can be newer than it accepts
        # (ROCm 7.14's clang rejects Visual Studio 18's <cmath>). Default to Visual Studio 2022's toolset.
        if(NOT DEFINED D4R_HIP_VCTOOLS)
            set(PROGRAM_FILES_X86 "ProgramFiles(x86)")
            execute_process(COMMAND "$ENV{${PROGRAM_FILES_X86}}/Microsoft Visual Studio/Installer/vswhere.exe"
                                    -version "[17.0,18.0)" -latest -property installationPath
                            OUTPUT_VARIABLE VS2022 OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
            if(VS2022 AND EXISTS "${VS2022}/VC/Auxiliary/Build/Microsoft.VCToolsVersion.default.txt")
                file(STRINGS "${VS2022}/VC/Auxiliary/Build/Microsoft.VCToolsVersion.default.txt" VCTOOLS_VERSION)
                set(VS2022_TOOLS "${VS2022}/VC/Tools/MSVC/${VCTOOLS_VERSION}")
            endif()
        endif()
        set(D4R_HIP_VCTOOLS "${VS2022_TOOLS}" CACHE PATH "MSVC toolset (VC/Tools/MSVC/<version>) for hipcc; empty: clang's choice")
        if(D4R_HIP_VCTOOLS)
            set(HIPCC_ENV "${CMAKE_COMMAND}" -E env "VCToolsInstallDir=${D4R_HIP_VCTOOLS}/")
        endif()
        set(HIP_DEFINES -D_CRT_SECURE_NO_WARNINGS) # the MSVC C runtime's getenv/fopen deprecations
    endif()
    # hipcc on Windows does not detect the GPU: without a target a kernel has no code object for it
    if(NOT DEFINED D4R_GPU_ARCH)
        find_program(AMDGPU_ARCH NAMES amdgpu-arch HINTS "${D4R_ROCM_DIR}/lib/llvm/bin" "${D4R_ROCM_DIR}/bin")
        if(AMDGPU_ARCH)
            execute_process(COMMAND "${AMDGPU_ARCH}" OUTPUT_VARIABLE DETECTED_ARCH OUTPUT_STRIP_TRAILING_WHITESPACE
                            ERROR_QUIET)
            string(REGEX REPLACE "[\r\n]+" ";" DETECTED_ARCH "${DETECTED_ARCH}")
            list(REMOVE_DUPLICATES DETECTED_ARCH)
            list(JOIN DETECTED_ARCH "," DETECTED_ARCH)
        endif()
    endif()
    set(D4R_GPU_ARCH "${DETECTED_ARCH}" CACHE STRING "GPU targets for hipcc, e.g. gfx1201 or gfx1101,gfx1201")
    set(HIP_ARCH)
    if(D4R_GPU_ARCH)
        set(HIP_ARCH "--offload-arch=${D4R_GPU_ARCH}")
    endif()
    # d4r_hipcc_tool(NAME SOURCE [hipcc arguments...]): an executable NAME built by hipcc from SOURCE
    macro(d4r_hipcc_tool NAME SOURCE)
        set(HIP_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${NAME}${CMAKE_EXECUTABLE_SUFFIX}")
        add_custom_command(OUTPUT "${HIP_OUTPUT}"
            COMMAND ${HIPCC_ENV} "${D4R_HIPCC}" ${HIP_ARCH} ${HIP_DEFINES} -O2 "-I${CMAKE_CURRENT_SOURCE_DIR}" "${SOURCE}" ${ARGN}
                    -o "${HIP_OUTPUT}"
            DEPENDS "${SOURCE}" VERBATIM)
        add_custom_target(${NAME} ALL DEPENDS "${HIP_OUTPUT}")
        list(APPEND D4R_BUILT ${NAME})
    endmacro()
    # kernels/tools: dump_runner runs a native kernel (.hsaco) on a replay dump (docs/native-kernels.md, the
    # opt-in GPU tests' D4R_DUMP_RUNNER); the FP8 test checks an unpaired FP8 WMMA code object
    d4r_hipcc_tool(dump_runner "${ROOT}/kernels/tools/dump_runner.cpp" -std=c++17)
    d4r_hipcc_tool(rrswin_fp8_unpaired_test "${ROOT}/kernels/tools/rrswin_fp8_unpaired_test.cpp" -std=c++17)
    if(Vulkan_FOUND)
        if(WIN32)
            get_filename_component(VULKAN_LIB_DIR "${Vulkan_LIBRARY}" DIRECTORY)
            set(VULKAN_LINK "-L${VULKAN_LIB_DIR}" -lvulkan-1)
        else()
            set(VULKAN_LINK -lvulkan)
        endif()
        d4r_hipcc_tool(hip_vk_texture_stale_repro "${CMAKE_CURRENT_SOURCE_DIR}/hip_vk_texture_stale_repro.cpp"
                       -std=c++17 "-I${Vulkan_INCLUDE_DIRS}" ${VULKAN_LINK})
    else()
        d4r_skip(hip_vk_texture_stale_repro "needs the Vulkan SDK")
    endif()
else()
    d4r_skip("dump_runner, rrswin_fp8_unpaired_test, hip_vk_texture_stale_repro" "needs hipcc (D4R_ROCM_DIR)")
endif()

# --- Vulkan probes (no CUDA/NGX); the staging shaders as scripts/build_vulkan_staging.sh builds them ---
find_program(GLSLANG_VALIDATOR NAMES glslangValidator HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin")
if(Vulkan_FOUND)
    d4r_tool(vulkan_depth_copy_probe SOURCES vulkan_depth_copy_probe.cpp LIBS Vulkan::Vulkan)
    # the headless kernel runners of vulkan/ and engine/ (kcase.py, engine/test_*.py, engine/d3d12/bench_*.py)
    d4r_tool(kvk SOURCES "${ROOT}/vulkan/kvk.cpp" LIBS Vulkan::Vulkan)
    d4r_tool(evk SOURCES "${ROOT}/engine/evk.cpp" LIBS Vulkan::Vulkan)
    if(GLSLANG_VALIDATOR AND Python3_FOUND)
        set(STAGING "${CMAKE_CURRENT_BINARY_DIR}/vulkan-staging")
        add_custom_command(OUTPUT "${STAGING}/d4r_vulkan_staging_spv.h"
            COMMAND "${CMAKE_COMMAND}" -E env "GLSLANG_VALIDATOR=${GLSLANG_VALIDATOR}"
                    "${Python3_EXECUTABLE}" "${ROOT}/scripts/build_vulkan_staging.py" "${STAGING}"
            DEPENDS "${ROOT}/scripts/build_vulkan_staging.py" "${ROOT}/kernels/support/vulkan_read.comp"
                    "${ROOT}/kernels/support/vulkan_wait.comp" "${ROOT}/kernels/support/vulkan_write.comp"
                    "${ROOT}/kernels/support/fsr_rr_prepare.comp" "${ROOT}/kernels/support/fsr_rr_compose.comp"
            VERBATIM)
        add_custom_target(d4r_vulkan_staging DEPENDS "${STAGING}/d4r_vulkan_staging_spv.h")
        foreach(TOOL IN ITEMS vulkan_staging_probe vulkan_output_conversion_probe)
            d4r_tool(${TOOL} SOURCES ${TOOL}.cpp LIBS Vulkan::Vulkan INCLUDES "${STAGING}")
            add_dependencies(${TOOL} d4r_vulkan_staging)
        endforeach()
    else()
        d4r_skip("vulkan_staging_probe, vulkan_output_conversion_probe" "needs glslangValidator and Python 3")
    endif()
else()
    d4r_skip("Vulkan probes" "no Vulkan SDK")
endif()

# --- Windows programs: harnesses and probes written for MinGW's headers (the same sources the Linux scripts
#     cross-compile). The NGX parameter objects NVIDIA's DLLs call into must use the MSVC C++ ABI, so they are
#     compiled by clang-cl and linked in, as scripts/build_d4r_nvngx_shim.sh does. ---
if(WIN32)
    d4r_tool(ngx_load_probe SOURCES ngx_load_probe.cpp)
    d4r_tool(d4r_asi_probe SOURCES d4r_asi_probe.cpp)
    if(MINGW)
        target_link_options(d4r_asi_probe PRIVATE -municode)
    endif()
endif()
if(MINGW AND NOT D4R_CLANG_CL)
    d4r_skip("d3d12_dlss_harness, vulkan_dlss_harness, rr_backend_check, ngx_cuda_probe"
             "needs clang-cl for the MSVC-ABI NGX objects; set D4R_CLANG_CL")
endif()
if(MINGW)
    function(d4r_msvc_object NAME SOURCE)
        set(OBJECT "${CMAKE_CURRENT_BINARY_DIR}/${NAME}.obj")
        add_custom_command(OUTPUT "${OBJECT}"
            COMMAND "${D4R_CLANG_CL}" --target=x86_64-pc-windows-msvc /nologo /std:c++20 /O2 /c /GS- /GR- /EHs-c- /Zl
                    "/Fo${OBJECT}" "${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE}"
            DEPENDS "${SOURCE}" VERBATIM)
        set(${NAME} "${OBJECT}" PARENT_SCOPE)
    endfunction()
    d4r_tool(fsr_rr_probe SOURCES fsr_rr_probe.cpp LIBS d3d12 dxgi INCLUDES "${Vulkan_INCLUDE_DIRS}")
    target_link_options(fsr_rr_probe PRIVATE -municode)
    d4r_tool(fsr_upscale_probe SOURCES fsr_upscale_probe.cpp LIBS d3d12 dxgi INCLUDES "${Vulkan_INCLUDE_DIRS}")
    target_link_options(fsr_upscale_probe PRIVATE -municode)
    d4r_tool(d3d12_hip_interop_probe SOURCES d3d12_hip_interop_probe.cpp LIBS d3d12 INCLUDES "${Vulkan_INCLUDE_DIRS}")
    if(D4R_CLANG_CL)
        d4r_msvc_object(NGX_PARAM d4r_ngx_param_msvc.cpp)
        d4r_tool(d3d12_dlss_harness SOURCES d3d12_dlss_harness.cpp d4r_ngx_param_host.cpp "${NGX_PARAM}"
                 INCLUDES "${Vulkan_INCLUDE_DIRS}" LIBS d3d12 dxgi)
        d4r_tool(vulkan_dlss_harness SOURCES vulkan_dlss_harness.cpp d4r_ngx_param_host.cpp "${NGX_PARAM}"
                 INCLUDES "${Vulkan_INCLUDE_DIRS}")
        d4r_tool(rr_backend_check SOURCES rr_backend_check.cpp d4r_ngx_param_host.cpp "${NGX_PARAM}"
                 LIBS d3d12 dxgi)
    endif()
    if(NVSDK_NGX_INCLUDE AND EXISTS "${NVSDK_NGX_INCLUDE}/nvsdk_ngx.h")
        d4r_tool(ngx_d3d12_probe SOURCES ngx_d3d12_probe.cpp INCLUDES "${NVSDK_NGX_INCLUDE}")
        d4r_tool(ngx_vulkan_probe SOURCES ngx_vulkan_probe.cpp INCLUDES "${NVSDK_NGX_INCLUDE}" "${Vulkan_INCLUDE_DIRS}")
        if(D4R_CLANG_CL)
            d4r_msvc_object(NGX_ABI ngx_param_abi_probe_msvc.cpp)
            d4r_tool(ngx_cuda_probe SOURCES ngx_cuda_probe.cpp "${NGX_ABI}" INCLUDES "${NVSDK_NGX_INCLUDE}")
        endif()
    else()
        d4r_skip("ngx_d3d12_probe, ngx_cuda_probe, ngx_vulkan_probe" "set NVSDK_NGX_INCLUDE")
    endif()
elseif(WIN32)
    d4r_skip("d3d12_dlss_harness, vulkan_dlss_harness, rr_backend_check, fsr_rr_probe, fsr_upscale_probe, d3d12_hip_interop_probe, ngx_*_probe"
             "written for MinGW's headers: configure with MSYS2 mingw64 g++")
endif()

list(JOIN D4R_BUILT ", " BUILT_TEXT)
message(STATUS "d4r tools: ${BUILT_TEXT}")
foreach(SKIPPED IN LISTS D4R_SKIPPED)
    message(STATUS "d4r tools: skipped ${SKIPPED}")
endforeach()
