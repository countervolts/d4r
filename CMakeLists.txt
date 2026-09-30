cmake_minimum_required(VERSION 3.24)
project(d4r_windows_diagnostics LANGUAGES C CXX)
if(NOT WIN32 OR NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
  message(FATAL_ERROR "These diagnostics require native Windows x64")
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(D4R_HIP_ROOT "$ENV{HIP_PATH}" CACHE PATH "One stable HIP SDK or TheRock installation")
if(NOT D4R_HIP_ROOT AND EXISTS "C:/Program Files/AMD/ROCm/7.2")
  set(D4R_HIP_ROOT "C:/Program Files/AMD/ROCm/7.2")
endif()
if(NOT EXISTS "${D4R_HIP_ROOT}/include/hip/hip_runtime_api.h")
  message(FATAL_ERROR "Set D4R_HIP_ROOT to a Windows HIP SDK (headers and bin/clang++.exe required)")
endif()
find_program(D4R_HIP_CLANG NAMES clang++.exe PATHS "${D4R_HIP_ROOT}/bin" "${D4R_HIP_ROOT}/lib/llvm/bin" NO_DEFAULT_PATH REQUIRED)
set(probe_object "${CMAKE_CURRENT_BINARY_DIR}/probe_gfx1201.hsaco")
add_custom_command(OUTPUT "${probe_object}"
  COMMAND "${D4R_HIP_CLANG}" -x hip --offload-arch=gfx1201
    --offload-device-only --no-gpu-bundle-output -mno-wavefrontsize64
    -nogpuinc -nogpulib -O2 "${CMAKE_CURRENT_SOURCE_DIR}/tools/windows/probe_kernel.hip"
    -o "${probe_object}"
  DEPENDS tools/windows/probe_kernel.hip VERBATIM)
add_custom_target(d4r_probe_code_object ALL DEPENDS "${probe_object}")
set(wmma_object "${CMAKE_CURRENT_BINARY_DIR}/wmma_gfx1201.hsaco")
add_custom_command(OUTPUT "${wmma_object}"
  COMMAND "${D4R_HIP_CLANG}" -x hip --offload-arch=gfx1201
    --offload-device-only --no-gpu-bundle-output -mno-wavefrontsize64
    -nogpuinc -nogpulib -O2 "${CMAKE_CURRENT_SOURCE_DIR}/tools/windows/wmma_gfx1201.hip"
    -o "${wmma_object}"
  DEPENDS tools/windows/wmma_gfx1201.hip kernels/common/wmma_backend.h
    kernels/common/wmma_layout.h VERBATIM)
add_custom_target(d4r_wmma_code_object ALL DEPENDS "${wmma_object}")
# K/M objects stay in experimental/, outside the ZLUDA native override
# directory, until each layer and then the complete network pass replay.
# M uses the existing FP16 widening baseline; native FP8 is not enabled here.
set(native_objects)
option(D4R_K_FP16_BASELINE "Round every K16 accumulation step like NVIDIA f16 WMMA" ON)
option(D4R_M_FP16_BASELINE "Preserve M e4m3 quantisation and f16 accumulation per K32" ON)
foreach(family k m)
  file(GLOB native_sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/${family}/*.hip")
  file(GLOB native_headers CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/${family}/*.h")
  foreach(source IN LISTS native_sources)
    get_filename_component(kernel_name "${source}" NAME_WE)
    set(native_object "${CMAKE_CURRENT_BINARY_DIR}/${kernel_name}_gfx1201.hsaco")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${source}")
    file(READ "${source}" kernel_source)
    string(REGEX MATCH "// d4r-build-flags: ([^\r\n]+)" flags_line "${kernel_source}")
    set(native_flags)
    if(family STREQUAL "k" AND D4R_K_FP16_BASELINE)
      list(APPEND native_flags -DD4R_K_FP16_BASELINE)
    endif()
    if(family STREQUAL "m" AND D4R_M_FP16_BASELINE)
      list(APPEND native_flags -DSWIN_EXACT -DSWIN_EXACT_PV)
    endif()
    if(flags_line)
      separate_arguments(source_flags NATIVE_COMMAND "${CMAKE_MATCH_1}")
      list(APPEND native_flags ${source_flags})
    endif()
    add_custom_command(OUTPUT "${native_object}"
      COMMAND "${D4R_HIP_CLANG}" -x hip -std=c++17 --offload-arch=gfx1201
        --offload-device-only --no-gpu-bundle-output -mno-wavefrontsize64
        -nogpuinc -nogpulib -O3 -DD4R_DEVICE_ONLY_MINIMAL ${native_flags}
        "${source}" -o "${native_object}"
      DEPENDS "${source}" ${native_headers} kernels/common/wmma_layout.h
        kernels/common/hip_device_minimal.h VERBATIM)
    list(APPEND native_objects "${native_object}")
    install(FILES "${native_object}" DESTINATION "experimental/${family}")
  endforeach()
endforeach()
add_custom_target(d4r_native_gfx1201_compile ALL DEPENDS ${native_objects})
foreach(probe hip_gfx1201_probe gfx12_wmma_probe k_module_probe native_replay_probe cuda_driver_probe cuda_mma_probe cuda_image_probe d3d12_command_probe d3d12_hip_interop_probe hip_stream_lifecycle_probe ngx_cuda_init_probe d3d12_dlss_probe)
  add_executable(d4r_${probe} tools/windows/${probe}.cpp)
  target_compile_definitions(d4r_${probe} PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX __HIP_PLATFORM_AMD__)
  target_include_directories(d4r_${probe} PRIVATE "${D4R_HIP_ROOT}/include")
  target_link_libraries(d4r_${probe} PRIVATE version dbghelp)
  if(MINGW)
    target_link_options(d4r_${probe} PRIVATE -static)
  endif()
  if(MSVC)
    target_compile_options(d4r_${probe} PRIVATE /W4 /EHsc)
    set_property(TARGET d4r_${probe} PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  else()
    target_compile_options(d4r_${probe} PRIVATE -Wall -Wextra)
  endif()
  install(TARGETS d4r_${probe} RUNTIME DESTINATION bin)
endforeach()
set(D4R_MINHOOK_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/external/MinHook" CACHE PATH "MinHook 1.3.4 source checkout")
if(NOT EXISTS "${D4R_MINHOOK_ROOT}/include/MinHook.h")
  message(FATAL_ERROR "Clone MinHook v1.3.4 into external/MinHook or set D4R_MINHOOK_ROOT")
endif()
add_subdirectory("${D4R_MINHOOK_ROOT}" "${CMAKE_CURRENT_BINARY_DIR}/minhook" EXCLUDE_FROM_ALL)
add_library(d4r_nvngx_windows SHARED tools/windows/d4r_nvngx_windows.cpp tools/windows/d3d12_command_hooks.cpp)
set_target_properties(d4r_nvngx_windows PROPERTIES PREFIX "" OUTPUT_NAME "d4r_nvngx")
target_compile_definitions(d4r_nvngx_windows PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX __HIP_PLATFORM_AMD__)
target_include_directories(d4r_nvngx_windows PRIVATE "${D4R_HIP_ROOT}/include")
target_link_libraries(d4r_nvngx_windows PRIVATE version dbghelp d3d12 dxguid minhook)
install(FILES "${D4R_MINHOOK_ROOT}/LICENSE.txt" DESTINATION licenses RENAME MinHook.txt)
if(MINGW)
  target_link_options(d4r_nvngx_windows PRIVATE -static)
endif()
install(TARGETS d4r_nvngx_windows RUNTIME DESTINATION bin)
if(MSVC)
  target_sources(d4r_ngx_cuda_init_probe PRIVATE tools/d4r_ngx_param_msvc.cpp tools/d4r_ngx_param_host.cpp)
  target_sources(d4r_d3d12_dlss_probe PRIVATE tools/d4r_ngx_param_msvc.cpp tools/d4r_ngx_param_host.cpp)
  target_sources(d4r_nvngx_windows PRIVATE tools/d4r_ngx_param_msvc.cpp tools/d4r_ngx_param_host.cpp)
else()
  find_program(D4R_CLANG_CL NAMES clang-cl.exe PATHS "${D4R_HIP_ROOT}/bin" "${D4R_HIP_ROOT}/lib/llvm/bin" NO_DEFAULT_PATH REQUIRED)
  set(ngx_parameter_object "${CMAKE_CURRENT_BINARY_DIR}/ngx_parameter_msvc.obj")
  add_custom_command(OUTPUT "${ngx_parameter_object}"
    COMMAND "${D4R_CLANG_CL}" /c /O2 /Zl /GS- /GR- /EHs-c-
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/d4r_ngx_param_msvc.cpp" "/Fo${ngx_parameter_object}"
    DEPENDS tools/d4r_ngx_param_msvc.cpp VERBATIM)
  set_source_files_properties("${ngx_parameter_object}" PROPERTIES GENERATED TRUE EXTERNAL_OBJECT TRUE)
  target_sources(d4r_ngx_cuda_init_probe PRIVATE "${ngx_parameter_object}" tools/d4r_ngx_param_host.cpp)
  target_sources(d4r_d3d12_dlss_probe PRIVATE "${ngx_parameter_object}" tools/d4r_ngx_param_host.cpp)
  target_sources(d4r_nvngx_windows PRIVATE "${ngx_parameter_object}" tools/d4r_ngx_param_host.cpp)
endif()
add_library(d4r_nvapi_trace SHARED tools/windows/nvapi_trace.cpp)
set_target_properties(d4r_nvapi_trace PROPERTIES PREFIX "" OUTPUT_NAME "nvapi64"
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/nvapi-trace"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/nvapi-trace")
target_compile_definitions(d4r_nvapi_trace PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
target_link_libraries(d4r_nvapi_trace PRIVATE version dbghelp)
if(MINGW)
  target_link_options(d4r_nvapi_trace PRIVATE -static)
endif()
install(TARGETS d4r_nvapi_trace RUNTIME DESTINATION nvapi-trace)
add_library(d4r_nvapi_compat SHARED tools/windows/nvapi_trace.cpp)
set_target_properties(d4r_nvapi_compat PROPERTIES PREFIX "" OUTPUT_NAME "nvapi64"
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/nvapi-compat"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/nvapi-compat")
target_compile_definitions(d4r_nvapi_compat PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX __HIP_PLATFORM_AMD__ D4R_NVAPI_COMPAT)
target_include_directories(d4r_nvapi_compat PRIVATE "${D4R_HIP_ROOT}/include")
target_link_libraries(d4r_nvapi_compat PRIVATE version dbghelp)
if(MINGW)
  target_link_options(d4r_nvapi_compat PRIVATE -static)
endif()
install(TARGETS d4r_nvapi_compat RUNTIME DESTINATION nvapi-compat)
target_link_libraries(d4r_d3d12_hip_interop_probe PRIVATE d3d12 dxgi dxguid)
target_link_libraries(d4r_d3d12_dlss_probe PRIVATE d3d12 dxgi dxguid)
target_link_libraries(d4r_d3d12_command_probe PRIVATE d3d12 dxgi dxguid)
add_dependencies(d4r_d3d12_command_probe d4r_nvngx_windows)
add_dependencies(d4r_hip_gfx1201_probe d4r_probe_code_object)
add_dependencies(d4r_gfx12_wmma_probe d4r_wmma_code_object)
add_dependencies(d4r_k_module_probe d4r_native_gfx1201_compile)
install(FILES "${probe_object}" DESTINATION bin)
install(FILES "${wmma_object}" DESTINATION bin)
install(FILES scripts/windows/test-windows-rdna4.ps1 DESTINATION .)
install(FILES scripts/windows/test-k-replay.ps1 DESTINATION .)
install(FILES scripts/windows/test-k-captures.ps1 DESTINATION .)
install(FILES scripts/windows/test-native-overrides.ps1 DESTINATION .)
install(FILES tools/windows/k_layer_validate.py DESTINATION .)
install(FILES tools/windows/k_replay_validate.py DESTINATION .)
install(FILES tools/windows/k_ptx_validate.py DESTINATION .)
install(FILES tools/windows/frame_compare.py DESTINATION .)
install(FILES tools/windows/m_replay_validate.py DESTINATION .)
install(FILES tools/windows/m_ptx_validate.py DESTINATION .)
install(FILES tools/windows/m_temporal_validate.py tools/windows/m_capture_compare.py DESTINATION .)
install(FILES scripts/windows/test-m-replay.ps1 DESTINATION .)
install(FILES scripts/windows/test-m-captures.ps1 DESTINATION .)
install(FILES kernels/tools/pwin_model.py kernels/tools/pwin_check.py kernels/tools/ptxsim.py
  kernels/tools/extract_dlss_ptx.py kernels/tools/kernel_manifest.py DESTINATION reference)
install(FILES kernels/tools/model_enc3.py kernels/tools/swin_model.py kernels/tools/swin_check.py DESTINATION reference)
install(FILES docs/windows-rdna4-port.md DESTINATION docs)
include(CTest)
add_test(NAME d3d12_command_backend COMMAND d4r_d3d12_command_probe
  --hip-root "${D4R_HIP_ROOT}" --module "$<TARGET_FILE:d4r_nvngx_windows>" --iterations 32)
set_tests_properties(d3d12_command_backend PROPERTIES TIMEOUT 120 LABELS "hardware;d3d12;command_lists;lifetime")
add_test(NAME hip_gfx1201 COMMAND d4r_hip_gfx1201_probe
  --hip-root "${D4R_HIP_ROOT}" --module "${probe_object}")
set_tests_properties(hip_gfx1201 PROPERTIES TIMEOUT 90 LABELS "hardware;gfx1201")
add_test(NAME gfx12_wmma COMMAND d4r_gfx12_wmma_probe
  --hip-root "${D4R_HIP_ROOT}" --module "${wmma_object}" --iterations 16)
set_tests_properties(gfx12_wmma PROPERTIES TIMEOUT 90 LABELS "hardware;gfx1201;wmma")
foreach(layer enc1 enc2)
  add_test(NAME k_${layer}_execution COMMAND d4r_k_module_probe
    --hip-root "${D4R_HIP_ROOT}"
    --module "${CMAKE_CURRENT_BINARY_DIR}/dltss_pwin_${layer}_layer_gfx1201.hsaco"
    --kernel-name ${layer})
  set_tests_properties(k_${layer}_execution PROPERTIES TIMEOUT 90 LABELS "hardware;gfx1201;k_execution")
endforeach()
add_test(NAME d3d12_hip_interop COMMAND d4r_d3d12_hip_interop_probe
  --hip-root "${D4R_HIP_ROOT}" --module "${probe_object}" --iterations 32)
set_tests_properties(d3d12_hip_interop PROPERTIES TIMEOUT 120 LABELS "hardware;interop")
add_test(NAME d3d12_hip_map_lifetime COMMAND d4r_d3d12_hip_interop_probe
  --hip-root "${D4R_HIP_ROOT}" --interop-mode map --iterations 64)
set_tests_properties(d3d12_hip_map_lifetime PROPERTIES TIMEOUT 120 LABELS "hardware;interop;lifetime")
set(D4R_ZLUDA_ROOT "" CACHE PATH "Local Windows ZLUDA directory (nvcuda.dll)")
if(EXISTS "${D4R_ZLUDA_ROOT}/nvcuda.dll")
  foreach(context primary created)
    add_test(NAME cuda_${context} COMMAND d4r_cuda_driver_probe
      --hip-root "${D4R_HIP_ROOT}" --cuda-dll "${D4R_ZLUDA_ROOT}/nvcuda.dll"
      --context ${context} --iterations 32)
    set_tests_properties(cuda_${context} PROPERTIES TIMEOUT 180 LABELS "hardware;zluda")
  endforeach()
  if(EXISTS "${D4R_ZLUDA_ROOT}/build-info.json")
    add_test(NAME cuda_f16_mma_reference COMMAND d4r_cuda_mma_probe
      --hip-root "${D4R_HIP_ROOT}" --cuda-dll "${D4R_ZLUDA_ROOT}/nvcuda.dll")
    set_tests_properties(cuda_f16_mma_reference PROPERTIES TIMEOUT 120
      LABELS "hardware;zluda;mma;reference"
      ENVIRONMENT "D4R_ZLUDA_WMMA=1;D4R_ZLUDA_WMMA_F16_REFERENCE=1")
    add_test(NAME cuda_images COMMAND d4r_cuda_image_probe
      --hip-root "${D4R_HIP_ROOT}" --cuda-dll "${D4R_ZLUDA_ROOT}/nvcuda.dll" --iterations 32)
    set_tests_properties(cuda_images PROPERTIES TIMEOUT 90 LABELS "hardware;zluda;descriptors")
    add_test(NAME d3d12_cuda_images COMMAND d4r_d3d12_dlss_probe
      --hip-root "${D4R_HIP_ROOT}" --cuda-dll "${D4R_ZLUDA_ROOT}/nvcuda.dll" --interop-mode images --iterations 1)
    set_tests_properties(d3d12_cuda_images PROPERTIES TIMEOUT 90 LABELS "hardware;interop;zluda;descriptors")
  endif()
endif()
