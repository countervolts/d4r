cmake_minimum_required(VERSION 3.24)
project(d4r_windows_diagnostics LANGUAGES CXX)
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
  DEPENDS tools/windows/wmma_gfx1201.hip kernels/common/wmma_backend.h VERBATIM)
add_custom_target(d4r_wmma_code_object ALL DEPENDS "${wmma_object}")
# These K objects stay in experimental/k, outside the ZLUDA native override
# directory, until each layer and then the complete network pass replay.
set(k_objects)
foreach(layer enc1 enc2)
  set(k_object "${CMAKE_CURRENT_BINARY_DIR}/dltss_pwin_${layer}_layer_gfx1201.hsaco")
  add_custom_command(OUTPUT "${k_object}"
    COMMAND "${D4R_HIP_CLANG}" -x hip -std=c++17 --offload-arch=gfx1201
      --offload-device-only --no-gpu-bundle-output -mno-wavefrontsize64
      -nogpuinc -nogpulib -O2 -DD4R_DEVICE_ONLY_MINIMAL
      "${CMAKE_CURRENT_SOURCE_DIR}/kernels/k/dltss_pwin_${layer}_layer.hip"
      -o "${k_object}"
    DEPENDS kernels/k/dltss_pwin_${layer}_layer.hip kernels/k/pwin_layer.h
      kernels/k/pwin_common.h kernels/common/wmma_backend.h
      kernels/common/hip_device_minimal.h VERBATIM)
  list(APPEND k_objects "${k_object}")
  install(FILES "${k_object}" DESTINATION experimental/k)
endforeach()
add_custom_target(d4r_k_gfx1201_compile ALL DEPENDS ${k_objects})
foreach(probe hip_gfx1201_probe gfx12_wmma_probe k_module_probe cuda_driver_probe d3d12_hip_interop_probe hip_stream_lifecycle_probe ngx_cuda_init_probe)
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
if(MSVC)
  target_sources(d4r_ngx_cuda_init_probe PRIVATE tools/d4r_ngx_param_msvc.cpp tools/d4r_ngx_param_host.cpp)
else()
  find_program(D4R_CLANG_CL NAMES clang-cl.exe PATHS "${D4R_HIP_ROOT}/bin" "${D4R_HIP_ROOT}/lib/llvm/bin" NO_DEFAULT_PATH REQUIRED)
  set(ngx_parameter_object "${CMAKE_CURRENT_BINARY_DIR}/ngx_parameter_msvc.obj")
  add_custom_command(OUTPUT "${ngx_parameter_object}"
    COMMAND "${D4R_CLANG_CL}" /c /O2 /Zl /GS- /GR- /EHs-c-
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/d4r_ngx_param_msvc.cpp" "/Fo${ngx_parameter_object}"
    DEPENDS tools/d4r_ngx_param_msvc.cpp VERBATIM)
  set_source_files_properties("${ngx_parameter_object}" PROPERTIES GENERATED TRUE EXTERNAL_OBJECT TRUE)
  target_sources(d4r_ngx_cuda_init_probe PRIVATE "${ngx_parameter_object}" tools/d4r_ngx_param_host.cpp)
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
add_dependencies(d4r_hip_gfx1201_probe d4r_probe_code_object)
add_dependencies(d4r_gfx12_wmma_probe d4r_wmma_code_object)
add_dependencies(d4r_k_module_probe d4r_k_gfx1201_compile)
install(FILES "${probe_object}" DESTINATION bin)
install(FILES "${wmma_object}" DESTINATION bin)
install(FILES scripts/windows/test-windows-rdna4.ps1 DESTINATION .)
install(FILES tools/windows/k_layer_validate.py DESTINATION .)
install(FILES kernels/tools/pwin_model.py DESTINATION reference)
install(FILES docs/windows-rdna4-port.md DESTINATION docs)
include(CTest)
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
endif()
