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
# Architecture parameterization adapted from realdody/d4r ae530b0.
# Keep explicit subtargets: no gfx11 substitution or untested future gfx12 ISA.
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/tools/windows/gpu-targets.json" d4r_target_json)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS tools/windows/gpu-targets.json)
string(JSON d4r_target_count LENGTH "${d4r_target_json}" targets)
math(EXPR d4r_target_last "${d4r_target_count} - 1")
set(d4r_supported_targets)
set(d4r_target_cpp_entries "")
foreach(index RANGE ${d4r_target_last})
  foreach(field architecture elfMachine isaMajor family)
    string(JSON target_${field} GET "${d4r_target_json}" targets ${index} ${field})
  endforeach()
  list(APPEND d4r_supported_targets "${target_architecture}")
  string(APPEND d4r_target_cpp_entries "  GpuTarget{\"${target_architecture}\", ${target_elfMachine}, ${target_isaMajor}, \"${target_family}\"},\n")
endforeach()
configure_file(tools/windows/gpu_targets.generated.h.in gpu_targets.generated.h @ONLY)
include_directories("${CMAKE_CURRENT_BINARY_DIR}")
set(D4R_GPU_ARCH "gfx1201" CACHE STRING "Native Windows package GPU target")
set_property(CACHE D4R_GPU_ARCH PROPERTY STRINGS ${d4r_supported_targets})
if(NOT D4R_GPU_ARCH IN_LIST d4r_supported_targets)
  message(FATAL_ERROR "D4R_GPU_ARCH must be one of: ${d4r_supported_targets}")
endif()
add_compile_definitions(D4R_TARGET_ARCH="${D4R_GPU_ARCH}")
configure_file(tools/windows/gpu-target.json.in gpu-target.json @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/gpu-target.json" DESTINATION .)
install(FILES tools/windows/gpu-targets.json DESTINATION .)
set(d4r_device_flags)
if(D4R_GPU_ARCH MATCHES "^gfx11")
  # Windows-only workaround informed by realdody/d4r 1897b4a. Keep the real
  # gfx11 layout; do not route RDNA3 through gfx12 fragments.
  list(APPEND d4r_device_flags -DD4R_WINDOWS_GFX11_LANE_EXCHANGE)
endif()
set(probe_object "${CMAKE_CURRENT_BINARY_DIR}/probe_${D4R_GPU_ARCH}.hsaco")
add_custom_command(OUTPUT "${probe_object}"
  COMMAND "${D4R_HIP_CLANG}" -x hip --offload-arch=${D4R_GPU_ARCH}
    --offload-device-only --no-gpu-bundle-output -mno-wavefrontsize64
    -nogpuinc -nogpulib -O2 "${CMAKE_CURRENT_SOURCE_DIR}/tools/windows/probe_kernel.hip"
    -o "${probe_object}"
  DEPENDS tools/windows/probe_kernel.hip VERBATIM)
add_custom_target(d4r_probe_code_object ALL DEPENDS "${probe_object}")
set(pixel_object "${CMAKE_CURRENT_BINARY_DIR}/pixel_convert_${D4R_GPU_ARCH}.hsaco")
add_custom_command(OUTPUT "${pixel_object}"
  COMMAND "${D4R_HIP_CLANG}" -x hip -std=c++17 --offload-arch=${D4R_GPU_ARCH}
    --offload-device-only --no-gpu-bundle-output -mno-wavefrontsize64
    -nogpuinc -nogpulib -O2 "${CMAKE_CURRENT_SOURCE_DIR}/tools/windows/pixel_convert_gfx1201.hip"
    -o "${pixel_object}"
  DEPENDS tools/windows/pixel_convert_gfx1201.hip tools/windows/pixel_format.h kernels/common/hip_device_minimal.h VERBATIM)
add_custom_target(d4r_pixel_code_object ALL DEPENDS "${pixel_object}")
set(wmma_object "${CMAKE_CURRENT_BINARY_DIR}/wmma_${D4R_GPU_ARCH}.hsaco")
add_custom_command(OUTPUT "${wmma_object}"
  COMMAND "${D4R_HIP_CLANG}" -x hip --offload-arch=${D4R_GPU_ARCH}
    --offload-device-only --no-gpu-bundle-output -mno-wavefrontsize64
    -nogpuinc -nogpulib -O2 ${d4r_device_flags} "${CMAKE_CURRENT_SOURCE_DIR}/tools/windows/wmma_gfx1201.hip"
    -o "${wmma_object}"
  DEPENDS tools/windows/wmma_gfx1201.hip kernels/common/wmma_backend.h
    kernels/common/wmma_layout.h kernels/common/wave32_exchange.h VERBATIM)
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
    set(native_object "${CMAKE_CURRENT_BINARY_DIR}/${kernel_name}_${D4R_GPU_ARCH}.hsaco")
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
      COMMAND "${D4R_HIP_CLANG}" -x hip -std=c++17 --offload-arch=${D4R_GPU_ARCH}
        --offload-device-only --no-gpu-bundle-output -mno-wavefrontsize64
        -nogpuinc -nogpulib -O3 -DD4R_DEVICE_ONLY_MINIMAL ${d4r_device_flags} ${native_flags}
        "${source}" -o "${native_object}"
      DEPENDS "${source}" ${native_headers} kernels/common/wmma_layout.h kernels/common/wave32_exchange.h
        kernels/common/hip_device_minimal.h VERBATIM)
    list(APPEND native_objects "${native_object}")
    install(FILES "${native_object}" DESTINATION "experimental/${family}")
  endforeach()
endforeach()
add_custom_target(d4r_native_compile ALL DEPENDS ${native_objects})
# Retain the old manual build target for existing scripts.
add_custom_target(d4r_native_gfx1201_compile DEPENDS d4r_native_compile)
foreach(probe gpu_inventory hip_gfx1201_probe gfx12_wmma_probe k_module_probe native_replay_probe cuda_driver_probe cuda_mma_probe cuda_image_probe d3d12_command_probe d3d12_pixel_probe d3d12_hip_interop_probe hip_stream_lifecycle_probe ngx_cuda_init_probe d3d12_dlss_probe)
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
foreach(probe binary16_probe gpu_target_probe)
  add_executable(d4r_${probe} tools/windows/${probe}.cpp)
  if(MINGW)
    target_link_options(d4r_${probe} PRIVATE -static)
  elseif(MSVC)
    set_property(TARGET d4r_${probe} PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  endif()
  install(TARGETS d4r_${probe} RUNTIME DESTINATION bin)
endforeach()
add_dependencies(d4r_gpu_target_probe d4r_probe_code_object d4r_pixel_code_object)
set(D4R_MINHOOK_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/external/MinHook" CACHE PATH "MinHook 1.3.4 source checkout")
if(NOT EXISTS "${D4R_MINHOOK_ROOT}/include/MinHook.h")
  message(FATAL_ERROR "Clone MinHook v1.3.4 into external/MinHook or set D4R_MINHOOK_ROOT")
endif()
add_subdirectory("${D4R_MINHOOK_ROOT}" "${CMAKE_CURRENT_BINARY_DIR}/minhook" EXCLUDE_FROM_ALL)
add_library(d4r_nvngx_windows SHARED tools/windows/d4r_nvngx_windows.cpp tools/windows/d3d12_command_hooks.cpp)
add_dependencies(d4r_nvngx_windows d4r_pixel_code_object)
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
target_link_libraries(d4r_d3d12_dlss_probe PRIVATE d3d12 dxgi dxguid d3dcompiler)
add_executable(d4r_debug_launcher tools/windows/debug_launcher.cpp)
target_include_directories(d4r_debug_launcher PRIVATE tools/windows)
target_link_libraries(d4r_debug_launcher PRIVATE version dbghelp)
install(TARGETS d4r_debug_launcher RUNTIME DESTINATION bin)
target_link_libraries(d4r_d3d12_command_probe PRIVATE d3d12 dxgi dxguid)
target_link_libraries(d4r_d3d12_pixel_probe PRIVATE d3d12 dxgi dxguid)
add_dependencies(d4r_d3d12_pixel_probe d4r_pixel_code_object)
add_dependencies(d4r_d3d12_command_probe d4r_nvngx_windows)
set(D4R_AGILITY_ROOT "" CACHE PATH "Optional unpacked Microsoft.Direct3D.D3D12 1.619.5 debug SDK")
if(EXISTS "${D4R_AGILITY_ROOT}/build/native/bin/x64/d3d12SDKLayers.dll")
  add_executable(d4r_d3d12_command_debug_probe EXCLUDE_FROM_ALL tools/windows/d3d12_command_probe.cpp)
  target_compile_definitions(d4r_d3d12_command_debug_probe PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX __HIP_PLATFORM_AMD__ D4R_DIAGNOSTIC_AGILITY)
  target_include_directories(d4r_d3d12_command_debug_probe PRIVATE "${D4R_HIP_ROOT}/include")
  target_link_libraries(d4r_d3d12_command_debug_probe PRIVATE version dbghelp d3d12 dxgi dxguid)
  if(MINGW)
    target_link_options(d4r_d3d12_command_debug_probe PRIVATE -static)
  endif()
  foreach(dll D3D12Core.dll d3d12SDKLayers.dll)
    configure_file("${D4R_AGILITY_ROOT}/build/native/bin/x64/${dll}" "${CMAKE_CURRENT_BINARY_DIR}/debug-d3d12/${dll}" COPYONLY)
  endforeach()
  configure_file("${D4R_AGILITY_ROOT}/LICENSE.txt" "${CMAKE_CURRENT_BINARY_DIR}/debug-d3d12/LICENSE.txt" COPYONLY)
endif()
add_dependencies(d4r_hip_gfx1201_probe d4r_probe_code_object)
add_dependencies(d4r_gfx12_wmma_probe d4r_wmma_code_object)
add_dependencies(d4r_k_module_probe d4r_native_compile)
install(FILES "${probe_object}" DESTINATION bin)
install(FILES "${pixel_object}" DESTINATION bin)
install(FILES "${wmma_object}" DESTINATION bin)
install(FILES scripts/windows/test-windows-rdna4.ps1 DESTINATION .)
install(FILES scripts/windows/gpu-target.ps1 DESTINATION .)
install(FILES scripts/windows/gpu-preflight.ps1 DESTINATION .)
install(FILES scripts/windows/test-gpu-target-contract.ps1 DESTINATION .)
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
install(FILES docs/windows-gpu-support.md DESTINATION docs)
include(CTest)
add_test(NAME binary16_rounding COMMAND d4r_binary16_probe)
set_tests_properties(binary16_rounding PROPERTIES TIMEOUT 30 LABELS "software;numerics")
add_test(NAME gpu_target_contract COMMAND d4r_gpu_target_probe "${probe_object}" "${pixel_object}")
set_tests_properties(gpu_target_contract PROPERTIES TIMEOUT 30 LABELS "software;architecture")
find_program(D4R_POWERSHELL NAMES pwsh.exe powershell.exe REQUIRED)
add_test(NAME gpu_target_script_contract COMMAND "${D4R_POWERSHELL}" -NoProfile -File
  "${CMAKE_CURRENT_SOURCE_DIR}/scripts/windows/test-gpu-target-contract.ps1"
  -PackageRoot "${CMAKE_CURRENT_BINARY_DIR}" -CodeObject "${probe_object}")
set_tests_properties(gpu_target_script_contract PROPERTIES TIMEOUT 30 LABELS "software;architecture")
add_test(NAME d3d12_gpu_pixel_formats COMMAND d4r_d3d12_pixel_probe
  --hip-root "${D4R_HIP_ROOT}" --module "${pixel_object}" --iterations 2)
set_tests_properties(d3d12_gpu_pixel_formats PROPERTIES TIMEOUT 120 LABELS "hardware;${D4R_GPU_ARCH};formats")
add_test(NAME d3d12_gpu_enhanced_formats COMMAND d4r_d3d12_pixel_probe
  --hip-root "${D4R_HIP_ROOT}" --module "${pixel_object}" --interop-mode enhanced --iterations 2)
set_tests_properties(d3d12_gpu_enhanced_formats PROPERTIES TIMEOUT 120 LABELS "hardware;${D4R_GPU_ARCH};formats;enhanced_barriers")
add_test(NAME d3d12_command_backend COMMAND d4r_d3d12_command_probe
  --hip-root "${D4R_HIP_ROOT}" --module "$<TARGET_FILE:d4r_nvngx_windows>" --interop-mode indirect --iterations 32)
set_tests_properties(d3d12_command_backend PROPERTIES TIMEOUT 120 LABELS "hardware;d3d12;command_lists;lifetime")
add_test(NAME d3d12_command_initial_pso COMMAND d4r_d3d12_command_probe
  --hip-root "${D4R_HIP_ROOT}" --module "$<TARGET_FILE:d4r_nvngx_windows>" --interop-mode initial-pso --iterations 32)
set_tests_properties(d3d12_command_initial_pso PROPERTIES TIMEOUT 120 LABELS "hardware;d3d12;command_lists;lifetime")
add_test(NAME d3d12_command_root_reset_warp COMMAND d4r_d3d12_command_probe
  --hip-root "${D4R_HIP_ROOT}" --module "$<TARGET_FILE:d4r_nvngx_windows>" --interop-mode indirect-root-reset --iterations 32)
set_tests_properties(d3d12_command_root_reset_warp PROPERTIES TIMEOUT 120 LABELS "software;d3d12;command_lists;lifetime"
  ENVIRONMENT "D4R_COMMAND_PROBE_WARP=1;D4R_COMMAND_PROBE_NO_HIP=1")
add_test(NAME hip_gfx1201 COMMAND d4r_hip_gfx1201_probe
  --hip-root "${D4R_HIP_ROOT}" --module "${probe_object}")
set_tests_properties(hip_gfx1201 PROPERTIES TIMEOUT 90 LABELS "hardware;${D4R_GPU_ARCH}")
add_test(NAME gfx12_wmma COMMAND d4r_gfx12_wmma_probe
  --hip-root "${D4R_HIP_ROOT}" --module "${wmma_object}" --iterations 16)
set_tests_properties(gfx12_wmma PROPERTIES TIMEOUT 90 LABELS "hardware;${D4R_GPU_ARCH};wmma")
foreach(layer enc1 enc2)
  add_test(NAME k_${layer}_execution COMMAND d4r_k_module_probe
    --hip-root "${D4R_HIP_ROOT}"
    --module "${CMAKE_CURRENT_BINARY_DIR}/dltss_pwin_${layer}_layer_${D4R_GPU_ARCH}.hsaco"
    --kernel-name ${layer})
  set_tests_properties(k_${layer}_execution PROPERTIES TIMEOUT 90 LABELS "hardware;${D4R_GPU_ARCH};k_execution")
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
    add_test(NAME cuda_images_async COMMAND d4r_cuda_image_probe
      --hip-root "${D4R_HIP_ROOT}" --cuda-dll "${D4R_ZLUDA_ROOT}/nvcuda.dll" --iterations 32)
    set_tests_properties(cuda_images_async PROPERTIES TIMEOUT 90 LABELS "hardware;zluda;descriptors;async"
      ENVIRONMENT "D4R_DIAG_ASYNC_ARRAY_COPY=1")
    add_test(NAME d3d12_cuda_images COMMAND d4r_d3d12_dlss_probe
      --hip-root "${D4R_HIP_ROOT}" --cuda-dll "${D4R_ZLUDA_ROOT}/nvcuda.dll" --interop-mode images --iterations 1)
    set_tests_properties(d3d12_cuda_images PROPERTIES TIMEOUT 90 LABELS "hardware;interop;zluda;descriptors")
  endif()
endif()
