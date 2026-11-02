cmake_minimum_required(VERSION 3.20)
project(d4r_d3d12_probe LANGUAGES CXX)
if(NOT WIN32)
    message(FATAL_ERROR "Build a Windows executable (native Windows or a Windows cross toolchain). Run that executable through Wine/vkd3d on Linux.")
endif()
add_executable(d4r-k-layer replay_layer.cpp)
target_compile_features(d4r-k-layer PRIVATE cxx_std_17)
target_compile_definitions(d4r-k-layer PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
target_link_libraries(d4r-k-layer PRIVATE d3d12 dxgi d3dcompiler)
if(MSVC)
    target_compile_options(d4r-k-layer PRIVATE /W4 /WX /permissive-)
else()
    target_compile_options(d4r-k-layer PRIVATE -Wall -Wextra -Werror)
endif()
# Experimental isolated M/L network replay; the parent engine build supplies complete-frame replay.
add_executable(d4r-m-layer replay_m_layer.cpp)
target_compile_features(d4r-m-layer PRIVATE cxx_std_17)
target_compile_definitions(d4r-m-layer PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
target_link_libraries(d4r-m-layer PRIVATE d3d12 dxgi d3dcompiler)
if(MSVC)
    target_compile_options(d4r-m-layer PRIVATE /W4 /WX /permissive-)
else()
    target_compile_options(d4r-m-layer PRIVATE -Wall -Wextra -Werror)
endif()
find_program(DXC_EXECUTABLE NAMES dxc REQUIRED)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/dec5.dxil"
    COMMAND "${DXC_EXECUTABLE}" -T cs_6_6 -E main -enable-16bit-types -O3
            -DC=160 -DNH=8 "${CMAKE_CURRENT_SOURCE_DIR}/layer_k.hlsl"
            -Fo "${CMAKE_CURRENT_BINARY_DIR}/dec5.dxil"
    DEPENDS layer_k.hlsl k_math.hlsli VERBATIM)
add_custom_target(d4r-k-shader ALL DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/dec5.dxil")
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/matrix.dxil"
    COMMAND "${DXC_EXECUTABLE}" -T cs_6_6 -E main -enable-16bit-types -O3
            "${CMAKE_CURRENT_SOURCE_DIR}/matrix_tile.hlsl" -Fo "${CMAKE_CURRENT_BINARY_DIR}/matrix.dxil"
    DEPENDS matrix_tile.hlsl ags_wmma.hlsli VERBATIM)
add_custom_target(d4r-matrix-shader ALL DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/matrix.dxil")
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/dec5-wmma.dxil"
    COMMAND "${DXC_EXECUTABLE}" -T cs_6_6 -E main -enable-16bit-types -O3 -DC=160 -DNH=8
            "${CMAKE_CURRENT_SOURCE_DIR}/layer_k_wmma.hlsl" -Fo "${CMAKE_CURRENT_BINARY_DIR}/dec5-wmma.dxil"
    DEPENDS layer_k_wmma.hlsl ags_wmma.hlsli k_math.hlsli VERBATIM)
add_custom_target(d4r-k-wmma-shader ALL DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/dec5-wmma.dxil")

option(D4R_D3D12_VULKAN_REFERENCE "Build the existing Vulkan runner for output comparisons" OFF)
if(D4R_D3D12_VULKAN_REFERENCE)
    find_package(Vulkan REQUIRED)
    add_executable(d4r-vulkan-layer ../evk.cpp)
    target_compile_features(d4r-vulkan-layer PRIVATE cxx_std_17)
    target_link_libraries(d4r-vulkan-layer PRIVATE Vulkan::Vulkan)
endif()

include(CTest)
if(BUILD_TESTING)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_test(NAME k-layer-cpu
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/test_reference.py")
    option(D4R_D3D12_WMMA_TESTS "Run experimental AMD WMMA tests (validated on RX 9070 XT)" OFF)
    option(D4R_D3D12_DEBUG_TESTS "Enable the optional D3D12 debug layer for WMMA accuracy tests" OFF)
    set(DEBUG_ARGS)
    if(D4R_D3D12_DEBUG_TESTS)
        list(APPEND DEBUG_ARGS --debug)
    endif()
    if(D4R_D3D12_WMMA_TESTS)
        foreach(TILE_CASE IN ITEMS "1;1" "128;1" "128;128")
            list(GET TILE_CASE 0 TILES)
            list(GET TILE_CASE 1 STEPS)
            set(MATRIX_ARGS --runner $<TARGET_FILE:d4r-k-layer> --tiles ${TILES} --steps ${STEPS}
                --repeats 20 --dxc "${DXC_EXECUTABLE}" ${DEBUG_ARGS})
            if(D4R_D3D12_VULKAN_REFERENCE)
                find_program(GLSLANG_EXECUTABLE NAMES glslangValidator REQUIRED)
                list(APPEND MATRIX_ARGS --vulkan-runner $<TARGET_FILE:d4r-vulkan-layer> --glslang "${GLSLANG_EXECUTABLE}")
            endif()
            add_test(NAME matrix-${TILES}-${STEPS}
                COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/test_matrix.py"
                "${CMAKE_CURRENT_BINARY_DIR}/test-matrix-${TILES}-${STEPS}" ${MATRIX_ARGS})
            set_tests_properties(matrix-${TILES}-${STEPS} PROPERTIES TIMEOUT 120)
        endforeach()
        add_test(NAME k-layer-wmma-dec5
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/test_layer.py"
            "${CMAKE_CURRENT_BINARY_DIR}/test-wmma-dec5" --runner $<TARGET_FILE:d4r-k-layer>
            --synthetic 160 --heads 8 --width 32 --height 32 --backend wmma ${DEBUG_ARGS}
            --dxc "${DXC_EXECUTABLE}" --repeats 5)
        add_test(NAME k-layer-wmma-shifted
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/test_layer.py"
            "${CMAKE_CURRENT_BINARY_DIR}/test-wmma-shifted" --runner $<TARGET_FILE:d4r-k-layer>
            --synthetic 64 --heads 2 --width 11 --height 9 --shift 4 3 --backend wmma ${DEBUG_ARGS}
            --dxc "${DXC_EXECUTABLE}" --repeats 5)
        set_tests_properties(k-layer-wmma-dec5 k-layer-wmma-shifted PROPERTIES TIMEOUT 120)
        foreach(VARIANT IN ITEMS wave-sync prefetch combined)
            add_test(NAME k-layer-wmma-${VARIANT}
                COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/test_layer.py"
                "${CMAKE_CURRENT_BINARY_DIR}/test-wmma-${VARIANT}" --runner $<TARGET_FILE:d4r-k-layer>
                --synthetic 96 --heads 4 --width 33 --height 17 --shift 7 3 --seed 917
                --backend wmma --wmma-variant ${VARIANT} ${DEBUG_ARGS}
                --dxc "${DXC_EXECUTABLE}" --repeats 5)
            set_tests_properties(k-layer-wmma-${VARIANT} PROPERTIES TIMEOUT 120)
        endforeach()
        if(D4R_D3D12_VULKAN_REFERENCE)
            add_test(NAME k-layer-wmma-vulkan-reference
                COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/test_layer.py"
                "${CMAKE_CURRENT_BINARY_DIR}/test-wmma-vulkan" --runner $<TARGET_FILE:d4r-k-layer>
                --synthetic 160 --heads 8 --width 32 --height 32 --backend wmma
                --dxc "${DXC_EXECUTABLE}" --repeats 20 --vulkan-time
                --glslang "${GLSLANG_EXECUTABLE}" --vulkan-runner $<TARGET_FILE:d4r-vulkan-layer>)
            set_tests_properties(k-layer-wmma-vulkan-reference PROPERTIES TIMEOUT 120)
        endif()
    endif()
    option(D4R_D3D12_GPU_TESTS "Run accuracy tests on the hardware D3D12 adapter" OFF)
    if(D4R_D3D12_GPU_TESTS)
        add_test(NAME k-layer-dec5
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/test_layer.py"
            "${CMAKE_CURRENT_BINARY_DIR}/test-dec5" --runner $<TARGET_FILE:d4r-k-layer>
            --synthetic 160 --heads 8 --dxc "${DXC_EXECUTABLE}" --repeats 5)
        add_test(NAME k-layer-shifted
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/test_layer.py"
            "${CMAKE_CURRENT_BINARY_DIR}/test-shifted" --runner $<TARGET_FILE:d4r-k-layer>
            --synthetic 64 --heads 2 --width 11 --height 9 --shift 4 3
            --dxc "${DXC_EXECUTABLE}" --repeats 5)
        set_tests_properties(k-layer-dec5 k-layer-shifted PROPERTIES TIMEOUT 120)
        if(D4R_D3D12_VULKAN_REFERENCE)
            find_program(GLSLANG_EXECUTABLE NAMES glslangValidator REQUIRED)
            add_test(NAME k-layer-vulkan-reference
                COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/test_layer.py"
                "${CMAKE_CURRENT_BINARY_DIR}/test-vulkan" --runner $<TARGET_FILE:d4r-k-layer>
                --synthetic 160 --heads 8 --dxc "${DXC_EXECUTABLE}" --repeats 5
                --glslang "${GLSLANG_EXECUTABLE}" --vulkan-runner $<TARGET_FILE:d4r-vulkan-layer> --vulkan-time)
            set_tests_properties(k-layer-vulkan-reference PROPERTIES TIMEOUT 120)
        endif()
    endif()
endif()

# Milestone 1: record-only host on a borrowed device, with a diagnostic shader in place of inference.
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/diagnostic_cs.h"
    COMMAND "${DXC_EXECUTABLE}" -T cs_6_0 -E main -O3 -Vn d4r_diagnostic_cs
            "${CMAKE_CURRENT_SOURCE_DIR}/diagnostic.hlsl" -Fh "${CMAKE_CURRENT_BINARY_DIR}/diagnostic_cs.h"
    DEPENDS diagnostic.hlsl VERBATIM)
add_library(d4r-d3d12-host STATIC host.cpp "${CMAKE_CURRENT_BINARY_DIR}/diagnostic_cs.h")
target_compile_features(d4r-d3d12-host PUBLIC cxx_std_17)
target_compile_definitions(d4r-d3d12-host PUBLIC NOMINMAX WIN32_LEAN_AND_MEAN)
target_include_directories(d4r-d3d12-host PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}" PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
target_link_libraries(d4r-d3d12-host PUBLIC d3d12)
add_executable(d4r-host-test test_host.cpp)
target_link_libraries(d4r-host-test PRIVATE d4r-d3d12-host dxgi)
foreach(TARGET IN ITEMS d4r-d3d12-host d4r-host-test)
    if(MSVC)
        target_compile_options(${TARGET} PRIVATE /W4 /WX /permissive-)
    else()
        target_compile_options(${TARGET} PRIVATE -Wall -Wextra -Werror)
    endif()
endforeach()
if(BUILD_TESTING AND D4R_D3D12_GPU_TESTS)
    add_test(NAME host-contract COMMAND d4r-host-test ${DEBUG_ARGS})
    set_tests_properties(host-contract PROPERTIES TIMEOUT 120)
endif()

# Milestone 1: D3D12 -> Vulkan interop cost (shared buffers and a shared fence imported as a timeline semaphore).
find_package(Vulkan)
if(Vulkan_FOUND)
    add_executable(d4r-interop-bench interop_bench.cpp)
    target_compile_features(d4r-interop-bench PRIVATE cxx_std_17)
    target_compile_definitions(d4r-interop-bench PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
    target_link_libraries(d4r-interop-bench PRIVATE d3d12 dxgi Vulkan::Vulkan)
    if(MSVC)
        target_compile_options(d4r-interop-bench PRIVATE /W4 /WX /permissive-)
    endif()
endif()
