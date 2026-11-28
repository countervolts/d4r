cmake_minimum_required(VERSION 3.20)
project(d4r_minecraft_native CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
find_package(Vulkan REQUIRED)
find_package(JNI REQUIRED)
find_program(GLSLANG glslangValidator REQUIRED)

set(PREP_SPV "${CMAKE_CURRENT_BINARY_DIR}/prep_spv.h")
add_custom_command(OUTPUT "${PREP_SPV}"
  COMMAND "${GLSLANG}" -V --target-env vulkan1.2 --vn kPrepSpv -o "${PREP_SPV}" "${CMAKE_CURRENT_SOURCE_DIR}/fg/prep.comp"
  DEPENDS fg/prep.comp VERBATIM)
add_custom_target(prep_shader DEPENDS "${PREP_SPV}")

# Frame-generation engine (replays recorded DLSS-G Lite launch scripts through the nvxrt Vulkan layer) + JNI bridge.
add_library(d4r_fg STATIC fg/fg_engine.cpp)
target_include_directories(d4r_fg PUBLIC fg)
target_link_libraries(d4r_fg PUBLIC Vulkan::Headers ${CMAKE_DL_LIBS})
add_library(d4r_mc SHARED fg/fg_mc.cpp)
add_dependencies(d4r_mc prep_shader)
target_include_directories(d4r_mc PRIVATE ${JNI_INCLUDE_DIRS} "${CMAKE_CURRENT_BINARY_DIR}")
target_link_libraries(d4r_mc PRIVATE d4r_fg ${CMAKE_DL_LIBS})
set_target_properties(d4r_mc PROPERTIES BUILD_WITH_INSTALL_RPATH ON INSTALL_RPATH "$ORIGIN")

# Offline test: replays a DLSS-G capture directory through the engine (bit-exact check against the real DLL's output).
add_executable(fg_replay_test fg/fg_replay_test.cpp)
target_link_libraries(fg_replay_test PRIVATE d4r_fg)
