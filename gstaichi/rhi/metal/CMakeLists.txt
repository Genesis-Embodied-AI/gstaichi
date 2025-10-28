# ./gstaichi/rhi/metal/CMakeLists.txt

set(METAL_RHI metal_rhi)
add_library(${METAL_RHI})
target_sources(${METAL_RHI}
  PRIVATE
    metal_api.cpp
    metal_device.mm
  )

# Workaround for macOS 15.0 SDK SIMD math compatibility issue
target_compile_definitions(${METAL_RHI} PRIVATE
  __sincospif=__sincosf
  __sincospi=__sincos
)

# TODO(#4832) some dependencies here should not be required as they
# are build requirements of other targets.
target_include_directories(${METAL_RHI}
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/external/SPIRV-Cross
    ${PROJECT_SOURCE_DIR}/external/spdlog/include
    ${PROJECT_SOURCE_DIR}/external/glad/include
  )
target_link_libraries(${METAL_RHI} PRIVATE spirv-cross-msl spirv-cross-core "-framework QuartzCore")
