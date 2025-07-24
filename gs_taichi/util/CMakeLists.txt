# ./taichi/util/CMakeLists.txt

add_library(gs_taichi_util)
target_sources(gs_taichi_util
  PRIVATE
    bit.cpp
    file_sequence_writer.cpp
    image_buffer.cpp
    image_io.cpp
    lang_util.cpp
    offline_cache.cpp
    short_name.cpp
    str.cpp
    testing.cpp
    zip.cpp
  )

target_include_directories(gs_taichi_util
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/external/include
    ${PROJECT_SOURCE_DIR}/external/eigen
    ${LLVM_INCLUDE_DIRS}
  )

target_link_libraries(gs_taichi_util PUBLIC gs_taichi_common)
