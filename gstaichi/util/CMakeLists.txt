# ./gstaichi/util/CMakeLists.txt

add_library(gstaichi_util)
target_sources(gstaichi_util
  PRIVATE
    bit.cpp
    file_sequence_writer.cpp
    lang_util.cpp
    offline_cache.cpp
    short_name.cpp
    str.cpp
    testing.cpp
    zip.cpp
  )

target_include_directories(gstaichi_util
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/external/include
    ${PROJECT_SOURCE_DIR}/external/eigen
    ${LLVM_INCLUDE_DIRS}
  )

target_link_libraries(gstaichi_util PUBLIC gstaichi_common)
