# ./gstaichi/common/CMakeLists.txt

add_library(gstaichi_common)
target_sources(gstaichi_common
  PRIVATE
    cleanup.cpp
    core.cpp
    json.cpp
    logging.cpp
    symbol_version.cpp
    virtual_dir.cpp
    zip.cpp
    miniz.c
    dynamic_loader.cpp
  )

target_include_directories(gstaichi_common
  PUBLIC
    ${PROJECT_SOURCE_DIR}/external/spdlog/include
  PRIVATE
    ${PROJECT_SOURCE_DIR}
  )
