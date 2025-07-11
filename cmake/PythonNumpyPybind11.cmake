# Python, numpy, and pybind11
execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import numpy;print(numpy.get_include())"
                OUTPUT_VARIABLE NUMPY_INCLUDE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)

message("-- Python: Using ${PYTHON_EXECUTABLE} as the interpreter")
message("    version: ${PYTHON_VERSION_STRING}")
message("    include: ${PYTHON_INCLUDE_DIR}")
message("    library: ${PYTHON_LIBRARY}")
message("    numpy include: ${NUMPY_INCLUDE_DIR}")

include_directories(${NUMPY_INCLUDE_DIR})

include(FetchContent)

FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.13.6  # Should be less than 3.0.0 for now
)

FetchContent_MakeAvailable(pybind11)
