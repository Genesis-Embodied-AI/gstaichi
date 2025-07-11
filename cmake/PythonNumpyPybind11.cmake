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

message("Using pybind11 version: ${PYBIND11_VERSION}")
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v${PYBIND11_VERSION}  # Should be less than 3.0.0 for now
     # pybind11 3.0.0 caused builds to start failing. This PR downgrades pybind11 to latest <3.0.0 release,
     # which fixes the issue
     # whilst we could probably get 3.0.0 working too, I feel that x.0.0 releases, in opensource (and closed source
     # too often) might not always be very stable, so we might want to wait 3-6 months before trying it?)
)

FetchContent_MakeAvailable(pybind11)
