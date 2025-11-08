#pragma once

#include <pybind11/pybind11.h>

// using py = pybind11;

namespace gstaichi::lang {
class Program;
class Ndarray;

pybind11::capsule dlpack_dump_ndarray_info(Program *program, Ndarray *ndarray);
}
