#pragma once

#include <pybind11/pybind11.h>

namespace gstaichi::lang {
class Program;
class Ndarray;

pybind11::capsule ndarray_to_dlpack(Program *program, Ndarray *ndarray);
}
