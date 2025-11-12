#pragma once

#include <pybind11/pybind11.h>

#include "gstaichi/ir/snode.h"

namespace gstaichi::lang {
class Program;
class Ndarray;

pybind11::capsule ndarray_to_dlpack(Program *program,
                                    pybind11::object owner,
                                    Ndarray *ndarray);
pybind11::capsule field_to_dlpack(Program *program,
                                  pybind11::object owner,
                                  SNode *snode);
}  // namespace gstaichi::lang
