/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#include "gs_taichi/python/exception.h"

namespace gs_taichi {

void raise_assertion_failure_in_python(const std::string &msg) {
  throw ExceptionForPython(msg);
}

}  // namespace taichi

void taichi_raise_assertion_failure_in_python(const char *msg) {
  gs_taichi::raise_assertion_failure_in_python(std::string(msg));
}
