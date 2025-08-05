/*******************************************************************************
    Copyright (c) The GsTaichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#include "gstaichi/python/exception.h"

namespace gstaichi {

void raise_assertion_failure_in_python(const std::string &msg) {
  throw ExceptionForPython(msg);
}

}  // namespace gstaichi

void gstaichi_raise_assertion_failure_in_python(const char *msg) {
  gstaichi::raise_assertion_failure_in_python(std::string(msg));
}
