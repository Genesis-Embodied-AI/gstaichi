/*******************************************************************************
    Copyright (c) The GsTaichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#include "gstaichi/python/export.h"
#include "gstaichi/common/interface.h"
#include "gstaichi/util/io.h"

namespace gstaichi {

PYBIND11_MODULE(gstaichi_python, m) {
  m.doc() = "gstaichi_python";

  for (auto &kv : InterfaceHolder::get_instance()->methods) {
    kv.second(&m);
  }

  export_lang(m);
  export_math(m);
  export_misc(m);
}

}  // namespace gstaichi
