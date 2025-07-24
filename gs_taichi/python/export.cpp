/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#include "gs_taichi/python/export.h"
#include "gs_taichi/common/interface.h"
#include "gs_taichi/util/io.h"

namespace gs_taichi {

PYBIND11_MODULE(gs_taichi_python, m) {
  m.doc() = "gs_taichi_python";

  for (auto &kv : InterfaceHolder::get_instance()->methods) {
    kv.second(&m);
  }

  export_lang(m);
  export_math(m);
  export_misc(m);
}

}  // namespace taichi
