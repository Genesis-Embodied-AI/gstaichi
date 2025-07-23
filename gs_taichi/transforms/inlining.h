#pragma once

#include "gs_taichi/ir/pass.h"

namespace gs_taichi::lang {

class InliningPass : public Pass {
 public:
  static const PassID id;

  struct Args {};
};

}  // namespace taichi::lang
