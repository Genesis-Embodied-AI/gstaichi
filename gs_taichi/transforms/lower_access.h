#pragma once

#include "gs_taichi/ir/pass.h"

namespace gs_taichi::lang {

class LowerAccessPass : public Pass {
 public:
  static const PassID id;

  struct Args {
    std::vector<SNode *> kernel_forces_no_activate;
    bool lower_atomic;
  };
};

}  // namespace gs_taichi::lang
