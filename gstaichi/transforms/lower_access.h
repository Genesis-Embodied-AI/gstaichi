#pragma once

#include "gstaichi/ir/pass.h"

namespace gstaichi::lang {

class LowerAccessPass : public Pass {
 public:
  static const PassID id;

  struct Args {
    std::vector<SNode *> kernel_forces_no_activate;
    bool lower_atomic;
  };
};

}  // namespace gstaichi::lang
