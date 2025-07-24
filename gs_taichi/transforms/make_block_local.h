#pragma once

#include "gs_taichi/ir/pass.h"

namespace gs_taichi::lang {

class MakeBlockLocalPass : public Pass {
 public:
  static const PassID id;

  struct Args {
    std::string kernel_name;
    bool verbose;
  };
};

}  // namespace gs_taichi::lang
