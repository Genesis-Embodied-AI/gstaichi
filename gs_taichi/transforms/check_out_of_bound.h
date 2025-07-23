#pragma once

#include "gs_taichi/ir/pass.h"

namespace gs_taichi::lang {

class CheckOutOfBoundPass : public Pass {
 public:
  static const PassID id;

  struct Args {
    std::string kernel_name;
  };
};

}  // namespace taichi::lang
