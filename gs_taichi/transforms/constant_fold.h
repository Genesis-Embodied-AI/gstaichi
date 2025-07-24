#pragma once

#include "gs_taichi/ir/pass.h"

namespace gs_taichi::lang {

class ConstantFoldPass : public Pass {
 public:
  static const PassID id;

  struct Args {
    Program *program;
  };
};

}  // namespace gs_taichi::lang
