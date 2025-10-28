#pragma once

#include "gstaichi/ir/pass.h"

namespace gstaichi::lang {

class ConstantFoldPass : public Pass {
 public:
  static const PassID id;

  struct Args {
    Program *program;
  };
};

}  // namespace gstaichi::lang
