#pragma once

#include "gstaichi/ir/pass.h"

namespace gstaichi::lang {

class MakeBlockLocalPass : public Pass {
 public:
  static const PassID id;

  struct Args {
    std::string kernel_name;
    bool verbose;
  };
};

}  // namespace gstaichi::lang
