#pragma once

#include "gstaichi/ir/pass.h"

namespace gstaichi::lang {

class DemoteMeshStatements : public Pass {
 public:
  static const PassID id;

  struct Args {
    std::string kernel_name;
  };
};

}  // namespace gstaichi::lang
