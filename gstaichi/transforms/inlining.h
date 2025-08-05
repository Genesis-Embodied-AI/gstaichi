#pragma once

#include "gstaichi/ir/pass.h"

namespace gstaichi::lang {

class InliningPass : public Pass {
 public:
  static const PassID id;

  struct Args {};
};

}  // namespace gstaichi::lang
