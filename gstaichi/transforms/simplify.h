#pragma once

#include "gstaichi/ir/pass.h"

namespace gstaichi::lang {

class FullSimplifyPass : public Pass {
 public:
  static const PassID id;

  struct Args {
    bool after_lower_access;
    // Switch off some optimization in store forwarding if there is an autodiff
    // pass after the full_simplify
    bool autodiff_enabled;
    std::string kernel_name = "";
    bool verbose = false;
  };
};

}  // namespace gstaichi::lang
