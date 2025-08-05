/*******************************************************************************
    Copyright (c) The GsTaichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#include "gstaichi/common/core.h"
#include "gstaichi/common/task.h"
#include "gstaichi/util/testing.h"

namespace gstaichi {

class RunTests : public Task {
  std::string run(const std::vector<std::string> &parameters) override {
    return std::to_string(run_tests(parameters));
  }
};

TI_IMPLEMENTATION(Task, RunTests, "test");

}  // namespace gstaichi
