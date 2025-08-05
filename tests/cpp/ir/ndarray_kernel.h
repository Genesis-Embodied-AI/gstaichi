#pragma once
#include "gstaichi/ir/ir_builder.h"
#include "gstaichi/ir/statements.h"
#include "gstaichi/inc/constants.h"
#include "gstaichi/program/program.h"

namespace gstaichi::lang {

std::unique_ptr<Kernel> setup_kernel1(Program *prog);

std::unique_ptr<Kernel> setup_kernel2(Program *prog);
}  // namespace gstaichi::lang
