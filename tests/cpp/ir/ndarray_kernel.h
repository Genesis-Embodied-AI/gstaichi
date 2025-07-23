#pragma once
#include "gs_taichi/ir/ir_builder.h"
#include "gs_taichi/ir/statements.h"
#include "gs_taichi/inc/constants.h"
#include "gs_taichi/program/program.h"

namespace gs_taichi::lang {

std::unique_ptr<Kernel> setup_kernel1(Program *prog);

std::unique_ptr<Kernel> setup_kernel2(Program *prog);
}  // namespace taichi::lang
