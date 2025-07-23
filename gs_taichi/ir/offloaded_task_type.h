#pragma once

#include "gs_taichi/common/core.h"

#include <string>

namespace gs_taichi::lang {

enum class OffloadedTaskType : int {
#define PER_TASK_TYPE(x) x,
#include "gs_taichi/inc/offloaded_task_type.inc.h"
#undef PER_TASK_TYPE
};

std::string offloaded_task_type_name(OffloadedTaskType tt);

}  // namespace taichi::lang
