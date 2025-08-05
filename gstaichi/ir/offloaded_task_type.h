#pragma once

#include "gstaichi/common/core.h"

#include <string>

namespace gstaichi::lang {

enum class OffloadedTaskType : int {
#define PER_TASK_TYPE(x) x,
#include "gstaichi/inc/offloaded_task_type.inc.h"
#undef PER_TASK_TYPE
};

std::string offloaded_task_type_name(OffloadedTaskType tt);

}  // namespace gstaichi::lang
