#pragma once

#include <string>

namespace gstaichi::lang {

enum class SNodeType {
#define PER_SNODE(x) x,
#include "gstaichi/inc/snodes.inc.h"
#undef PER_SNODE
};

std::string snode_type_name(SNodeType t);

bool is_gc_able(SNodeType t);

}  // namespace gstaichi::lang
