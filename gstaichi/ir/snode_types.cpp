#include "gstaichi/ir/snode_types.h"

#include "gstaichi/common/logging.h"

namespace gstaichi::lang {

std::string snode_type_name(SNodeType t) {
  switch (t) {
#define PER_SNODE(i) \
  case SNodeType::i: \
    return #i;

#include "gstaichi/inc/snodes.inc.h"

#undef PER_SNODE
    default:
      TI_NOT_IMPLEMENTED;
  }
}

bool is_gc_able(SNodeType t) {
  return (t == SNodeType::pointer || t == SNodeType::dynamic);
}

}  // namespace gstaichi::lang
