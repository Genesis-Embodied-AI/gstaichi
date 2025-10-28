#pragma once

#include "gstaichi/ir/pass.h"

namespace gstaichi::lang {

class GatherUniquelyAccessedBitStructsPass : public Pass {
 public:
  static const PassID id;

  struct Result {
    std::unordered_map<OffloadedStmt *,
                       std::unordered_map<const SNode *, GlobalPtrStmt *>>
        uniquely_accessed_bit_structs;
  };
};

}  // namespace gstaichi::lang
