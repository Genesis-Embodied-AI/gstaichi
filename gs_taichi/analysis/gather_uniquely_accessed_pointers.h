#pragma once

#include "gs_taichi/ir/pass.h"

namespace gs_taichi::lang {

class GatherUniquelyAccessedBitStructsPass : public Pass {
 public:
  static const PassID id;

  struct Result {
    std::unordered_map<OffloadedStmt *,
                       std::unordered_map<const SNode *, GlobalPtrStmt *>>
        uniquely_accessed_bit_structs;
  };
};

}  // namespace taichi::lang
