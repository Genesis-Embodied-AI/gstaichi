#include "gs_taichi/ir/analysis.h"
#include "gs_taichi/ir/control_flow_graph.h"
#include "gs_taichi/ir/ir.h"
#include "gs_taichi/ir/statements.h"
#include "gs_taichi/ir/transforms.h"

#include <queue>
#include <unordered_map>

namespace gs_taichi::lang {

namespace irpass {

bool determine_ad_stack_size(IRNode *root, const CompileConfig &config) {
  if (irpass::analysis::gather_statements(root, [&](Stmt *s) {
        if (auto ad_stack = s->cast<AdStackAllocaStmt>()) {
          return ad_stack->max_size == 0;  // adaptive
        }
        return false;
      }).empty()) {
    return false;  // no AD-stacks with adaptive size
  }
  auto cfg = analysis::build_cfg(root);
  cfg->simplify_graph();
  cfg->determine_ad_stack_size(config.default_ad_stack_size);
  return true;
}

}  // namespace irpass

}  // namespace gs_taichi::lang
