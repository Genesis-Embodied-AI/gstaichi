#include "gs_taichi/ir/ir.h"
#include "gs_taichi/ir/statements.h"
#include "gs_taichi/ir/transforms.h"
#include "gs_taichi/ir/visitors.h"
#include "gs_taichi/system/profiler.h"

namespace gs_taichi::lang {

namespace {

// Remove all the loop_unique statements.

class RemoveLoopUnique : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;
  DelayedIRModifier modifier;

  void visit(LoopUniqueStmt *stmt) override {
    stmt->replace_usages_with(stmt->input);
    modifier.erase(stmt);
  }

  static bool run(IRNode *node) {
    RemoveLoopUnique pass;
    node->accept(&pass);
    return pass.modifier.modify_ir();
  }
};

}  // namespace

namespace irpass {

bool remove_loop_unique(IRNode *root) {
  TI_AUTO_PROF;
  return RemoveLoopUnique::run(root);
}

}  // namespace irpass

}  // namespace gs_taichi::lang
