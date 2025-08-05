#include "gstaichi/ir/ir.h"
#include "gstaichi/ir/statements.h"
#include "gstaichi/ir/transforms.h"
#include "gstaichi/ir/visitors.h"
#include "gstaichi/system/profiler.h"

namespace gstaichi::lang {

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

}  // namespace gstaichi::lang
