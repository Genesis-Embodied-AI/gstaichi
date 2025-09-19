#include "gstaichi/ir/ir.h"

namespace gstaichi::lang {

    namespace irpass {

class ExtractLocalPointers : public BasicStmtVisitor {
 public:
  ImmediateIRModifier immediate_modifier_;
  DelayedIRModifier delayed_modifier_;

  std::unordered_map<std::pair<Stmt *, int>,
                     Stmt *,
                     hashing::Hasher<std::pair<Stmt *, int>>>
      first_matrix_ptr_;  // mapping an (AllocaStmt, integer) pair to the
                          // first MatrixPtrStmt representing it
  std::unordered_map<int, Stmt *>
      first_const_;  // mapping an integer to the first ConstStmt representing
                     // it
  Block *top_level_;

  explicit ExtractLocalPointers(IRNode *root) : immediate_modifier_(root);
  void visit(OffloadedStmt *stmt) override;
  void visit(MatrixPtrStmt *stmt) override;
  static bool run(IRNode *node);

 private:
  using BasicStmtVisitor::visit;
};

    bool scalarize(IRNode *root, bool half2_optimization_enabled);
} // namespace irpass

}  // namespace gstaichi::lang  