#include "gstaichi/ir/ir.h"
#include "gstaichi/ir/statements.h"
#include "gstaichi/ir/transforms.h"
#include "gstaichi/ir/visitors.h"
#include "gstaichi/system/profiler.h"

namespace gstaichi::lang {

// Unconditionally eliminate ContinueStmt's at **ends** of loops
// But NOT if they are from function returns (unwinds) or inside branches
// Also eliminate function-return unwinds that are at the top-level kernel scope
class UselessContinueEliminator : public IRVisitor {
 public:
  bool modified;
  bool inside_offloaded = false;

  UselessContinueEliminator() : modified(false) {
    allow_undefined_visitor = true;
  }

  void visit(OffloadedStmt *stmt) override {
    bool prev_inside = inside_offloaded;
    inside_offloaded = true;
    if (stmt->body)
      stmt->body->accept(this);
    inside_offloaded = prev_inside;
  }

  void visit(ContinueStmt *stmt) override {
    // Erase continues that are not from function returns
    if (!stmt->from_function_return) {
      stmt->parent->erase(stmt);
      modified = true;
      return;
    }

    // If we're inside an offloaded kernel and this is a function-return unwind,
    // it's meaningless at kernel scope (there's no function to return from).
    // These come from inlined void functions at the Python frontend level.
    if (inside_offloaded && stmt->from_function_return) {
      stmt->parent->erase(stmt);
      modified = true;
      return;
    }
  }

  void visit(IfStmt *if_stmt) override {
    // Don't recurse into if statements - continues inside branches are not
    // "at the end" of the loop, they are conditional and should be kept
  }
};

// Eliminate useless ContinueStmt, the statements after ContinueStmt and
// unreachable if branches
class UnreachableCodeEliminator : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;
  bool modified;
  UselessContinueEliminator useless_continue_eliminator;
  DelayedIRModifier modifier;

  UnreachableCodeEliminator() : modified(false) {
    allow_undefined_visitor = true;
  }

  void visit(Block *stmt_list) override {
    const int block_size = stmt_list->size();

    // First, eliminate function-return unwinds that may have escaped to kernel
    // scope
    for (int i = 0; i < block_size; i++) {
      if (auto *cont = stmt_list->statements[i]->cast<ContinueStmt>()) {
        if (cont->from_function_return) {
          // Function-return unwinds at kernel scope are meaningless
          modifier.erase(cont);
          modified = true;
          // Continue checking - there may be more
        }
      }
    }

    for (int i = 0; i < block_size - 1; i++) {
      if (stmt_list->statements[i]->is<ContinueStmt>()) {
        // Eliminate statements after ContinueStmt
        for (int j = block_size - 1; j > i; j--)
          stmt_list->erase(j);
        modified = true;
        break;
      }
      // Check if this is an if-statement where both branches end with
      // unwind/return (making code after it unreachable)
      if (auto *if_stmt = stmt_list->statements[i]->cast<IfStmt>()) {
        bool true_branch_returns = false;
        bool false_branch_returns = false;

        // Check if true branch ends with return/unwind
        if (if_stmt->true_statements &&
            !if_stmt->true_statements->statements.empty()) {
          auto *last_stmt = if_stmt->true_statements->statements.back().get();
          true_branch_returns =
              last_stmt->is<ContinueStmt>() || last_stmt->is<ReturnStmt>();
        }

        // Check if false branch ends with return/unwind
        if (if_stmt->false_statements &&
            !if_stmt->false_statements->statements.empty()) {
          auto *last_stmt = if_stmt->false_statements->statements.back().get();
          false_branch_returns =
              last_stmt->is<ContinueStmt>() || last_stmt->is<ReturnStmt>();
        }

        // If both branches return, code after the if-statement is unreachable
        if (true_branch_returns && false_branch_returns) {
          // Erase all statements after this if-statement
          for (int j = block_size - 1; j > i; j--)
            stmt_list->erase(j);
          modified = true;
          break;
        }
      }
    }
    for (auto &stmt : stmt_list->statements)
      stmt->accept(this);
  }

  void visit_loop(Block *body) {
    if (body->size())
      body->back()->accept(&useless_continue_eliminator);
    body->accept(this);
  }

  void visit(RangeForStmt *stmt) override {
    visit_loop(stmt->body.get());
  }

  void visit(StructForStmt *stmt) override {
    visit_loop(stmt->body.get());
  }

  void visit(MeshForStmt *stmt) override {
    visit_loop(stmt->body.get());
  }

  void visit(WhileStmt *stmt) override {
    visit_loop(stmt->body.get());
  }

  void visit(OffloadedStmt *stmt) override {
    if (stmt->tls_prologue)
      stmt->tls_prologue->accept(this);

    if (stmt->mesh_prologue)
      stmt->mesh_prologue->accept(this);

    if (stmt->bls_prologue)
      stmt->bls_prologue->accept(this);

    if (stmt->task_type == OffloadedStmt::TaskType::range_for ||
        stmt->task_type == OffloadedStmt::TaskType::mesh_for ||
        stmt->task_type == OffloadedStmt::TaskType::struct_for)
      visit_loop(stmt->body.get());
    else if (stmt->body) {
      // For non-loop offloaded tasks (like serial), eliminate function-return
      // unwinds since they're meaningless at kernel scope
      eliminate_kernel_scope_unwinds(stmt->body.get());
      stmt->body->accept(this);
    }

    if (stmt->bls_epilogue)
      stmt->bls_epilogue->accept(this);

    if (stmt->tls_epilogue)
      stmt->tls_epilogue->accept(this);
  }

  void visit(IfStmt *if_stmt) override {
    if (if_stmt->cond->is<ConstStmt>()) {
      if (if_stmt->cond->as<ConstStmt>()->val.equal_value(0)) {
        // if (0)
        if (if_stmt->false_statements) {
          modifier.insert_before(
              if_stmt,
              VecStatement(std::move(if_stmt->false_statements->statements)));
        }
        modifier.erase(if_stmt);
        modified = true;
        return;
      } else {
        // if (1)
        if (if_stmt->true_statements) {
          modifier.insert_before(
              if_stmt,
              VecStatement(std::move(if_stmt->true_statements->statements)));
        }
        modifier.erase(if_stmt);
        modified = true;
        return;
      }
    }
    if (if_stmt->true_statements)
      if_stmt->true_statements->accept(this);
    if (if_stmt->false_statements)
      if_stmt->false_statements->accept(this);
  }

  static bool run(IRNode *node) {
    bool modified = false;
    while (true) {
      UnreachableCodeEliminator eliminator;
      node->accept(&eliminator);
      eliminator.modifier.modify_ir();
      if (eliminator.modified ||
          eliminator.useless_continue_eliminator.modified) {
        modified = true;
      } else {
        break;
      }
    }
    return modified;
  }

 private:
  // Eliminate unwind statements from function returns at kernel scope.
  // These are meaningless after frontend inlining of void functions.
  void eliminate_kernel_scope_unwinds(Block *block) {
    bool any_modified = false;
    for (auto &stmt : block->statements) {
      if (auto *cont = stmt->cast<ContinueStmt>()) {
        if (cont->from_function_return) {
          modifier.erase(cont);
          any_modified = true;
        }
      }
      // Recursively check inside if-statements
      if (auto *if_stmt = stmt->cast<IfStmt>()) {
        if (if_stmt->true_statements)
          eliminate_kernel_scope_unwinds_recursive(
              if_stmt->true_statements.get());
        if (if_stmt->false_statements)
          eliminate_kernel_scope_unwinds_recursive(
              if_stmt->false_statements.get());
      }
    }
    if (any_modified)
      modified = true;
  }

  void eliminate_kernel_scope_unwinds_recursive(Block *block) {
    for (auto &stmt : block->statements) {
      if (auto *cont = stmt->cast<ContinueStmt>()) {
        if (cont->from_function_return) {
          modifier.erase(cont);
          modified = true;
        }
      }
      if (auto *if_stmt = stmt->cast<IfStmt>()) {
        if (if_stmt->true_statements)
          eliminate_kernel_scope_unwinds_recursive(
              if_stmt->true_statements.get());
        if (if_stmt->false_statements)
          eliminate_kernel_scope_unwinds_recursive(
              if_stmt->false_statements.get());
      }
    }
  }
};

namespace irpass {
bool unreachable_code_elimination(IRNode *root) {
  TI_AUTO_PROF;
  return UnreachableCodeEliminator::run(root);
}
}  // namespace irpass

}  // namespace gstaichi::lang
