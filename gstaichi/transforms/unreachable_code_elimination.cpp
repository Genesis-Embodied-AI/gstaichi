#include "gstaichi/ir/ir.h"
#include "gstaichi/ir/statements.h"
#include "gstaichi/ir/transforms.h"
#include "gstaichi/ir/visitors.h"
#include "gstaichi/system/profiler.h"

namespace gstaichi::lang {

// Unconditionally eliminate ContinueStmt's at **ends** of loops
class UselessContinueEliminator : public IRVisitor {
 public:
  bool modified;

  UselessContinueEliminator() : modified(false) {
    allow_undefined_visitor = true;
  }

  void visit(ContinueStmt *stmt) override {
    stmt->parent->erase(stmt);
    modified = true;
  }

  void visit(IfStmt *if_stmt) override {
    if (if_stmt->true_statements && if_stmt->true_statements->size())
      if_stmt->true_statements->back()->accept(this);
    if (if_stmt->false_statements && if_stmt->false_statements->size())
      if_stmt->false_statements->back()->accept(this);
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
    TI_WARN("[unreachable_code_elimination] Processing Block with {} statements", stmt_list->size());
    const int block_size = stmt_list->size();
    
    // Diagnostic: log all statement types
    for (int i = 0; i < block_size; i++) {
      bool is_continue = stmt_list->statements[i]->is<ContinueStmt>();
      bool is_return = stmt_list->statements[i]->is<ReturnStmt>();
      TI_WARN("[unreachable_code_elimination] Statement {}: id={}, is_ContinueStmt={}, is_ReturnStmt={}", 
              i, stmt_list->statements[i]->id, is_continue, is_return);
    }
    
    for (int i = 0; i < block_size - 1; i++) {
      if (stmt_list->statements[i]->is<ContinueStmt>()) {
        // Eliminate statements after ContinueStmt
        TI_WARN("[unreachable_code_elimination] Found ContinueStmt at index {}, removing {} statements after it", 
                i, block_size - 1 - i);
        for (int j = block_size - 1; j > i; j--)
          stmt_list->erase(j);
        modified = true;
        break;
      }
      if (stmt_list->statements[i]->is<ReturnStmt>()) {
        // Eliminate statements after ReturnStmt
        TI_WARN("[unreachable_code_elimination] Found ReturnStmt at index {}, removing {} statements after it", 
                i, block_size - 1 - i);
        for (int j = block_size - 1; j > i; j--)
          stmt_list->erase(j);
        modified = true;
        break;
      }
    }
    
    // Also check the last statement (if block_size > 0)
    if (block_size > 0 && !modified) {
      int last_idx = block_size - 1;
      if (stmt_list->statements[last_idx]->is<ReturnStmt>()) {
        TI_WARN("[unreachable_code_elimination] Found ReturnStmt at last index {}, no statements after it to remove", 
                last_idx);
        // No statements after it, so nothing to remove
      }
    }
    
    TI_WARN("[unreachable_code_elimination] Block now has {} statements after processing", stmt_list->size());
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
    else if (stmt->body)
      stmt->body->accept(this);

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
};

namespace irpass {
bool unreachable_code_elimination(IRNode *root) {
  TI_AUTO_PROF;
  return UnreachableCodeEliminator::run(root);
}
}  // namespace irpass

}  // namespace gstaichi::lang
