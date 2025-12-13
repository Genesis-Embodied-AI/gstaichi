#include "gstaichi/transforms/inlining.h"
#include "gstaichi/ir/analysis.h"
#include "gstaichi/ir/ir.h"
#include "gstaichi/ir/statements.h"
#include "gstaichi/ir/frontend_ir.h"
#include "gstaichi/ir/transforms.h"
#include "gstaichi/ir/visitors.h"
#include "gstaichi/program/program.h"
#include <functional>

namespace gstaichi::lang {

// Inline all functions.
class Inliner : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;

  explicit Inliner() {
  }

  void visit(FuncCallStmt *stmt) override {
    auto *func = stmt->func;
    TI_ASSERT(func);
    TI_ASSERT(func->parameter_list.size() == stmt->args.size());
    TI_ASSERT(func->ir->is<Block>());
    TI_ASSERT(func->rets.size() <= 1);
    auto inlined_ir = irpass::analysis::clone(func->ir.get());
    if (!func->parameter_list.empty()) {
      irpass::replace_statements(
          inlined_ir.get(),
          /*filter=*/[&](Stmt *s) { return s->is<ArgLoadStmt>(); },
          /*finder=*/
          [&](Stmt *s) {
            // Note: Functions in gstaichi do not support argpack.
            TI_ASSERT(s->as<ArgLoadStmt>()->arg_id.size() == 1);
            return stmt->args[s->as<ArgLoadStmt>()->arg_id[0]];
          });
    }
    if (func->rets.empty()) {
      modifier_.replace_with(
          stmt, VecStatement(std::move(inlined_ir->as<Block>()->statements)));
    } else {
      if (irpass::analysis::gather_statements(inlined_ir.get(), [&](Stmt *s) {
            return s->is<ReturnStmt>();
          }).size() > 1) {
        TI_WARN(
            "Multiple returns in function \"{}\" may not be handled "
            "properly.\n{}",
            func->get_name(), stmt->get_tb());
      }
      // Use a local variable to store the return value
      auto *return_address = inlined_ir->as<Block>()->insert(
          Stmt::make<AllocaStmt>(func->rets[0].dt), /*location=*/0);
      irpass::replace_and_insert_statements(
          inlined_ir.get(),
          /*filter=*/[&](Stmt *s) { return s->is<ReturnStmt>(); },
          /*generator=*/
          [&](Stmt *s) {
            TI_ASSERT(s->as<ReturnStmt>()->values.size() == 1);
            return Stmt::make<LocalStoreStmt>(return_address,
                                              s->as<ReturnStmt>()->values[0]);
          });
      modifier_.insert_before(stmt,
                              std::move(inlined_ir->as<Block>()->statements));
      // Load the return value here
      modifier_.replace_with(stmt, Stmt::make<LocalLoadStmt>(return_address));
    }
  }

  class ScopeAdjuster {
   public:
    Stmt *outermost_loop = nullptr;

    void find_outermost_loop(Stmt *stmt) {
      // Handle both frontend and lowered IR loops
      if (stmt->is<RangeForStmt>() || stmt->is<StructForStmt>() ||
          stmt->is<WhileStmt>() || stmt->is<FrontendForStmt>() ||
          stmt->is<FrontendWhileStmt>()) {
        outermost_loop = stmt;
      }
      // Visit children - need to handle Block separately since it contains
      // statements
      if (auto *block = stmt->cast<Block>()) {
        for (auto &s : block->statements) {
          find_outermost_loop(s.get());
        }
      } else if (auto *if_stmt = stmt->cast<IfStmt>()) {
        if (if_stmt->true_statements) {
          for (auto &s : if_stmt->true_statements->statements) {
            find_outermost_loop(s.get());
          }
        }
        if (if_stmt->false_statements) {
          for (auto &s : if_stmt->false_statements->statements) {
            find_outermost_loop(s.get());
          }
        }
      } else if (auto *while_stmt = stmt->cast<WhileStmt>()) {
        // while_stmt->body is Block *, visit its statements
        for (auto &s : while_stmt->body->statements) {
          find_outermost_loop(s.get());
        }
      } else if (auto *for_stmt = stmt->cast<RangeForStmt>()) {
        // for_stmt->body is Block *, visit its statements
        for (auto &s : for_stmt->body->statements) {
          find_outermost_loop(s.get());
        }
      } else if (auto *for_stmt = stmt->cast<StructForStmt>()) {
        // for_stmt->body is Block *, visit its statements
        for (auto &s : for_stmt->body->statements) {
          find_outermost_loop(s.get());
        }
      } else if (auto *frontend_while = stmt->cast<FrontendWhileStmt>()) {
        for (auto &s : frontend_while->body->statements) {
          find_outermost_loop(s.get());
        }
      } else if (auto *frontend_for = stmt->cast<FrontendForStmt>()) {
        for (auto &s : frontend_for->body->statements) {
          find_outermost_loop(s.get());
        }
      }
      // Add other statement types as needed
    }

    void adjust_scopes(Stmt *stmt) {
      // Handle both frontend and lowered IR continues from function returns
      if (auto *cont = stmt->cast<ContinueStmt>();
          cont && cont->from_function_return) {
        cont->scope = outermost_loop;
      } else if (auto *frontend_cont = stmt->cast<FrontendContinueStmt>();
                 frontend_cont && frontend_cont->function_loop_depth > 0) {
        // Set scope for frontend continues with unwind depth (from function
        // returns)
        frontend_cont->scope = outermost_loop;
      }
      // Visit children same as above
      if (auto *block = stmt->cast<Block>()) {
        for (auto &s : block->statements) {
          adjust_scopes(s.get());
        }
      } else if (auto *if_stmt = stmt->cast<IfStmt>()) {
        if (if_stmt->true_statements) {
          for (auto &s : if_stmt->true_statements->statements) {
            adjust_scopes(s.get());
          }
        }
        if (if_stmt->false_statements) {
          for (auto &s : if_stmt->false_statements->statements) {
            adjust_scopes(s.get());
          }
        }
      } else if (auto *while_stmt = stmt->cast<WhileStmt>()) {
        // while_stmt->body is Block *, visit its statements
        for (auto &s : while_stmt->body->statements) {
          adjust_scopes(s.get());
        }
      } else if (auto *for_stmt = stmt->cast<RangeForStmt>()) {
        // for_stmt->body is Block *, visit its statements
        for (auto &s : for_stmt->body->statements) {
          adjust_scopes(s.get());
        }
      } else if (auto *for_stmt = stmt->cast<StructForStmt>()) {
        // for_stmt->body is Block *, visit its statements
        for (auto &s : for_stmt->body->statements) {
          adjust_scopes(s.get());
        }
      } else if (auto *frontend_while = stmt->cast<FrontendWhileStmt>()) {
        for (auto &s : frontend_while->body->statements) {
          adjust_scopes(s.get());
        }
      } else if (auto *frontend_for = stmt->cast<FrontendForStmt>()) {
        for (auto &s : frontend_for->body->statements) {
          adjust_scopes(s.get());
        }
      }
    }
  };

  void adjust_function_return_scopes(IRNode *node) {
    ScopeAdjuster adjuster;
    auto *root_block = node->as<Block>();
    for (auto &s : root_block->statements) {
      adjuster.find_outermost_loop(
          s.get());  // Find outermost loop in statements
    }
    for (auto &s : root_block->statements) {
      adjuster.adjust_scopes(s.get());  // Adjust scopes in statements
    }
  }

  static bool run(IRNode *node) {
    Inliner inliner;
    inliner.adjust_function_return_scopes(node);
    bool modified = false;
    while (true) {
      node->accept(&inliner);
      if (inliner.modifier_.modify_ir())
        modified = true;
      else
        break;
    }
    return modified;
  }

 private:
  DelayedIRModifier modifier_;
};

const PassID InliningPass::id = "InliningPass";

namespace irpass {

bool inlining(IRNode *root,
              const CompileConfig &config,
              const InliningPass::Args &args) {
  TI_AUTO_PROF;
  return Inliner::run(root);
}

}  // namespace irpass

}  // namespace gstaichi::lang
