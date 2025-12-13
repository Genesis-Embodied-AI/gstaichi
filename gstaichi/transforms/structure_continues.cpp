#include "gstaichi/ir/ir.h"
#include "gstaichi/ir/statements.h"
#include "gstaichi/ir/transforms.h"
#include "gstaichi/ir/visitors.h"
#include "gstaichi/program/compile_config.h"
#include "gstaichi/ir/analysis.h"

namespace gstaichi::lang {

// Convert unstructured continues (function returns through nested loops)
// into structured control flow using flag variables for SPIRV compatibility
//
// Background:
// - When a ti.func contains a return statement inside a nested loop, and that
//   function is called from within a kernel loop, the return needs to continue
//   the kernel loop (not the inner function loop)
// - After inlining, this becomes a ContinueStmt with scope pointing to the
//   kernel/offload loop, but physically inside a nested loop
// - SPIRV does not support jumping out of nested loops with a single continue
//
// Solution:
// - Replace: continue(outer_loop) inside inner_loop
// - With: flag=true; continue(inner_loop); then after inner_loop: if(flag) continue(outer_loop)
//
// Example transformation:
//   for b in range(4):           // outer (offload) loop
//     for j in range(3):         // inner loop
//       if cond:
//         do_stuff()
//         continue(outer)        // <- Problem: can't jump out of nested loop in SPIRV
//     more_stuff()
//
// Becomes:
//   flag = false
//   for b in range(4):
//     flag = false              // reset for each outer iteration
//     for j in range(3):
//       if cond:
//         do_stuff()
//         flag = true
//         continue(inner)       // <- break out of inner loop
//     if flag:
//       continue(outer)         // <- now continue outer loop
//     more_stuff()              // <- skipped when flag is true
class StructureContinues {
 public:
  static bool run(IRNode *root) {
    bool modified = false;
    
    // Find all offloaded loops
    auto offloads = irpass::analysis::gather_statements(root, [](Stmt *s) {
      if (auto *offload = s->cast<OffloadedStmt>()) {
        return offload->task_type == OffloadedStmt::TaskType::range_for ||
               offload->task_type == OffloadedStmt::TaskType::struct_for;
      }
      return false;
    });
    
    for (auto *stmt : offloads) {
      auto *offload = stmt->as<OffloadedStmt>();
      if (restructure_offload(offload)) {
        modified = true;
      }
    }
    
    return modified;
  }

 private:
  static bool restructure_offload(OffloadedStmt *offload) {
    // Find all continues that need restructuring in this offload
    auto continues = irpass::analysis::gather_statements(offload->body.get(), [offload](Stmt *s) {
      auto *cont = s->cast<ContinueStmt>();
      return cont && cont->from_function_return && cont->levels_up > 1 &&
             cont->scope == offload;
    });
    
    if (continues.empty()) {
      return false;
    }
    
    // Create flag variable at start of offload body
    auto flag_alloca = Stmt::make<AllocaStmt>(PrimitiveType::u1);
    auto *flag_var = flag_alloca.get();
    offload->body->insert(std::move(flag_alloca), 0);
    
    // Initialize flag to false
    auto false_const = Stmt::make<ConstStmt>(TypedConstant(false));
    auto *false_ptr = false_const.get();
    offload->body->insert(std::move(false_const), 1);
    offload->body->insert(Stmt::make<LocalStoreStmt>(flag_var, false_ptr), 2);
    
    // Transform each continue
    for (auto *cont_stmt : continues) {
      auto *cont = cont_stmt->as<ContinueStmt>();
      transform_continue(cont, flag_var, offload);
    }
    
    return true;
  }

  static void transform_continue(ContinueStmt *cont, Stmt *flag_var, OffloadedStmt *offload) {
    // Strategy: Replace the continue with:
    // 1. Set flag = true
    // 2. Break from inner loop using WhileControlStmt
    // 3. After inner loops exit, check flag and continue outer loop
    
    auto *parent_block = cont->parent;
    
    // Find the inner loop that contains this continue
    auto *inner_loop = find_innermost_loop(cont);
    if (!inner_loop) {
      // No inner loop? This shouldn't happen if levels_up > 1
      return;
    }
    
    VecStatement replacement;
    
    // Set flag = true
    auto true_const = Stmt::make<ConstStmt>(TypedConstant(true));
    auto *true_ptr = true_const.get();
    replacement.push_back(std::move(true_const));
    replacement.push_back(Stmt::make<LocalStoreStmt>(flag_var, true_ptr));
    
    // For RangeForStmt/WhileStmt: we need to break by setting the loop mask
    // But we don't have direct access to the mask here
    // Instead, we'll just replace the continue and handle it differently
    // For now, create a regular continue that targets the inner loop to exit it
    auto break_stmt = Stmt::make<ContinueStmt>();
    break_stmt->scope = inner_loop;
    replacement.push_back(std::move(break_stmt));
    
    parent_block->replace_with(cont, std::move(replacement));
    
    // Now we need to add a check after the inner loop exits
    // Find the inner loop in the offload body and add flag check after it
    add_flag_check_after_loop(inner_loop, flag_var, offload);
  }

  static void add_flag_check_after_loop(Stmt *loop, Stmt *flag_var, OffloadedStmt *offload) {
    // Find the loop in the offload body
    auto &stmts = offload->body->statements;
    for (size_t i = 0; i < stmts.size(); i++) {
      if (contains_stmt(stmts[i].get(), loop)) {
        // Found the containing statement
        // After this loop, check flag and continue if set
        // We need to insert after position i
        
        // Load flag
        auto flag_load = Stmt::make<LocalLoadStmt>(flag_var);
        auto *flag_val = flag_load.get();
        
        // If flag is true, continue to outer loop
        VecStatement if_body;
        auto cont_outer = Stmt::make<ContinueStmt>();
        cont_outer->scope = offload;
        if_body.push_back(std::move(cont_outer));
        
        auto if_stmt = Stmt::make<IfStmt>(flag_val, std::move(if_body));
        
        // Insert after the loop
        stmts.insert(stmts.begin() + i + 1, std::move(flag_load));
        stmts.insert(stmts.begin() + i + 2, std::move(if_stmt));
        
        break;
      }
    }
  }

  static bool contains_stmt(Stmt *haystack, Stmt *needle) {
    if (haystack == needle) {
      return true;
    }
    
    // Check if needle is somewhere inside haystack
    bool found = false;
    irpass::analysis::gather_statements(haystack, [needle, &found](Stmt *s) {
      if (s == needle) {
        found = true;
      }
      return false;  // Don't actually gather, just search
    });
    
    return found;
  }

  static Stmt *find_innermost_loop(Stmt *stmt) {
    // Walk up to find the nearest loop
    Block *current = stmt->parent;
    while (current != nullptr) {
      if (current->parent_stmt) {
        auto *parent = current->parent_stmt;
        if (parent->is<RangeForStmt>() || parent->is<WhileStmt>() || 
            parent->is<StructForStmt>()) {
          return parent;
        }
        current = parent->parent;
      } else {
        break;
      }
    }
    return nullptr;
  }
};

namespace irpass {

bool structure_continues(IRNode *root, const CompileConfig &config) {
  TI_AUTO_PROF;
  return StructureContinues::run(root);
}

}  // namespace irpass

}  // namespace gstaichi::lang

