#include "gstaichi/ir/ir.h"
#include "gstaichi/ir/statements.h"
#include "gstaichi/ir/transforms.h"
#include "gstaichi/ir/visitors.h"
#include "gstaichi/program/compile_config.h"
#include "gstaichi/ir/analysis.h"
#include <unordered_map>
#include <typeinfo>

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
// - With: flag=true; continue(inner_loop); then after inner_loop: if(flag)
// continue(outer_loop)
//
// Example transformation:
//   for b in range(4):           // outer (offload) loop
//     for j in range(3):         // inner loop
//       if cond:
//         do_stuff()
//         continue(outer)        // <- Problem: can't jump out of nested loop
//         in SPIRV
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

    TI_INFO("[structure_continues] Starting, root type: {}",
            root ? typeid(*root).name() : "nullptr");

    // The root might be a Block containing offloads, or it might be
    // something else. Let's check if root itself is an OffloadedStmt
    if (auto *offload = root->cast<OffloadedStmt>()) {
      TI_INFO("[structure_continues] Root is an OffloadedStmt directly");
      if (offload->task_type == OffloadedStmt::TaskType::range_for ||
          offload->task_type == OffloadedStmt::TaskType::struct_for) {
        return restructure_offload(offload);
      }
      return false;
    }

    // If it's a Block, directly check its statements
    // Note: gather_statements doesn't work for top-level OffloadedStmts
    // because BasicStmtVisitor descends into them without calling the test
    // function
    std::vector<OffloadedStmt *> offloads;
    if (auto *block = root->cast<Block>()) {
      TI_INFO("[structure_continues] Root is a Block with {} statements",
              block->statements.size());
      for (auto &stmt : block->statements) {
        if (auto *offload = stmt->cast<OffloadedStmt>()) {
          TI_INFO("[structure_continues]   Found OffloadedStmt task_type={}",
                  (int)offload->task_type);
          if (offload->task_type == OffloadedStmt::TaskType::range_for ||
              offload->task_type == OffloadedStmt::TaskType::struct_for) {
            offloads.push_back(offload);
          }
        }
      }
    }

    TI_INFO("[structure_continues] Found {} offloaded loops to check",
            offloads.size());

    for (auto *offload : offloads) {
      if (restructure_offload(offload)) {
        modified = true;
      }
    }

    TI_INFO("[structure_continues] Modified: {}", modified);
    return modified;
  }

 private:
  static bool restructure_offload(OffloadedStmt *offload) {
    TI_INFO("[structure_continues] Checking offload {}", (void *)offload);

    // Find all continues that need restructuring in this offload
    auto continues = irpass::analysis::gather_statements(
        offload->body.get(), [offload](Stmt *s) {
          auto *cont = s->cast<ContinueStmt>();
          if (!cont) {
            return false;
          }

          TI_INFO(
              "[structure_continues]   Found continue: scope={}, "
              "scope==offload={}",
              (void *)cont->scope, cont->scope == offload);

          // Check if this continue is inside a nested loop but targets the
          // offload This happens after inlining when a function return becomes
          // a continue
          if (cont->scope == offload) {
            // Find if there's an inner loop between the continue and the
            // offload
            auto *inner_loop = find_innermost_loop(cont);
            TI_INFO(
                "[structure_continues]     inner_loop={}, "
                "inner_loop==offload={}",
                (void *)inner_loop, inner_loop == offload);
            if (inner_loop && inner_loop != offload) {
              TI_INFO(
                  "[structure_continues]     -> This continue needs "
                  "restructuring!");
              return true;
            }
          }

          return false;
        });

    TI_INFO(
        "[structure_continues] Found {} continues to restructure in offload",
        continues.size());

    if (continues.empty()) {
      return false;
    }

    // Group continues by their innermost loop
    std::unordered_map<Stmt *, std::vector<Stmt *>> loop_to_continues;
    for (auto *cont_stmt : continues) {
      auto *inner_loop = find_innermost_loop(cont_stmt);
      if (inner_loop) {
        loop_to_continues[inner_loop].push_back(cont_stmt);
      }
    }

    // Collect insertions to make (loop, flag_var pairs)
    // We'll do insertions in reverse order to avoid index shifting issues
    std::vector<std::pair<Stmt *, Stmt *>> loops_and_flags;

    // Process each loop that has continues
    for (auto &[inner_loop, loop_continues] : loop_to_continues) {
      // Create flag variable for this loop
      auto flag_alloca = Stmt::make<AllocaStmt>(PrimitiveType::u1);
      auto *flag_var = flag_alloca.get();
      offload->body->insert(std::move(flag_alloca), 0);

      // Initialize flag to false
      auto false_const = Stmt::make<ConstStmt>(TypedConstant(false));
      auto *false_ptr = false_const.get();
      offload->body->insert(std::move(false_const), 1);
      offload->body->insert(Stmt::make<LocalStoreStmt>(flag_var, false_ptr), 2);

      // Transform each continue in this loop
      for (auto *cont_stmt : loop_continues) {
        transform_continue(cont_stmt->as<ContinueStmt>(), flag_var, inner_loop,
                           offload);
      }

      // Remember to add flag check after this loop
      loops_and_flags.push_back({inner_loop, flag_var});
    }

    // Add flag checks after loops (in reverse order to avoid index issues)
    for (auto it = loops_and_flags.rbegin(); it != loops_and_flags.rend();
         ++it) {
      add_flag_check_after_loop(it->first, it->second, offload);
    }

    return true;
  }

  static void transform_continue(ContinueStmt *cont,
                                 Stmt *flag_var,
                                 Stmt *inner_loop,
                                 OffloadedStmt *offload) {
    // Strategy: Replace the continue with structured control flow
    // 1. Set flag = true
    // 2. Continue to inner loop (breaking out of it)

    auto *parent_block = cont->parent;

    TI_INFO("Transforming continue: inner_loop={}, offload={}",
            (void *)inner_loop, (void *)offload);

    VecStatement replacement;

    // Set flag = true
    auto true_const = Stmt::make<ConstStmt>(TypedConstant(true));
    auto *true_ptr = true_const.get();
    replacement.push_back(std::move(true_const));
    replacement.push_back(Stmt::make<LocalStoreStmt>(flag_var, true_ptr));

    // Create a continue that breaks out of the inner loop
    auto inner_continue = Stmt::make<ContinueStmt>();
    inner_continue->as<ContinueStmt>()->scope = inner_loop;
    replacement.push_back(std::move(inner_continue));

    parent_block->replace_with(cont, std::move(replacement));
  }

  static void add_flag_check_after_loop(Stmt *loop,
                                        Stmt *flag_var,
                                        OffloadedStmt *offload) {
    // Find the loop in the offload body
    auto &stmts = offload->body->statements;
    for (size_t i = 0; i < stmts.size(); i++) {
      if (contains_stmt(stmts[i].get(), loop)) {
        // Found the containing statement
        TI_INFO("Found loop at index {} in offload body (total {} stmts)", i,
                stmts.size());

        // We need to:
        // 1. Insert flag reset BEFORE the loop (at position i)
        // 2. Insert flag check AFTER the loop (at position i+3, since we added
        // 2 stmts before)

        // Create all statements first
        auto false_const_before = Stmt::make<ConstStmt>(TypedConstant(false));
        auto *false_ptr_before = false_const_before.get();
        auto reset_before =
            Stmt::make<LocalStoreStmt>(flag_var, false_ptr_before);

        auto flag_load = Stmt::make<LocalLoadStmt>(flag_var);
        auto *flag_val = flag_load.get();

        auto cont_outer = Stmt::make<ContinueStmt>();
        cont_outer->as<ContinueStmt>()->scope = offload;

        auto if_stmt = Stmt::make<IfStmt>(flag_val);
        auto if_body_block = std::make_unique<Block>();
        if_body_block->insert(std::move(cont_outer), 0);
        if_stmt->as<IfStmt>()->set_true_statements(std::move(if_body_block));

        // Now insert them using Block::insert() which properly sets parent
        // pointers Insert in reverse order of position to avoid index shifting
        // issues After loop: insert flag check (will be at position i+1 after
        // we insert before)
        offload->body->insert(std::move(if_stmt), i + 1);
        offload->body->insert(std::move(flag_load), i + 1);

        // Before loop: insert flag reset
        offload->body->insert(std::move(reset_before), i);
        offload->body->insert(std::move(false_const_before), i);

        TI_INFO("Inserted flag reset before loop and flag check after loop");
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
      auto *parent_stmt = current->parent_stmt();
      if (parent_stmt) {
        if (parent_stmt->is<RangeForStmt>() || parent_stmt->is<WhileStmt>() ||
            parent_stmt->is<StructForStmt>()) {
          return parent_stmt;
        }
        current = parent_stmt->parent;
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
