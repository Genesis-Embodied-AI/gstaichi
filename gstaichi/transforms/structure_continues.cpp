#include "gstaichi/ir/ir.h"
#include "gstaichi/ir/statements.h"
#include "gstaichi/ir/transforms.h"
#include "gstaichi/ir/visitors.h"
#include "gstaichi/program/compile_config.h"
#include "gstaichi/ir/analysis.h"
#include <unordered_map>
#include <typeinfo>

namespace gstaichi::lang {

// Convert unstructured continues and breaks (function returns through nested loops)
// into structured control flow using flag variables for SPIRV compatibility
//
// Background:
// - When a ti.func contains a return statement inside a nested loop, and that
//   function is called from within a kernel loop, the return needs to continue/break
//   the outer loop (not the inner function loop)
// - After inlining, this becomes a ContinueStmt/BreakStmt with scope pointing to the
//   outer loop, but physically inside a nested inner loop
// - SPIRV does not support jumping out of nested loops with a single continue/break
//
// Solution:
// - Replace: continue/break(outer_loop) inside inner_loop
// - With: flag=true; continue/break(inner_loop); then after inner_loop: if(flag)
// continue/break(outer_loop)
//
// Example transformation for break:
//   while-true {              // outer (function wrapper)
//     for j in range(3):      // inner loop
//       if cond:
//         do_stuff()
//         break(while-true)   // <- Problem: can't jump out of nested loop in SPIRV
//     more_stuff()
//   }
//
// Becomes:
//   flag = false
//   while-true {
//     for j in range(3):
//       if cond:
//         do_stuff()
//         flag = true
//         break(for-j)        // <- break out of inner loop
//     if flag:
//       break(while-true)     // <- now break outer loop
//     more_stuff()            // <- skipped when flag is true
//   }
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

    // Find all breaks that need restructuring (targeting outer loops from inside inner loops)
    auto breaks = irpass::analysis::gather_statements(
        offload->body.get(), [](Stmt *s) {
          auto *brk = s->cast<BreakStmt>();
          if (!brk || !brk->scope) {
            return false;
          }

          TI_INFO("[structure_continues]   Found break: scope={}", 
                  (void *)brk->scope);

          // Check if this break targets an outer loop while being inside an inner loop
          auto *inner_loop = find_innermost_loop(brk);
          if (inner_loop && inner_loop != brk->scope) {
            // The break is inside inner_loop but targets brk->scope (an outer loop)
            TI_INFO(
                "[structure_continues]     inner_loop={}, targets outer scope",
                (void *)inner_loop);
            TI_INFO(
                "[structure_continues]     -> This break needs restructuring!");
            return true;
          }

          return false;
        });

    TI_INFO(
        "[structure_continues] Found {} continues and {} breaks to restructure in offload",
        continues.size(), breaks.size());

    if (continues.empty() && breaks.empty()) {
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

    // Group breaks by their innermost loop
    std::unordered_map<Stmt *, std::vector<Stmt *>> loop_to_breaks;
    for (auto *brk_stmt : breaks) {
      auto *inner_loop = find_innermost_loop(brk_stmt);
      if (inner_loop) {
        loop_to_breaks[inner_loop].push_back(brk_stmt);
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
      offload->body->insert(
          Stmt::make<LocalStoreStmt>(flag_var, false_ptr), 2);

      // Transform each continue in this loop
      for (auto *cont_stmt : loop_continues) {
        transform_continue(cont_stmt->as<ContinueStmt>(), flag_var, inner_loop,
                          offload);
      }

      // Remember to add flag check after this loop
      loops_and_flags.push_back({inner_loop, flag_var});
    }

    // Process each loop that has breaks
    for (auto &[inner_loop, loop_breaks] : loop_to_breaks) {
      // Create flag variable for this loop
      auto flag_alloca = Stmt::make<AllocaStmt>(PrimitiveType::u1);
      auto *flag_var = flag_alloca.get();
      offload->body->insert(std::move(flag_alloca), 0);

      // Initialize flag to false
      auto false_const = Stmt::make<ConstStmt>(TypedConstant(false));
      auto *false_ptr = false_const.get();
      offload->body->insert(std::move(false_const), 1);
      offload->body->insert(
          Stmt::make<LocalStoreStmt>(flag_var, false_ptr), 2);

      // Transform each break in this loop
      for (auto *brk_stmt : loop_breaks) {
        transform_break(brk_stmt->as<BreakStmt>(), flag_var, inner_loop);
      }

      // Remember to add flag check after this loop
      loops_and_flags.push_back({inner_loop, flag_var});
    }

    // Add flag checks after loops (in reverse order to avoid index issues)
    // For continues: add checks in offload body to continue the offload loop
    // For breaks: add checks in the parent block to break the outer loop
    for (auto it = loops_and_flags.rbegin(); it != loops_and_flags.rend();
         ++it) {
      auto *inner_loop = it->first;
      auto *flag_var = it->second;
      
      // Check if this is from a break or continue by looking at what was transformed
      // For now, assume all in this list are from continues (targeting offload)
      bool is_from_continue = false;
      for (auto *cont_stmt : continues) {
        if (find_innermost_loop(cont_stmt) == inner_loop) {
          is_from_continue = true;
          break;
        }
      }
      
      if (is_from_continue) {
        add_flag_check_after_loop(inner_loop, flag_var, offload);
      } else {
        // For breaks, add flag check in the parent block
        add_flag_check_for_break(inner_loop, flag_var);
      }
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

  static void transform_break(BreakStmt *brk,
                              Stmt *flag_var,
                              Stmt *inner_loop) {
    // Strategy: Replace the break with structured control flow
    // 1. Set flag = true
    // 2. Break out of the inner loop

    auto *parent_block = brk->parent;

    TI_INFO("Transforming break: inner_loop={}, outer_scope={}",
            (void *)inner_loop, (void *)brk->scope);

    VecStatement replacement;

    // Set flag = true
    auto true_const = Stmt::make<ConstStmt>(TypedConstant(true));
    auto *true_ptr = true_const.get();
    replacement.push_back(std::move(true_const));
    replacement.push_back(Stmt::make<LocalStoreStmt>(flag_var, true_ptr));

    // Create a break that breaks out of the inner loop
    auto inner_break = Stmt::make<BreakStmt>();
    inner_break->as<BreakStmt>()->scope = inner_loop;
    replacement.push_back(std::move(inner_break));

    parent_block->replace_with(brk, std::move(replacement));
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

  static void add_flag_check_for_break(Stmt *inner_loop, Stmt *flag_var) {
    // For breaks, we need to add the flag check in the parent block
    // (e.g., the while-true wrapper) right after the inner loop

    // Find the parent block (should be the while-true or other outer loop's body)
    auto *parent_block = inner_loop->parent;
    if (!parent_block) {
      TI_WARN("Cannot find parent block for inner_loop");
      return;
    }

    // Find the position of inner_loop in the parent block
    auto &stmts = parent_block->statements;
    size_t loop_pos = 0;
    bool found = false;
    for (size_t i = 0; i < stmts.size(); i++) {
      if (stmts[i].get() == inner_loop) {
        loop_pos = i;
        found = true;
        break;
      }
    }

    if (!found) {
      TI_WARN("Cannot find inner_loop in parent block");
      return;
    }

    TI_INFO("Adding flag check after inner loop at position {} in parent block", 
            loop_pos);

    // Create flag check: if (flag) { break(outer_loop) }
    auto flag_load = Stmt::make<LocalLoadStmt>(flag_var);
    auto *flag_val = flag_load.get();

    // The break should target the parent loop (e.g., while-true)
    // We need to find what the original break was targeting
    // For now, create a break targeting the parent's parent loop
    auto *outer_loop = parent_block->parent_stmt();
    auto outer_break = Stmt::make<BreakStmt>();
    if (outer_loop) {
      outer_break->as<BreakStmt>()->scope = outer_loop;
    }

    auto if_stmt = Stmt::make<IfStmt>(flag_val);
    auto if_body_block = std::make_unique<Block>();
    if_body_block->insert(std::move(outer_break), 0);
    if_stmt->as<IfStmt>()->set_true_statements(std::move(if_body_block));

    // Insert after the loop
    parent_block->insert(std::move(if_stmt), loop_pos + 1);
    parent_block->insert(std::move(flag_load), loop_pos + 1);

    TI_INFO("Inserted flag check after inner loop");
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

// Simpler pass for non-offloaded IR: structure breaks from function returns
// that target outer loops from inside inner loops. This must run BEFORE
// simplification to prevent CFG optimization from incorrectly eliminating them.
bool structure_function_return_breaks(IRNode *root) {
  TI_AUTO_PROF;
  TI_INFO("[structure_function_return_breaks] Starting");
  
  bool modified = false;
  
  // Find all breaks with from_function_return==true that target non-immediate parent loops
  auto breaks = irpass::analysis::gather_statements(root, [](Stmt *s) {
    auto *brk = s->cast<BreakStmt>();
    if (!brk || !brk->from_function_return || !brk->scope) {
      return false;
    }
    
    // Find the innermost loop containing this break
    Block *current = brk->parent;
    Stmt *innermost_loop = nullptr;
    while (current != nullptr) {
      auto *parent_stmt = current->parent_stmt();
      if (parent_stmt && (parent_stmt->is<RangeForStmt>() || 
                         parent_stmt->is<WhileStmt>())) {
        innermost_loop = parent_stmt;
        break;
      }
      if (parent_stmt) {
        current = parent_stmt->parent;
      } else {
        break;
      }
    }
    
    // If the break targets a loop that's not the innermost, it needs restructuring
    if (innermost_loop && innermost_loop != brk->scope) {
      TI_INFO("[structure_function_return_breaks] Found break targeting outer loop");
      return true;
    }
    
    return false;
  });
  
  TI_INFO("[structure_function_return_breaks] Found {} breaks to restructure", breaks.size());
  
  for (auto *brk : breaks) {
    // Find the innermost loop
    Block *current = brk->parent;
    Stmt *inner_loop = nullptr;
    while (current != nullptr) {
      auto *parent_stmt = current->parent_stmt();
      if (parent_stmt && (parent_stmt->is<RangeForStmt>() || 
                         parent_stmt->is<WhileStmt>())) {
        inner_loop = parent_stmt;
        break;
      }
      if (parent_stmt) {
        current = parent_stmt->parent;
      } else {
        break;
      }
    }
    
    if (!inner_loop) {
      continue;
    }
    
    // Create a flag variable before the inner loop
    auto *inner_loop_parent = inner_loop->parent;
    size_t loop_pos = 0;
    for (size_t i = 0; i < inner_loop_parent->statements.size(); i++) {
      if (inner_loop_parent->statements[i].get() == inner_loop) {
        loop_pos = i;
        break;
      }
    }
    
    // Create flag variable
    auto flag_var = Stmt::make<AllocaStmt>(PrimitiveType::u1);
    inner_loop_parent->insert(std::move(flag_var), loop_pos);
    auto *flag_ptr = inner_loop_parent->statements[loop_pos].get();
    loop_pos++; // Adjust for inserted statement
    
    // Initialize flag to false
    auto false_const = Stmt::make<ConstStmt>(TypedConstant(false));
    inner_loop_parent->insert(std::move(false_const), loop_pos);
    auto *false_ptr = inner_loop_parent->statements[loop_pos].get();
    loop_pos++;
    
    auto init_store = Stmt::make<LocalStoreStmt>(flag_ptr, false_ptr);
    inner_loop_parent->insert(std::move(init_store), loop_pos);
    loop_pos++;
    
    // Replace the break with: flag=true; break(inner_loop)
    auto *brk_parent = brk->parent;
    VecStatement replacement;
    
    auto true_const = Stmt::make<ConstStmt>(TypedConstant(true));
    replacement.push_back(std::move(true_const));
    auto *true_ptr = replacement.back().get();
    
    replacement.push_back(Stmt::make<LocalStoreStmt>(flag_ptr, true_ptr));
    
    auto inner_break = Stmt::make<BreakStmt>();
    inner_break->as<BreakStmt>()->scope = inner_loop;
    replacement.push_back(std::move(inner_break));
    
    brk_parent->replace_with(brk, std::move(replacement));
    
    // Add flag check after the inner loop: if (flag) break(outer_loop)
    auto flag_load = Stmt::make<LocalLoadStmt>(flag_ptr);
    inner_loop_parent->insert(std::move(flag_load), loop_pos + 1);
    auto *flag_val = inner_loop_parent->statements[loop_pos + 1].get();
    
    auto outer_break = Stmt::make<BreakStmt>();
    auto *orig_scope = static_cast<BreakStmt*>(brk)->scope;  // Save original scope before brk is deleted
    outer_break->as<BreakStmt>()->scope = orig_scope;
    
    auto if_stmt = Stmt::make<IfStmt>(flag_val);
    auto if_body = std::make_unique<Block>();
    if_body->insert(std::move(outer_break), 0);
    if_stmt->as<IfStmt>()->set_true_statements(std::move(if_body));
    
    inner_loop_parent->insert(std::move(if_stmt), loop_pos + 2);
    
    modified = true;
    TI_INFO("[structure_function_return_breaks] Restructured break");
  }
  
  TI_INFO("[structure_function_return_breaks] Modified: {}", modified);
  return modified;
}

}  // namespace irpass

}  // namespace gstaichi::lang
