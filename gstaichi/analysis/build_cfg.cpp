#include "gstaichi/ir/control_flow_graph.h"
#include "gstaichi/ir/ir.h"
#include "gstaichi/ir/statements.h"
#include "gstaichi/program/function.h"

namespace gstaichi::lang {

struct CFGFuncKey {
  FunctionKey func_key{"", -1, -1};
  bool in_parallel_for{false};

  bool operator==(const CFGFuncKey &other_key) const {
    return func_key == other_key.func_key &&
           in_parallel_for == other_key.in_parallel_for;
  }
};

}  // namespace gstaichi::lang

namespace std {
template <>
struct hash<gstaichi::lang::CFGFuncKey> {
  std::size_t operator()(const gstaichi::lang::CFGFuncKey &key) const noexcept {
    return std::hash<gstaichi::lang::FunctionKey>()(key.func_key) ^
           ((std::size_t)key.in_parallel_for << 32);
  }
};
}  // namespace std

namespace gstaichi::lang {

/**
 * Build a control-flow graph. The resulting graph is guaranteed to have an
 * empty start node and an empty final node.
 *
 * In the following docstrings, node... means a CFGNode's corresponding
 * statements in the CHI IR. Other blocks are just Blocks in the CHI IR.
 * Nodes denoted with "()" mean not yet created when visiting the Stmt/Block.
 *
 * Structures like
 * node_a {
 *   ...
 * } -> node_b, node_c;
 * means node_a has edges to node_b and node_c, or equivalently, node_b and
 * node_c appear in the |next| field of node_a.
 *
 * Structures like
 * node_a {
 *   ...
 * } -> node_b, [node_c if "cond"];
 * means node_a has an edge to node_b, and node_a has an edge to node_c iff
 * the condition "cond" is true.
 *
 * When there can be many CFGNodes in a Block, internal nodes are omitted for
 * simplicity.
 *
 * TODO(#2193): Make sure ReturnStmt is handled properly.
 */
class CFGBuilder : public IRVisitor {
 public:
  CFGBuilder()
      : current_block_(nullptr),
        last_node_in_current_block_(nullptr),
        current_stmt_id_(-1),
        begin_location_(-1),
        current_offload_(nullptr),
        in_parallel_for_(false) {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
    graph_ = std::make_unique<ControlFlowGraph>();
    // Make an empty start node.
    auto start_node = graph_->push_back();
    prev_nodes_.push_back(start_node);
  }

  void visit(Stmt *stmt) override {
    if (stmt->is_container_statement()) {
      TI_ERROR("Visitor for container statement undefined.");
    }
  }

  /**
   * Create a node for the current control-flow graph,
   * mark the current statement as the end location (exclusive) of the node,
   * and add edges from |prev_nodes| to the node.
   *
   * @param next_begin_location The location in the IR block of the first
   * statement in the next node, if the next node is in the same IR block of
   * the node to be returned. Otherwise, next_begin_location must be -1.
   * @return The node which is just created.
   */
  CFGNode *new_node(int next_begin_location) {
    auto node = graph_->push_back(
        current_block_, begin_location_, /*end_location=*/current_stmt_id_,
        /*is_parallel_executed=*/in_parallel_for_,
        /*prev_node_in_same_block=*/last_node_in_current_block_);
    for (auto &prev_node : prev_nodes_) {
      // Now that the "(next node)" is created, we should insert edges
      // "node... -> (next node)" here.
      CFGNode::add_edge(prev_node, node);
    }
    prev_nodes_.clear();
    begin_location_ = next_begin_location;
    last_node_in_current_block_ = node;
    return node;
  }

  /**
   * Structure:
   *
   * block {
   *   node {
   *     ...
   *   } -> node_loop_begin, (the next node after the loop);
   *   continue;
   *   (next node) {
   *     ...
   *   }
   * }
   *
   * Note that the edges are inserted in visit_loop().
   */
  void visit(ContinueStmt *stmt) override {
    // Don't put ContinueStmt in any CFGNodes.
    auto node = new_node(current_stmt_id_ + 1);
    
    if (stmt->from_function_return) {
      // Function returns should break out of loops (exit to next statement),
      // not continue them (jump to loop beginning).
      // Store the unwind node so we can connect it to the appropriate
      // "after-loop" target in run().
      UnwindInfo info;
      info.node = node;
      info.cont_stmt = stmt;
      info.enclosing_loops = loop_scope_stack_;
      unwind_nodes_.push_back(info);
      // Do NOT add to prev_nodes_ - statements after return are unreachable
    } else {
      // Normal continues jump to loop beginnings.
      // Store them so we can connect them in run().
      UnwindInfo info;
      info.node = node;
      info.cont_stmt = stmt;
      info.enclosing_loops = loop_scope_stack_;
      unwind_nodes_.push_back(info);
      // Do NOT add to prev_nodes_ - continues jump back to loop beginning
    }
  }

  }

  /**
   * Structure:
   *
   * block {
   *   node {
   *     ...
   *   } -> (next node), (the next node after the loop);
   *   while_control (possibly break);
   *   (next node) {
   *     ...
   *   }
   * }
   *
   * Note that the edges are inserted in visit_loop().
   */
  void visit(WhileControlStmt *stmt) override {
    // Don't put WhileControlStmt in any CFGNodes.
    auto node = new_node(current_stmt_id_ + 1);
    breaks_in_current_loop_.push_back(node);
    prev_nodes_.push_back(node);
  }

  /**
   * Structure:
   *
   * node_before_if {
   *   ...
   * } -> node_true_branch_begin, node_false_branch_begin;
   * if (...) {
   *   node_true_branch_begin {
   *     ...
   *   } -> ... -> node_true_branch_end;
   *   node_true_branch_end {
   *     ...
   *   } -> (next node);
   * } else {
   *   node_false_branch_begin {
   *     ...
   *   } -> ... -> node_false_branch_end;
   *   node_false_branch_end {
   *     ...
   *   } -> (next node);
   * }
   * (next node) {
   *   ...
   * }
   */
  void visit(IfStmt *if_stmt) override {
    auto before_if = new_node(-1);
    TI_INFO("[CFG DEBUG] IfStmt: created before_if node_id={}",
            graph_->size() - 1);

    CFGNode *true_branch_end = nullptr;
    if (if_stmt->true_statements) {
      auto true_branch_begin = graph_->size();
      TI_INFO("[CFG DEBUG] IfStmt: visiting true branch, first node will be {}",
              true_branch_begin);
      if_stmt->true_statements->accept(this);
      CFGNode::add_edge(before_if, graph_->nodes[true_branch_begin].get());
      true_branch_end = graph_->back();
      TI_INFO(
          "[CFG DEBUG] IfStmt: true branch end node_id={}, begin={}, end={}, "
          "prev.size={}",
          graph_->size() - 1, true_branch_end->begin_location,
          true_branch_end->end_location, true_branch_end->prev.size());
    }
    CFGNode *false_branch_end = nullptr;
    if (if_stmt->false_statements) {
      auto false_branch_begin = graph_->size();
      TI_INFO(
          "[CFG DEBUG] IfStmt: visiting false branch, first node will be {}",
          false_branch_begin);
      if_stmt->false_statements->accept(this);
      CFGNode::add_edge(before_if, graph_->nodes[false_branch_begin].get());
      false_branch_end = graph_->back();
      TI_INFO(
          "[CFG DEBUG] IfStmt: false branch end node_id={}, begin={}, end={}, "
          "prev.size={}",
          graph_->size() - 1, false_branch_end->begin_location,
          false_branch_end->end_location, false_branch_end->prev.size());
    }
    TI_INFO("[CFG DEBUG] IfStmt: prev_nodes_.size() before assertion = {}",
            prev_nodes_.size());
    TI_ASSERT(prev_nodes_.empty());
    if (if_stmt->true_statements)
      prev_nodes_.push_back(true_branch_end);
    if (if_stmt->false_statements)
      prev_nodes_.push_back(false_branch_end);
    if (!if_stmt->true_statements || !if_stmt->false_statements)
      prev_nodes_.push_back(before_if);
    TI_INFO("[CFG DEBUG] IfStmt: prev_nodes_.size() after setting = {}",
            prev_nodes_.size());
    // Container statements don't belong to any CFGNodes.
    begin_location_ = current_stmt_id_ + 1;
  }

  /**
   * Structure ([(next node) if !is_while_true] means the node has an edge to
   * (next node) only when is_while_true is false):
   *
   * node_before_loop {
   *   ...
   * } -> node_loop_begin, [(next node) if !is_while_true];
   * loop (...) {
   *   node_loop_begin {
   *     ...
   *   } -> ... -> node_loop_end;
   *   node_loop_end {
   *     ...
   *   } -> node_loop_begin, [(next node) if !is_while_true];
   * }
   * (next node) {
   *   ...
   * }
   */
  void visit_loop(Block *body, CFGNode *before_loop, bool is_while_true,
                  Stmt *loop_stmt) {
    int loop_stmt_id = current_stmt_id_;
    auto backup_breaks = std::move(breaks_in_current_loop_);
    breaks_in_current_loop_.clear();

    // Push this loop onto the scope stack
    loop_scope_stack_.push_back(loop_stmt);

    auto loop_begin_index = graph_->size();
    body->accept(this);
    auto loop_begin = graph_->nodes[loop_begin_index].get();
    CFGNode::add_edge(before_loop, loop_begin);
    auto loop_end = graph_->back();
    CFGNode::add_edge(loop_end, loop_begin);
    if (!is_while_true) {
      prev_nodes_.push_back(before_loop);
      prev_nodes_.push_back(loop_end);
    }
    
    // Breaks exit the loop and flow to subsequent statements
    for (auto &node : breaks_in_current_loop_) {
      prev_nodes_.push_back(node);
    }

    // Record the loop-begin node for this loop (for normal continues).
    loop_to_begin_node_[loop_stmt] = loop_begin;

    // Pop this loop from the scope stack
    loop_scope_stack_.pop_back();

    // Container statements don't belong to any CFGNodes.
    begin_location_ = loop_stmt_id + 1;
    breaks_in_current_loop_ = std::move(backup_breaks);
  }

  void visit(WhileStmt *stmt) override {
    visit_loop(stmt->body.get(), new_node(-1), true, stmt);
  }

  void visit(RangeForStmt *stmt) override {
    auto old_in_parallel_for = in_parallel_for_;
    if (!current_offload_)
      in_parallel_for_ = true;
    visit_loop(stmt->body.get(), new_node(-1), false, stmt);
    in_parallel_for_ = old_in_parallel_for;
  }

  void visit(StructForStmt *stmt) override {
    auto old_in_parallel_for = in_parallel_for_;
    if (!current_offload_)
      in_parallel_for_ = true;
    visit_loop(stmt->body.get(), new_node(-1), false, stmt);
    in_parallel_for_ = old_in_parallel_for;
  }

  void visit(MeshForStmt *stmt) override {
    auto old_in_parallel_for = in_parallel_for_;
    if (!current_offload_)
      in_parallel_for_ = true;
    visit_loop(stmt->body.get(), new_node(-1), false, stmt);
    in_parallel_for_ = old_in_parallel_for;
  }

  /**
   * Structure:
   *
   * node_before_offload {
   *   ...
   * } -> node_tls_prologue;
   * node_tls_prologue {
   *   ...
   * } -> node_mesh_prologue;
   * node_mesh_prologue:
   *   ...
   * } -> node_bls_prologue;
   * node_bls_prologue {
   *   ...
   * } -> node_body;
   * node_body {
   *   ...
   * } -> node_bls_epilogue;
   * node_bls_epilogue {
   *   ...
   * } -> node_tls_epilogue;
   * node_tls_epilogue {
   *   ...
   * } -> (next node);
   * (next node) {
   *   ...
   * }
   */
  void visit(OffloadedStmt *stmt) override {
    current_offload_ = stmt;
    if (stmt->tls_prologue) {
      auto before_offload = new_node(-1);
      int offload_stmt_id = current_stmt_id_;
      auto block_begin_index = graph_->size();
      stmt->tls_prologue->accept(this);
      prev_nodes_.push_back(graph_->back());
      // Container statements don't belong to any CFGNodes.
      begin_location_ = offload_stmt_id + 1;
      CFGNode::add_edge(before_offload, graph_->nodes[block_begin_index].get());
    }
    if (stmt->mesh_prologue) {
      auto before_offload = new_node(-1);
      int offload_stmt_id = current_stmt_id_;
      auto block_begin_index = graph_->size();
      stmt->mesh_prologue->accept(this);
      prev_nodes_.push_back(graph_->back());
      // Container statements don't belong to any CFGNodes.
      begin_location_ = offload_stmt_id + 1;
      CFGNode::add_edge(before_offload, graph_->nodes[block_begin_index].get());
    }
    if (stmt->bls_prologue) {
      auto before_offload = new_node(-1);
      int offload_stmt_id = current_stmt_id_;
      auto block_begin_index = graph_->size();
      stmt->bls_prologue->accept(this);
      prev_nodes_.push_back(graph_->back());
      // Container statements don't belong to any CFGNodes.
      begin_location_ = offload_stmt_id + 1;
      CFGNode::add_edge(before_offload, graph_->nodes[block_begin_index].get());
    }
    if (stmt->has_body()) {
      auto before_offload = new_node(-1);
      int offload_stmt_id = current_stmt_id_;
      auto block_begin_index = graph_->size();
      if (stmt->task_type == OffloadedStmt::TaskType::range_for ||
          stmt->task_type == OffloadedStmt::TaskType::struct_for ||
          stmt->task_type == OffloadedStmt::TaskType::mesh_for) {
        in_parallel_for_ = true;
        // Track this offloaded loop as a scope for continues to target
        loop_scope_stack_.push_back(stmt);
      }
      stmt->body->accept(this);
      auto block_begin = graph_->nodes[block_begin_index].get();
      
      // Record the loop-begin node for this offloaded loop
      if (stmt->task_type == OffloadedStmt::TaskType::range_for ||
          stmt->task_type == OffloadedStmt::TaskType::struct_for ||
          stmt->task_type == OffloadedStmt::TaskType::mesh_for) {
        loop_to_begin_node_[stmt] = block_begin;
        loop_scope_stack_.pop_back();
      }
      
      in_parallel_for_ = false;
      prev_nodes_.push_back(graph_->back());
      // Container statements don't belong to any CFGNodes.
      begin_location_ = offload_stmt_id + 1;
      CFGNode::add_edge(before_offload, block_begin);
    }
    if (stmt->bls_epilogue) {
      auto before_offload = new_node(-1);
      int offload_stmt_id = current_stmt_id_;
      auto block_begin_index = graph_->size();
      stmt->bls_epilogue->accept(this);
      prev_nodes_.push_back(graph_->back());
      // Container statements don't belong to any CFGNodes.
      begin_location_ = offload_stmt_id + 1;
      CFGNode::add_edge(before_offload, graph_->nodes[block_begin_index].get());
    }
    if (stmt->tls_epilogue) {
      auto before_offload = new_node(-1);
      int offload_stmt_id = current_stmt_id_;
      auto block_begin_index = graph_->size();
      stmt->tls_epilogue->accept(this);
      prev_nodes_.push_back(graph_->back());
      // Container statements don't belong to any CFGNodes.
      begin_location_ = offload_stmt_id + 1;
      CFGNode::add_edge(before_offload, graph_->nodes[block_begin_index].get());
    }
    current_offload_ = nullptr;
  }

  /**
   * Structure:
   *
   * graph->start_node {
   *   // no statements
   * } -> node_block_begin if this is the first top-level block;
   * block {
   *   node_block_begin {
   *     ...
   *   } -> ... -> node_block_end;
   *   node_block_end {
   *     ...
   *   }
   * }
   *
   * graph->final_node = node_block_end;
   */
  void visit(Block *block) override {
    auto backup_block = current_block_;
    auto backup_last_node = last_node_in_current_block_;
    auto backup_stmt_id = current_stmt_id_;
    // |begin_location| must be -1 (indicating we are not building any CFGNode)
    // when the |current_block| changes.
    TI_ASSERT(begin_location_ == -1);
    TI_ASSERT(prev_nodes_.empty() || graph_->size() == 1);
    current_block_ = block;
    last_node_in_current_block_ = nullptr;
    begin_location_ = 0;

    TI_INFO("[CFG DEBUG] Block: visiting block with {} statements, block={}",
            block->size(), fmt::ptr(block));

    for (int i = 0; i < (int)block->size(); i++) {
      current_stmt_id_ = i;
      TI_INFO("[CFG DEBUG] Block: visiting stmt {}: {}", i,
              block->statements[i]->name());
      block->statements[i]->accept(this);
    }
    current_stmt_id_ = block->size();
    TI_INFO("[CFG DEBUG] Block: creating final node, prev_nodes_.size()={}",
            prev_nodes_.size());
    new_node(-1);  // Each block has a deterministic last node.
    graph_->final_node = (int)graph_->size() - 1;
    TI_INFO("[CFG DEBUG] Block: final node_id={}, begin={}, end={}",
            graph_->size() - 1, graph_->back()->begin_location,
            graph_->back()->end_location);

    current_block_ = backup_block;
    last_node_in_current_block_ = backup_last_node;
    current_stmt_id_ = backup_stmt_id;
  }

  static std::unique_ptr<ControlFlowGraph> run(IRNode *root) {
    CFGBuilder builder;
    root->accept(&builder);
    if (!builder.graph_->nodes[builder.graph_->final_node]->empty()) {
      // Make the final node empty (by adding an empty final node).
      builder.graph_->push_back();
      CFGNode::add_edge(builder.graph_->nodes[builder.graph_->final_node].get(),
                        builder.graph_->back());
      builder.graph_->final_node = (int)builder.graph_->size() - 1;
    }
    
    // Connect all continue/unwind nodes to their appropriate targets:
    // - Normal continues → loop beginning (restart iteration)
    // - Function return unwinds → after loop (break out, continue next statement)
    //
    // Both can have:
    // 1. Explicit scope set → use that
    // 2. levels_up to determine target (levels_up=1 means innermost enclosing loop)
    // 3. Neither → final node (exit entire kernel)
    for (auto &unwind_info : builder.unwind_nodes_) {
      CFGNode *node = unwind_info.node;
      ContinueStmt *cont_stmt = unwind_info.cont_stmt;
      CFGNode *target_node = nullptr;
      Stmt *target_scope = nullptr;
      
      // Determine which loop scope to target
      if (cont_stmt->scope != nullptr) {
        // Case 1: Explicit scope
        target_scope = cont_stmt->scope;
      } else if (cont_stmt->levels_up > 0 &&
                 !unwind_info.enclosing_loops.empty()) {
        // Case 2: Use levels_up to index into the enclosing loops
        // levels_up=1 means the innermost enclosing loop
        // levels_up=2 means the next outer loop, etc.
        int target_idx =
            (int)unwind_info.enclosing_loops.size() - cont_stmt->levels_up;
        if (target_idx >= 0 &&
            target_idx < (int)unwind_info.enclosing_loops.size()) {
          target_scope = unwind_info.enclosing_loops[target_idx];
        } else {
          TI_WARN(
              "[CFG] Continue/unwind has levels_up={} but only {} enclosing loops. "
              "Will target final node.",
              cont_stmt->levels_up, unwind_info.enclosing_loops.size());
        }
      }
      
      // Look up the target node for this scope
      if (target_scope != nullptr) {
        if (cont_stmt->from_function_return) {
          // Function returns should semantically break out of loops and continue
          // to the next statement. However, accurately modeling this in the CFG
          // is complex because the "after-loop" node might not exist yet or might
          // be ambiguous (multiple exit points from the loop).
          //
          // For correctness (especially for DSE), we conservatively connect
          // function returns to the final node. This ensures that stores before
          // the return are considered live (they're visible when the function
          // returns/kernel exits).
          //
          // TODO: For more precise CFG, we could track loop exit points and
          // connect to the appropriate "after-loop" node.
          target_node = builder.graph_->nodes[builder.graph_->final_node].get();
          TI_INFO(
              "[CFG] Function return unwind to final node: node={}, scope={}, "
              "levels_up={}",
              (void *)node, (void *)target_scope, cont_stmt->levels_up);
        } else {
          // Normal continues jump to loop beginning
          auto it = builder.loop_to_begin_node_.find(target_scope);
          if (it != builder.loop_to_begin_node_.end()) {
            target_node = it->second;
            TI_INFO(
                "[CFG] Normal continue to loop begin: node={}, scope={}, "
                "levels_up={}",
                (void *)node, (void *)target_scope, cont_stmt->levels_up);
          }
        }
        
        if (target_node == nullptr) {
          TI_WARN(
              "[CFG] Continue/unwind has target scope but target not found. "
              "Scope={}, from_function_return={}. Will target final node.",
              (void *)target_scope, cont_stmt->from_function_return);
        }
      }
      
      // If we couldn't find a specific target, connect to the final node
      if (target_node == nullptr) {
        target_node = builder.graph_->nodes[builder.graph_->final_node].get();
        TI_INFO(
            "[CFG] Connecting to final node: node={}, scope={}, levels_up={}, "
            "from_function_return={}",
            (void *)node, (void *)cont_stmt->scope, cont_stmt->levels_up,
            cont_stmt->from_function_return);
      }
      
      CFGNode::add_edge(node, target_node);
    }

    return std::move(builder.graph_);
  }

 private:
  struct UnwindInfo {
    CFGNode *node;
    ContinueStmt *cont_stmt;
    // Snapshot of loop_scope_stack_ at the point where this unwind was created
    // Used to resolve levels_up when scope is not explicitly set
    std::vector<Stmt *> enclosing_loops;
  };
  
  std::unique_ptr<ControlFlowGraph> graph_;
  Block *current_block_;
  CFGNode *last_node_in_current_block_;
  std::vector<CFGNode *> breaks_in_current_loop_;
  // All continue nodes (normal and unwind continues)
  std::vector<UnwindInfo> unwind_nodes_;
  // Stack of loop scopes being visited (for tracking nested loops)
  std::vector<Stmt *> loop_scope_stack_;
  // Map from loop scope statement to its loop-begin CFG node (for normal continues)
  std::unordered_map<Stmt *, CFGNode *> loop_to_begin_node_;
  int current_stmt_id_;
  int begin_location_;
  std::vector<CFGNode *> prev_nodes_;
  OffloadedStmt *current_offload_;
  bool in_parallel_for_;
  std::unordered_map<CFGFuncKey, CFGNode *> node_func_begin_;
  std::unordered_map<CFGFuncKey, CFGNode *> node_func_end_;
};

namespace irpass::analysis {
std::unique_ptr<ControlFlowGraph> build_cfg(IRNode *root) {
  return CFGBuilder::run(root);
}
}  // namespace irpass::analysis

}  // namespace gstaichi::lang
