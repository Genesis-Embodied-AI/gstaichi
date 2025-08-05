#pragma once
#include "gstaichi/codegen/spirv/spirv_codegen.h"
#include "gstaichi/codegen/spirv/snode_struct_compiler.h"
#include "gstaichi/codegen/spirv/kernel_utils.h"

#include "gstaichi/rhi/metal/metal_device.h"
#include "gstaichi/runtime/gfx/runtime.h"
#include "gstaichi/runtime/gfx/snode_tree_manager.h"

#include "gstaichi/common/logging.h"
#include "gstaichi/struct/snode_tree.h"
#include "gstaichi/program/snode_expr_utils.h"
#include "gstaichi/program/program_impl.h"
#include "gstaichi/program/program.h"
#include "gstaichi/runtime/program_impls/gfx/gfx_program.h"

namespace gstaichi::lang {

class MetalProgramImpl : public GfxProgramImpl {
 public:
  explicit MetalProgramImpl(CompileConfig &config);

  void materialize_runtime(KernelProfilerBase *profiler,
                           uint64 **result_buffer_ptr) override;

  void enqueue_compute_op_lambda(
      std::function<void(Device *device, CommandList *cmdlist)> op,
      const std::vector<ComputeOpImageRef> &image_refs) override;
};

}  // namespace gstaichi::lang
