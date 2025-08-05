#include "gstaichi/runtime/program_impls/metal/metal_program.h"

#include "gstaichi/analysis/offline_cache_util.h"
#include "gstaichi/codegen/spirv/kernel_compiler.h"
#include "gstaichi/codegen/spirv/compiled_kernel_data.h"
#include "gstaichi/rhi/metal/metal_device.h"
#include "gstaichi/runtime/gfx/snode_tree_manager.h"
#include "gstaichi/runtime/gfx/kernel_launcher.h"
#include "gstaichi/util/offline_cache.h"
#include "gstaichi/rhi/common/host_memory_pool.h"

namespace gstaichi::lang {

MetalProgramImpl::MetalProgramImpl(CompileConfig &config)
    : GfxProgramImpl(config) {
}

void MetalProgramImpl::materialize_runtime(KernelProfilerBase *profiler,
                                           uint64 **result_buffer_ptr) {
  *result_buffer_ptr = (uint64 *)HostMemoryPool::get_instance().allocate(
      sizeof(uint64) * gstaichi_result_buffer_entries, 8);

  device_ = std::unique_ptr<metal::MetalDevice>(metal::MetalDevice::create());

  gfx::GfxRuntime::Params params;
  params.device = device_.get();
  runtime_ = std::make_unique<gfx::GfxRuntime>(std::move(params));
  snode_tree_mgr_ = std::make_unique<gfx::SNodeTreeManager>(runtime_.get());
}

void MetalProgramImpl::enqueue_compute_op_lambda(
    std::function<void(Device *device, CommandList *cmdlist)> op,
    const std::vector<ComputeOpImageRef> &image_refs) {
  runtime_->enqueue_compute_op_lambda(op, image_refs);
}

}  // namespace gstaichi::lang
