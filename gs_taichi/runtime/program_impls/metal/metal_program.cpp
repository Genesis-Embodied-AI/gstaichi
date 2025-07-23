#include "gs_taichi/runtime/program_impls/metal/metal_program.h"

#include "gs_taichi/analysis/offline_cache_util.h"
#include "gs_taichi/aot/graph_data.h"
#include "gs_taichi/codegen/spirv/kernel_compiler.h"
#include "gs_taichi/codegen/spirv/compiled_kernel_data.h"
#include "gs_taichi/rhi/metal/metal_device.h"
#include "gs_taichi/runtime/gfx/aot_module_builder_impl.h"
#include "gs_taichi/runtime/gfx/snode_tree_manager.h"
#include "gs_taichi/runtime/gfx/aot_module_loader_impl.h"
#include "gs_taichi/runtime/gfx/kernel_launcher.h"
#include "gs_taichi/util/offline_cache.h"
#include "gs_taichi/rhi/common/host_memory_pool.h"

namespace gs_taichi::lang {

MetalProgramImpl::MetalProgramImpl(CompileConfig &config)
    : GfxProgramImpl(config) {
}

void MetalProgramImpl::materialize_runtime(KernelProfilerBase *profiler,
                                           uint64 **result_buffer_ptr) {
  *result_buffer_ptr = (uint64 *)HostMemoryPool::get_instance().allocate(
      sizeof(uint64) * taichi_result_buffer_entries, 8);

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

}  // namespace taichi::lang
