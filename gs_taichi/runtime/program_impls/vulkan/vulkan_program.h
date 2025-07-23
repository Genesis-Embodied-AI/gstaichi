#pragma once
#include "gs_taichi/aot/module_loader.h"
#include "gs_taichi/codegen/spirv/spirv_codegen.h"
#include "gs_taichi/codegen/spirv/snode_struct_compiler.h"
#include "gs_taichi/codegen/spirv/kernel_utils.h"

#include "gs_taichi/rhi/vulkan/vulkan_device_creator.h"
#include "gs_taichi/rhi/vulkan/vulkan_utils.h"
#include "gs_taichi/rhi/vulkan/vulkan_loader.h"
#include "gs_taichi/runtime/gfx/runtime.h"
#include "gs_taichi/runtime/gfx/snode_tree_manager.h"
#include "gs_taichi/rhi/vulkan/vulkan_device.h"

#include "gs_taichi/common/logging.h"
#include "gs_taichi/struct/snode_tree.h"
#include "gs_taichi/program/snode_expr_utils.h"
#include "gs_taichi/program/program_impl.h"
#include "gs_taichi/program/program.h"
#include "gs_taichi/runtime/program_impls/gfx/gfx_program.h"

#include <optional>

namespace gs_taichi::lang {

namespace vulkan {
class VulkanDeviceCreator;
}

class VulkanProgramImpl : public GfxProgramImpl {
 public:
  explicit VulkanProgramImpl(CompileConfig &config);
  ~VulkanProgramImpl() override;

  void materialize_runtime(KernelProfilerBase *profiler,
                           uint64 **result_buffer_ptr) override;

  Device *get_compute_device() override {
    if (embedded_device_) {
      return embedded_device_->device();
    }
    return nullptr;
  }

  Device *get_graphics_device() override {
    if (embedded_device_) {
      return embedded_device_->device();
    }
    return nullptr;
  }

  void finalize() override;

  void enqueue_compute_op_lambda(
      std::function<void(Device *device, CommandList *cmdlist)> op,
      const std::vector<ComputeOpImageRef> &image_refs) override;

 private:
  std::unique_ptr<vulkan::VulkanDeviceCreator> embedded_device_{nullptr};
};
}  // namespace taichi::lang
