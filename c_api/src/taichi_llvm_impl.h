#pragma once
#ifdef TI_WITH_LLVM

#include "taichi_core_impl.h"

#ifdef TI_WITH_CUDA
#include "gs_taichi/platform/cuda/detect_cuda.h"
#endif

namespace gs_taichi::lang {
class LlvmRuntimeExecutor;
struct CompileConfig;
}  // namespace gs_taichi::lang

namespace capi {

class LlvmRuntime : public Runtime {
 public:
  LlvmRuntime(gs_taichi::Arch arch);
  virtual ~LlvmRuntime();

  void check_runtime_error();
  gs_taichi::lang::Device &get() override;

 private:
  /* Internally used interfaces */
  TiAotModule load_aot_module(const char *module_path) override;
  TiMemory allocate_memory(
      const gs_taichi::lang::Device::AllocParams &params) override;
  void free_memory(TiMemory devmem) override;

  void buffer_copy(const gs_taichi::lang::DevicePtr &dst,
                   const gs_taichi::lang::DevicePtr &src,
                   size_t size) override;

  void flush() override;

  void wait() override;

 private:
  std::unique_ptr<gs_taichi::lang::CompileConfig> cfg_{nullptr};
  std::unique_ptr<gs_taichi::lang::LlvmRuntimeExecutor> executor_{nullptr};
  gs_taichi::uint64 *result_buffer{nullptr};
};

}  // namespace capi

#endif  // TI_WITH_LLVM
