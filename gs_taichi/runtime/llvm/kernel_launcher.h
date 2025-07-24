#pragma once

#include "gs_taichi/program/kernel_launcher.h"
#include "gs_taichi/codegen/llvm/compiled_kernel_data.h"
#include "gs_taichi/runtime/llvm/llvm_runtime_executor.h"

namespace gs_taichi::lang {
namespace LLVM {

class KernelLauncher : public lang::KernelLauncher {
 public:
  struct Config {
    LlvmRuntimeExecutor *executor{nullptr};
  };

  explicit KernelLauncher(Config config);

  void launch_kernel(const lang::CompiledKernelData &compiled_kernel_data,
                     LaunchContextBuilder &ctx) override;

  virtual void launch_llvm_kernel(Handle handle, LaunchContextBuilder &ctx) = 0;
  virtual Handle register_llvm_kernel(
      const LLVM::CompiledKernelData &compiled) = 0;

 protected:
  Handle make_handle() {
    Handle handle;
    handle.set_launch_id(launch_id_counter_++);
    return handle;
  }

  LlvmRuntimeExecutor *get_runtime_executor() {
    return config_.executor;
  }

 private:
  Config config_;
  int launch_id_counter_{0};
};

}  // namespace LLVM
}  // namespace gs_taichi::lang
