#pragma once

#include "gstaichi/codegen/llvm/compiled_kernel_data.h"
#include "gstaichi/runtime/llvm/kernel_launcher.h"

namespace gstaichi::lang {
namespace cuda {

struct KernelLauncherContext {
  JITModule *jit_module{nullptr};
  std::vector<std::pair<std::vector<int>, Callable::Parameter>> parameters;
  std::vector<OffloadedTask> offloaded_tasks;
};

class KernelLauncher : public LLVM::KernelLauncher {
  using Base = LLVM::KernelLauncher;

 public:
  using Base::Base;

  void launch_llvm_kernel(Handle handle, LaunchContextBuilder &ctx) override;
  Handle register_llvm_kernel(
      const LLVM::CompiledKernelData &compiled) override;

 private:
  std::vector<KernelLauncherContext> contexts_;
};

}  // namespace cuda
}  // namespace gstaichi::lang
