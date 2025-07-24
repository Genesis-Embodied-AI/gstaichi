#pragma once

#include "gs_taichi/codegen/compiled_kernel_data.h"
#include "gs_taichi/program/launch_context_builder.h"

namespace gs_taichi::lang {

class KernelLauncher {
 public:
  using Handle = KernelLaunchHandle;

  virtual void launch_kernel(const CompiledKernelData &compiled_kernel_data,
                             LaunchContextBuilder &ctx) = 0;

  virtual ~KernelLauncher() = default;
};

}  // namespace gs_taichi::lang
