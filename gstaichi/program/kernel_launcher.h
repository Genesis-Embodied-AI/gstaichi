#pragma once

#include "gstaichi/codegen/compiled_kernel_data.h"
#include "gstaichi/program/launch_context_builder.h"

namespace gstaichi::lang {

class KernelLauncher {
 public:
  using Handle = KernelLaunchHandle;

  virtual void launch_kernel(const CompiledKernelData &compiled_kernel_data,
                             LaunchContextBuilder &ctx) = 0;

  virtual ~KernelLauncher() = default;
};

}  // namespace gstaichi::lang
