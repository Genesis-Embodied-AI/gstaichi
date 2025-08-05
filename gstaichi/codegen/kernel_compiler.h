#pragma once

#include <memory>

#include "gstaichi/program/kernel.h"
#include "gstaichi/program/compile_config.h"
#include "gstaichi/codegen/compiled_kernel_data.h"

namespace gstaichi::lang {

class KernelCompiler {
 public:
  using IRNodePtr = std::unique_ptr<IRNode>;
  using CKDPtr = std::unique_ptr<CompiledKernelData>;

  // AST -> CHI IR
  virtual IRNodePtr compile(const CompileConfig &compile_config,
                            const Kernel &kernel_def) const = 0;

  // CHI IR -> CompiledKernelData
  virtual CKDPtr compile(const CompileConfig &compile_config,
                         const DeviceCapabilityConfig &device_caps,
                         const Kernel &kernel_def,
                         IRNode &chi_ir) const = 0;

  virtual ~KernelCompiler() = default;
};

}  // namespace gstaichi::lang
