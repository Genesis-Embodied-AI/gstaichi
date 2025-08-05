#pragma once

#include "gstaichi/codegen/kernel_compiler.h"
#include "gstaichi/codegen/compiled_kernel_data.h"
#include "gstaichi/runtime/llvm/llvm_context.h"

namespace gstaichi::lang {
namespace LLVM {

class KernelCompiler : public lang::KernelCompiler {
 public:
  struct Config {
    GsTaichiLLVMContext *tlctx{nullptr};
  };

  explicit KernelCompiler(Config config);

  IRNodePtr compile(const CompileConfig &compile_config,
                    const Kernel &kernel_def) const override;

  CKDPtr compile(const CompileConfig &compile_config,
                 const DeviceCapabilityConfig &device_caps,
                 const Kernel &kernel_def,
                 IRNode &chi_ir) const override;

 private:
  Config config_;
};

}  // namespace LLVM
}  // namespace gstaichi::lang
