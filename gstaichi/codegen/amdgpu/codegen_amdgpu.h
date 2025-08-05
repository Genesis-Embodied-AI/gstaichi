// The AMDGPU backend
#pragma once

#include "gstaichi/codegen/codegen.h"
#include "gstaichi/codegen/llvm/codegen_llvm.h"

namespace gstaichi {
namespace lang {

class KernelCodeGenAMDGPU : public KernelCodeGen {
 public:
  KernelCodeGenAMDGPU(const CompileConfig &config,
                      const Kernel *kernel,
                      IRNode *ir,
                      GsTaichiLLVMContext &tlctx)
      : KernelCodeGen(config, kernel, ir, tlctx) {
  }

// TODO: Stop defining this macro guards in the headers
#ifdef TI_WITH_LLVM
  LLVMCompiledTask compile_task(
      int task_codegen_id,
      const CompileConfig &config,
      std::unique_ptr<llvm::Module> &&module = nullptr,
      IRNode *block = nullptr) override;
#endif  // TI_WITH_LLVM
};

}  // namespace lang
}  // namespace gstaichi
