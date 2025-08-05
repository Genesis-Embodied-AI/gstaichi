// The CUDA backend

#pragma once

#include "gstaichi/codegen/codegen.h"
#include "gstaichi/codegen/llvm/codegen_llvm.h"

namespace gstaichi::lang {

class KernelCodeGenCUDA : public KernelCodeGen {
 public:
  explicit KernelCodeGenCUDA(const CompileConfig &compile_config,
                             const Kernel *kernel,
                             IRNode *ir,
                             GsTaichiLLVMContext &tlctx)
      : KernelCodeGen(compile_config, kernel, ir, tlctx) {
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

}  // namespace gstaichi::lang
