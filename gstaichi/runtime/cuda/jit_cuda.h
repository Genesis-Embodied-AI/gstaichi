#include <memory>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"

#include "gstaichi/rhi/cuda/cuda_context.h"
#include "gstaichi/rhi/cuda/cuda_driver.h"
#include "gstaichi/jit/jit_session.h"
#include "gstaichi/util/lang_util.h"
#include "gstaichi/program/program.h"
#include "gstaichi/system/timer.h"
#include "gstaichi/util/file_sequence_writer.h"

#define TI_RUNTIME_HOST
#include "gstaichi/program/context.h"
#undef TI_RUNTIME_HOST

namespace gstaichi::lang {

#if defined(TI_WITH_CUDA)
class JITModuleCUDA : public JITModule {
 private:
  void *module_;

 public:
  explicit JITModuleCUDA(void *module);
  void *lookup_function(const std::string &name) override;
  void call(const std::string &name,
            const std::vector<void *> &arg_pointers,
            const std::vector<int> &arg_sizes) override;
  void launch(const std::string &name,
              std::size_t grid_dim,
              std::size_t block_dim,
              std::size_t dynamic_shared_mem_bytes,
              const std::vector<void *> &arg_pointers,
              const std::vector<int> &arg_sizes) override;
  bool direct_dispatch() const override;
};

class JITSessionCUDA : public JITSession {
 public:
  llvm::DataLayout data_layout;

  JITSessionCUDA(GsTaichiLLVMContext *tlctx,
                 const CompileConfig &config,
                 llvm::DataLayout data_layout);
  JITModule *add_module(std::unique_ptr<llvm::Module> M, int max_reg) override;
  llvm::DataLayout get_data_layout() override;

 private:
  std::string compile_module_to_ptx(std::unique_ptr<llvm::Module> &module);
};

#endif

std::unique_ptr<JITSession> create_llvm_jit_session_cuda(
    GsTaichiLLVMContext *tlctx,
    const CompileConfig &config,
    Arch arch);

}  // namespace gstaichi::lang
