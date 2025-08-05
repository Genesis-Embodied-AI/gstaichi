#pragma once

#include <memory>
#include <functional>

#include "gstaichi/runtime/llvm/llvm_fwd.h"
#include "gstaichi/util/lang_util.h"
#include "gstaichi/jit/jit_module.h"

namespace gstaichi::lang {

// Backend JIT compiler for all archs

class GsTaichiLLVMContext;
struct CompileConfig;

class JITSession {
 protected:
  GsTaichiLLVMContext *tlctx_;
  const CompileConfig &config_;

  std::vector<std::unique_ptr<JITModule>> modules;

 public:
  JITSession(GsTaichiLLVMContext *tlctx, const CompileConfig &config);

  virtual JITModule *add_module(std::unique_ptr<llvm::Module> M,
                                int max_reg = 0) = 0;

  // virtual void remove_module(JITModule *module) = 0;

  virtual void *lookup(const std::string Name) {
    TI_NOT_IMPLEMENTED
  }

  virtual llvm::DataLayout get_data_layout() = 0;

  static std::unique_ptr<JITSession> create(GsTaichiLLVMContext *tlctx,
                                            const CompileConfig &config,
                                            Arch arch);

  virtual ~JITSession() = default;
};

}  // namespace gstaichi::lang
