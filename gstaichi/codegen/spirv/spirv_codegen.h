#pragma once

#include "gstaichi/util/lang_util.h"

#include "gstaichi/codegen/spirv/snode_struct_compiler.h"
#include "gstaichi/codegen/spirv/kernel_utils.h"

#include <spirv-tools/libspirv.hpp>
#include <spirv-tools/optimizer.hpp>

namespace gstaichi::lang {

class Kernel;

namespace spirv {

class KernelCodegen {
 public:
  struct Params {
    std::string ti_kernel_name;
    const Kernel *kernel{nullptr};
    const IRNode *ir_root{nullptr};
    std::vector<CompiledSNodeStructs> compiled_structs;
    Arch arch;
    DeviceCapabilityConfig caps;
    bool enable_spv_opt{true};
  };

  explicit KernelCodegen(const Params &params);

  void run(GsTaichiKernelAttributes &kernel_attribs,
           std::vector<std::vector<uint32_t>> &generated_spirv);

 private:
  Params params_;
  KernelContextAttributes ctx_attribs_;

  std::unique_ptr<spvtools::Optimizer> spirv_opt_{nullptr};
  std::unique_ptr<spvtools::SpirvTools> spirv_tools_{nullptr};
  spvtools::OptimizerOptions spirv_opt_options_;
};

}  // namespace spirv
}  // namespace gstaichi::lang
