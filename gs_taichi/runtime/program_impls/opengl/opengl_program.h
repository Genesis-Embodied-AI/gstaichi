#pragma once

#include "gs_taichi/runtime/gfx/runtime.h"
#include "gs_taichi/runtime/gfx/snode_tree_manager.h"
#include "gs_taichi/program/program_impl.h"
#include "gs_taichi/runtime/program_impls/gfx/gfx_program.h"

namespace gs_taichi::lang {

class OpenglProgramImpl : public GfxProgramImpl {
 public:
  explicit OpenglProgramImpl(CompileConfig &config);

  void finalize() override;

  void materialize_runtime(KernelProfilerBase *profiler,
                           uint64 **result_buffer_ptr) override;
};

}  // namespace taichi::lang
