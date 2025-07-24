#include "gs_taichi/analysis/bls_analyzer.h"
#include "gs_taichi/ir/ir.h"
#include "gs_taichi/ir/statements.h"
#include "gs_taichi/ir/transforms.h"
#include "gs_taichi/ir/analysis.h"
#include "gs_taichi/ir/scratch_pad.h"
#include "gs_taichi/system/profiler.h"

namespace gs_taichi::lang {

// TODO: rename scratch_pad to block_local_cache? Need to get rid of the
// scratch_pad term

namespace irpass {

std::unique_ptr<ScratchPads> initialize_scratch_pad(OffloadedStmt *offload) {
  TI_AUTO_PROF
  TI_ASSERT(offload->task_type == OffloadedTaskType::struct_for);
  std::unique_ptr<ScratchPads> pads;
  pads = std::make_unique<ScratchPads>();
  for (auto snode : offload->mem_access_opt.get_snodes_with_flag(
           SNodeAccessFlag::block_local)) {
    pads->insert(snode);
  }
  BLSAnalyzer bls_analyzer(offload, pads.get());
  bool analysis_ok = bls_analyzer.run();
  if (!analysis_ok) {
    TI_ERROR("BLS analysis failed !");
  }
  pads->finalize();
  return pads;
}

}  // namespace irpass

}  // namespace gs_taichi::lang
