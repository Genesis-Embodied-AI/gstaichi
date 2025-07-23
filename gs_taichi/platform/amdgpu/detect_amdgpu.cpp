#include "gs_taichi/platform/amdgpu/detect_amdgpu.h"

#if defined(TI_WITH_AMDGPU)
#include "gs_taichi/rhi/amdgpu/amdgpu_driver.h"
#endif

namespace gs_taichi {

bool is_rocm_api_available() {
#if defined(TI_WITH_AMDGPU)
  return lang::AMDGPUDriver::get_instance_without_context().detected();
#else
  return false;
#endif
}

}  // namespace taichi
