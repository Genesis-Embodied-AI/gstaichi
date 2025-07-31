#include "taichi/platform/cuda/detect_cuda.h"

#if defined(TI_WITH_CUDA)
#include "taichi/rhi/cuda/cuda_driver.h"
#endif

namespace taichi {

bool is_cuda_api_available() {
#if defined(TI_WITH_CUDA)
  try {
    auto &instance = lang::CUDADriver::get_instance_without_context();
    int count;
    instance.device_get_count(&count);
    if(count <= 0) {
      std::cerr << "No CUDA devices found." << std::endl;
      return false;
    }
    return instance.detected();
  } catch (const std::exception &e) {
    std::cerr << "Error occurred while checking CUDA availability: " << e.what() << std::endl;
    return false;
  } catch(...) {
    std::cerr << "Unknown error occurred while checking CUDA availability." << std::endl;
    return false;
  }
#else
  return false;
#endif
}

}  // namespace taichi
