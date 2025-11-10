#include "dlpack_funcs.h"

#include "dlpack/dlpack.h"

#include "gstaichi/program/ndarray.h"
#include "gstaichi/program/program.h"
#if TI_WITH_CUDA
#include "gstaichi/rhi/cuda/cuda_device.h"
#endif  // TI_WITH_CUDA
#include "gstaichi/rhi/cpu/cpu_device.h"

namespace gstaichi {
namespace lang {
pybind11::capsule dlpack_dump_ndarray_info(Program *program, Ndarray *ndarray) {
    std::cout << "dump_ndarray_info" << std::endl;
  std::cout << ndarray->read_int({0, 0}) << std::endl;
  // int *data_ptr = reinterpret_cast<int *>(get_ndarray_data_ptr
  
  DeviceAllocation devalloc = ndarray->get_device_allocation();

  cpu::CpuDevice *cpu_device = dynamic_cast<cpu::CpuDevice *>(devalloc.device);
  if(cpu_device != nullptr) {
    std::cout << "cpu_device not nullptr " << (cpu_device != nullptr) << std::endl;
  }
  #if TI_WITH_CUDA
  cuda::CudaDevice *cuda_device = dynamic_cast<cuda::CudaDevice *>(devalloc.device);
  if(cuda_device != nullptr) {
    std::cout << "cuda_device not nullptr " << (cuda_device != nullptr) << std::endl;
  }
  #endif // TI_WITH_CUDA

    MyData *my_data = new MyData;
    my_data->value = 31;

    // return my_data;

    // DLTensor dl_tensor;
    // dl_tensor.data = reinterpret_cast<void *>(devalloc.ptr);

    auto deleter = [](PyObject *capsule) {
        MyData *my_data = static_cast<MyData *>(PyCapsule_GetPointer(capsule, "my_data"));
        delete my_data;
    };

    pybind11::capsule capsule = pybind11::capsule(static_cast<void *>(my_data), "my_data", deleter);
    return capsule;
}
}
}
