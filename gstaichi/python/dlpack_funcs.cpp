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
  void *raw_ptr = nullptr;
  DLDeviceType device_type = DLDeviceType::kDLCPU;
  if(cpu_device != nullptr) {
    std::cout << "cpu_device not nullptr " << (cpu_device != nullptr) << std::endl;
    // std::cout << " dev_alloc.ptr " << devalloc.ptr << std::endl;
    cpu::CpuDevice::AllocInfo alloc_info = cpu_device->get_alloc_info(devalloc);
    raw_ptr = alloc_info.ptr;
    std::cout << " dev_alloc.ptr " << raw_ptr << std::endl;
    std::cout << ((int *)raw_ptr)[0] << std::endl;
    std::cout << ((int *)raw_ptr)[1] << std::endl;
    std::cout << ((int *)raw_ptr)[3] << std::endl;
  }
#if TI_WITH_CUDA
  cuda::CudaDevice *cuda_device = dynamic_cast<cuda::CudaDevice *>(devalloc.device);
  if(cuda_device != nullptr) {
    std::cout << "cuda_device not nullptr " << (cuda_device != nullptr) << std::endl;
    cuda::CudaDevice::AllocInfo alloc_info = cuda_device->get_alloc_info(devalloc);
    raw_ptr = alloc_info.ptr;
    device_type = DLDeviceType::kDLCUDA;
    std::cout << " dev_alloc.ptr " << raw_ptr << std::endl;
  }
#endif // TI_WITH_CUDA

    MyData *my_data = new MyData;
    my_data->value = 31;

    // return my_data;

    int64_t *shape = new int64_t[2];
    shape[0] = 10;
    shape[1] = 3;
    int64_t *strides = new int64_t[2];
    strides[0] = 3;
    strides[1] = 1;

    // DLManagedTensorVersioned *managed_tensor = new DLManagedTensorVersioned;
    DLManagedTensor *managed_tensor = new DLManagedTensor();

    DLTensor &dl_tensor = managed_tensor->dl_tensor;
    dl_tensor.data = raw_ptr;
    dl_tensor.device.device_type = device_type;
    dl_tensor.device.device_id = 0;
    dl_tensor.ndim = 2;
    dl_tensor.dtype = DLDataType{kDLInt, 32, 1};
    dl_tensor.shape = shape;
    dl_tensor.strides = strides;
    dl_tensor.byte_offset = 0;

    std::cout << " device type " << device_type << " " << managed_tensor->dl_tensor.device.device_type <<std::endl;

    // managed_tensor->dl_tensor = dl_tensor;
    managed_tensor->manager_ctx = ndarray;
    managed_tensor->deleter = [](DLManagedTensor *self) {
        // MyData *my_data = static_cast<MyData *>(self->manager_ctx);
        // delete my_data;
        delete[] self->dl_tensor.shape;
        delete[] self->dl_tensor.strides;
        delete self;
    };
    auto deleter = [](PyObject *capsule) {
        // MyData *my_data = static_cast<MyData *>(PyCapsule_GetPointer(capsule, "my_data"));
        // delete my_data;
    };

    // ndarray->inc_ref();

    // pybind11::capsule capsule = pybind11::capsule(static_cast<void *>(my_data), "my_data", deleter);
    pybind11::capsule capsule = pybind11::capsule(static_cast<void *>(managed_tensor), "dltensor", deleter);
    return capsule;
}
}
}
