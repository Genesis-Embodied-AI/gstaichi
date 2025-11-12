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
pybind11::capsule ndarray_to_dlpack(Program *program,
                                    pybind11::object owner,
                                    Ndarray *ndarray) {
  auto *owner_holder = new pybind11::object(owner);
  new pybind11::object(owner);

  DeviceAllocation devalloc = ndarray->get_device_allocation();

  void *raw_ptr = nullptr;
  DLDeviceType device_type = DLDeviceType::kDLCPU;

  Arch arch = program->compile_config().arch;
  if(arch_is_cpu(arch)) {
    cpu::CpuDevice *cpu_device = static_cast<cpu::CpuDevice *>(devalloc.device);
    cpu::CpuDevice::AllocInfo alloc_info = cpu_device->get_alloc_info(devalloc);
    raw_ptr = alloc_info.ptr;
  }
#if TI_WITH_CUDA
  else if (arch_is_cuda(arch)) {
    cuda::CudaDevice *cuda_device = static_cast<cuda::CudaDevice *>(devalloc.device);
    cuda::CudaDevice::AllocInfo alloc_info = cuda_device->get_alloc_info(devalloc);
    raw_ptr = alloc_info.ptr;
    device_type = DLDeviceType::kDLCUDA;
  }
#endif  // TI_WITH_CUDA

  if (raw_ptr == nullptr) {
    TI_ERROR("Unsupported device type for DLPack conversion");
  }

  std::vector<int> ndarray_shape = ndarray->total_shape();
  int ndim = ndarray_shape.size();

  int64_t *shape = nullptr;
  if (ndim > 0) {
    shape = new int64_t[ndim];
    for (int i = 0; i < ndim; i++) {
      shape[i] = ndarray_shape[i];
    }
  }

  int64_t *strides = nullptr;
  if (ndim > 0) {
    strides = new int64_t[ndim];
    strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--) {
      strides[i] = strides[i + 1] * shape[i + 1];
    }
  }

  DataType ndarray_data_type = ndarray->get_element_data_type();
  uint8_t data_type_code = kDLInt;

  uint8_t element_bits = 32;
  PrimitiveTypeID type_id = ndarray_data_type->as<PrimitiveType>()->type;
  switch (type_id) {
    case PrimitiveTypeID::i32: {
      data_type_code = static_cast<uint8_t>(kDLInt);
      element_bits = 32;
      break;
    }
    case PrimitiveTypeID::i64: {
      data_type_code = static_cast<uint8_t>(kDLInt);
      element_bits = 64;
      break;
    }
    case PrimitiveTypeID::f32: {
      data_type_code = static_cast<uint8_t>(kDLFloat);
      element_bits = 32;
      break;
    }
    case PrimitiveTypeID::f64: {
      data_type_code = static_cast<uint8_t>(kDLFloat);
      element_bits = 64;
      break;
    }
    case PrimitiveTypeID::u1: {
      data_type_code = static_cast<uint8_t>(kDLBool);
      element_bits = 8;
      break;
    }
    default: {
      TI_ERROR("unsupported ndarray data type for dlpack");
    }
  }

  DLManagedTensor *managed_tensor = new DLManagedTensor();

  DLTensor &dl_tensor = managed_tensor->dl_tensor;
  dl_tensor.data = raw_ptr;
  dl_tensor.device.device_type = device_type;
  dl_tensor.device.device_id = 0;
  dl_tensor.ndim = ndim;
  dl_tensor.dtype = DLDataType{data_type_code, element_bits, 1};
  dl_tensor.shape = shape;
  dl_tensor.strides = strides;
  dl_tensor.byte_offset = 0;

  managed_tensor->manager_ctx = static_cast<void *>(owner_holder);
  managed_tensor->deleter = [](DLManagedTensor *self) {
    auto *owner = reinterpret_cast<pybind11::object *>(self->manager_ctx);
    pybind11::gil_scoped_acquire gil;
    delete owner;  // DECREFs the Python object
    if (self->dl_tensor.shape != nullptr) {
      delete[] self->dl_tensor.shape;
      delete[] self->dl_tensor.strides;
    }
    delete self;
  };
  auto deleter = [](PyObject *capsule) {};

  pybind11::capsule capsule = pybind11::capsule(
      static_cast<void *>(managed_tensor), "dltensor", deleter);
  return capsule;
}
}  // namespace lang
}  // namespace gstaichi
