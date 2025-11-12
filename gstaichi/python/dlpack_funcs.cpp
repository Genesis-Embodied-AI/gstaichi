#include "dlpack_funcs.h"

#include "dlpack/dlpack.h"

#include "gstaichi/program/ndarray.h"
#include "gstaichi/program/program.h"
#if TI_WITH_CUDA
#include "gstaichi/rhi/cuda/cuda_device.h"
#endif  // TI_WITH_CUDA
#include "gstaichi/rhi/cpu/cpu_device.h"

struct MyData {
  int value;
};

namespace gstaichi {
namespace lang {

void validate_arch(Arch arch) {
  if(!arch_is_cpu(arch) && !arch_is_cuda(arch)) {
    TI_ERROR("DLPack conversion is only supported on CPU and CUDA archs");
  }
}

void *get_raw_ptr(Arch arch, Program *program, DeviceAllocation dev_alloc, DLDeviceType *out_device_type) {
    void *raw_ptr = nullptr;
  DLDeviceType device_type = DLDeviceType::kDLCPU;

  if (arch_is_cpu(arch)) {
    cpu::CpuDevice *cpu_device = static_cast<cpu::CpuDevice *>(dev_alloc.device);
    device_type = DLDeviceType::kDLCPU;
    cpu::CpuDevice::AllocInfo alloc_info = cpu_device->get_alloc_info(dev_alloc);
    raw_ptr = alloc_info.ptr;
  }
#if TI_WITH_CUDA
  else if (arch_is_cuda(arch)) {
    cuda::CudaDevice *cuda_device =
        static_cast<cuda::CudaDevice *>(dev_alloc.device);
    device_type = DLDeviceType::kDLCUDA;
    cuda::CudaDevice::AllocInfo alloc_info =
        cuda_device->get_alloc_info(dev_alloc);
    raw_ptr = alloc_info.ptr;
  }
#endif  // TI_WITH_CUDA

  std::cout << "1" << std::endl;
  if (raw_ptr == nullptr) {
    TI_ERROR("Unsupported device type for DLPack conversion");
  }
  return raw_ptr;
}

void get_type_info(DataType dt,
                   uint8_t *p_data_type_code,
                   uint8_t *p_element_bits) {
  PrimitiveType *dt_as_primitive = dt->as<PrimitiveType>();
  if(dt_as_primitive == nullptr) {
    TI_ERROR("unsupported non-primitive data type for dlpack");
  }
  PrimitiveTypeID type_id = dt_as_primitive->type;
  switch (type_id) {
    case PrimitiveTypeID::i32: {
      *p_data_type_code = static_cast<uint8_t>(kDLInt);
      *p_element_bits = 32;
      break;
    }
    case PrimitiveTypeID::i64: {
      *p_data_type_code = static_cast<uint8_t>(kDLInt);
      *p_element_bits = 64;
      break;
    }
    case PrimitiveTypeID::f32: {
      *p_data_type_code = static_cast<uint8_t>(kDLFloat);
      *p_element_bits = 32;
      break;
    }
    case PrimitiveTypeID::f64: {
      *p_data_type_code = static_cast<uint8_t>(kDLFloat);
      *p_element_bits = 64;
      break;
    }
    case PrimitiveTypeID::u1: {
      *p_data_type_code = static_cast<uint8_t>(kDLBool);
      *p_element_bits = 8;
      break;
    }
    default: {
      TI_ERROR("unsupported ndarray data type for dlpack");
    }
  }
}

pybind11::capsule field_to_dlpack(Program *program,
                                    pybind11::object owner,
                                    SNode *snode) {
  if(!snode->is_path_all_dense) {
    TI_ERROR("Only dense fields are supported for dlpack conversion");
  }

  Arch arch = program->compile_config().arch;
  validate_arch(arch);

  int tree_id = snode->get_snode_tree_id();
  DevicePtr tree_device_ptr = program->get_snode_tree_device_ptr(tree_id);

  DLDeviceType device_type = DLDeviceType::kDLCPU;
  void *raw_ptr = get_raw_ptr(arch, program, tree_device_ptr, &device_type);

  DataType dt = snode->dt;

  uint8_t element_bits = 32;
  uint8_t data_type_code = kDLInt;
  get_type_info(dt, &data_type_code, &element_bits);

  int ndim = snode->num_active_indices;
  int64_t *shape = nullptr;
  if (ndim > 0) {
    shape = new int64_t[ndim];
    for(int i = 0; i < ndim; i++) {
      if(snode->physical_index_position[i] != i) {
        TI_ERROR("SNode has non-sequential physical index mapping, which is not supported currently for dlpack conversion");
      }
      int axis_shape = snode->shape_along_axis(i);
      shape[i] = axis_shape;
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

  managed_tensor->deleter = [](DLManagedTensor *self) {
    if (self->dl_tensor.shape != nullptr) {
      delete[] self->dl_tensor.shape;
      delete[] self->dl_tensor.strides;
    }
    delete self;
  };
  auto capsule_deleter = [](PyObject *capsule) {};

  pybind11::capsule capsule =
      pybind11::capsule(managed_tensor, "dltensor", capsule_deleter);
  return capsule;
}

pybind11::capsule ndarray_to_dlpack(Program *program,
                                    pybind11::object owner,
                                    Ndarray *ndarray) {
  Arch arch = program->compile_config().arch;
  validate_arch(arch);

  auto *owner_holder = new pybind11::object(owner);

  DeviceAllocation devalloc = ndarray->get_device_allocation();
  DLDeviceType device_type = DLDeviceType::kDLCPU;
  void *raw_ptr = get_raw_ptr(arch, program, devalloc, &device_type);

  if (raw_ptr == nullptr) {
    TI_ERROR("Unsupported device type for DLPack conversion");
  }

  std::vector<int> ndarray_shape = ndarray->total_shape();
  int ndim = ndarray_shape.size();

  int64_t *shape = nullptr;
  if (ndim > 0) {
    shape = new int64_t[ndim];
    std::copy(ndarray_shape.begin(), ndarray_shape.end(), shape);
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
  uint8_t element_bits = 0;
  // PrimitiveTypeID type_id = ndarray_data_type->as<PrimitiveType>()->type;
  get_type_info(ndarray_data_type, &data_type_code, &element_bits);

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

  managed_tensor->manager_ctx = owner_holder;
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
  auto capsule_deleter = [](PyObject *capsule) {};

  pybind11::capsule capsule =
      pybind11::capsule(managed_tensor, "dltensor", capsule_deleter);
  return capsule;
}
}  // namespace lang
}  // namespace gstaichi
