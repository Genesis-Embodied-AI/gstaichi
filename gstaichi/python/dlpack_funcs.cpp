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

void validate_arch(Arch arch) {
  if (!arch_is_cpu(arch) && !arch_is_cuda(arch)) {
    TI_ERROR("DLPack conversion is only supported on CPU and CUDA archs");
  }
}

std::pair<void *, DLDeviceType> get_raw_ptr(Arch arch,
                                            Program *program,
                                            DeviceAllocation dev_alloc) {
  void *raw_ptr = nullptr;
  DLDeviceType device_type = DLDeviceType::kDLCPU;
  if (arch_is_cpu(arch)) {
    cpu::CpuDevice *cpu_device =
        static_cast<cpu::CpuDevice *>(dev_alloc.device);
    device_type = DLDeviceType::kDLCPU;
    cpu::CpuDevice::AllocInfo alloc_info =
        cpu_device->get_alloc_info(dev_alloc);
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

  if (raw_ptr == nullptr) {
    TI_ERROR("Unsupported device type for DLPack conversion");
  }
  return std::make_pair(raw_ptr, device_type);
}

std::pair<uint8_t, uint8_t> get_type_info(DataType dt) {
  PrimitiveType *dt_as_primitive = dt->as<PrimitiveType>();
  if (dt_as_primitive == nullptr) {
    TI_ERROR("unsupported non-primitive data type for dlpack");
  }
  PrimitiveTypeID type_id = dt_as_primitive->type;
  uint8_t data_type_code = kDLInt;
  uint8_t element_bits = 0;
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
  return std::make_pair(data_type_code, element_bits);
}

void validate_axis_ordering(SNode *snode, int ndim) {
  std::vector<int> memory_layout_order;
  const SNode *current = snode;
  while (current->parent != nullptr) {
    current = current->parent;
  }
  std::vector<const SNode *> path;
  current = snode;
  while (current != nullptr) {
    path.push_back(current);
    current = current->parent;
  }
  std::reverse(path.begin(), path.end());  // Now path is root -> ... -> place

  for (const SNode *node : path) {
    if (node->type == SNodeType::dense) {
      for (int phys_axis = 0; phys_axis < gstaichi_max_num_indices;
           phys_axis++) {
        if (node->extractors[phys_axis].active) {
          bool was_in_parent = false;
          if (node->parent != nullptr) {
            was_in_parent = node->parent->extractors[phys_axis].active;
          }
          if (!was_in_parent) {
            memory_layout_order.push_back(phys_axis);
          }
        }
      }
    }
  }

  bool has_non_ijk_ordering = false;
  if (memory_layout_order.size() != ndim) {
    has_non_ijk_ordering = true;
  } else {
    for (size_t i = 0; i < memory_layout_order.size(); i++) {
      if (memory_layout_order[i] != static_cast<int>(i)) {
        has_non_ijk_ordering = true;
        break;
      }
    }
  }
  if (has_non_ijk_ordering) {
    TI_ERROR(
        "SNode must have axes in order i, j, k, ... in order to use to_dlpack")
  }
}

pybind11::capsule field_to_dlpack(Program *program,
                                  SNode *snode,
                                  int element_ndim,
                                  int n,
                                  int m) {
  if (!snode->is_path_all_dense) {
    TI_ERROR("Only dense fields are supported for dlpack conversion");
  }

  Arch arch = program->compile_config().arch;
  validate_arch(arch);

  int tree_id = snode->get_snode_tree_id();
  DevicePtr tree_device_ptr = program->get_snode_tree_device_ptr(tree_id);

  int field_in_tree_offset = program->get_field_in_tree_offset(tree_id, snode);

  void *raw_ptr = nullptr;
  DLDeviceType device_type = DLDeviceType::kDLCPU;
  std::tie(raw_ptr, device_type) = get_raw_ptr(arch, program, tree_device_ptr);
  raw_ptr = (void *)((uint64_t)raw_ptr + field_in_tree_offset);

  DataType dt = snode->dt;

  uint8_t element_bits = 32;
  uint8_t data_type_code = kDLInt;
  std::tie(data_type_code, element_bits) = get_type_info(dt);

  int ndim = snode->num_active_indices;

  validate_axis_ordering(snode, ndim);

  int full_ndim = ndim + element_ndim;
  int64_t *shape = nullptr;
  std::cout << "index offsets.size() " << snode->index_offsets.size()
            << std::endl;
  if (full_ndim > 0) {
    shape = new int64_t[full_ndim];
    for (int i = 0; i < ndim; i++) {
      if (snode->physical_index_position[i] != i) {
        TI_ERROR(
            "SNode has non-sequential physical index mapping, which is not "
            "supported currently for dlpack conversion");
      }
      int axis_shape = snode->shape_along_axis(i);
      std::cout << "axis_shape: " << axis_shape << " physical_index_position "
                << snode->physical_index_position[i] << std::endl;
      shape[i] = axis_shape;
      AxisExtractor axis = snode->extractors[i];
      std::cout << "axis: " << " num_elements_from_root "
                << axis.num_elements_from_root << " shape " << axis.shape
                << " acc_shape " << axis.acc_shape << std::endl;
      // std::cout << " index offset " << snode->index_offsets[i] << std::endl;
    }
    if (element_ndim >= 1) {
      shape[ndim] = n;
    }
    if (element_ndim == 2) {
      shape[ndim + 1] = m;
    }
  }

  int64_t *strides = nullptr;
  if (full_ndim > 0) {
    strides = new int64_t[full_ndim];
    strides[full_ndim - 1] = 1;
    for (int i = full_ndim - 2; i >= 0; i--) {
      strides[i] = strides[i + 1] * shape[i + 1];
    }
  }

  DLManagedTensor *managed_tensor = new DLManagedTensor();

  DLTensor &dl_tensor = managed_tensor->dl_tensor;
  dl_tensor.data = raw_ptr;
  dl_tensor.device.device_type = device_type;
  dl_tensor.device.device_id = 0;
  dl_tensor.ndim = full_ndim;
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
  void *raw_ptr = nullptr;
  std::tie(raw_ptr, device_type) = get_raw_ptr(arch, program, devalloc);

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
  std::tie(data_type_code, element_bits) = get_type_info(ndarray_data_type);

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
