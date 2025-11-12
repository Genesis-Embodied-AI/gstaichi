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
pybind11::capsule field_to_dlpack(Program *program,
                                    pybind11::object owner,
                                    SNode *snode) {
  // auto *owner_holder = new pybind11::object(owner);
  // new pybind11::object(owner);
  std::cout << "snode " << snode << " id " << snode->id << " depth " << snode->depth << " name " << snode->name << " chunk_size " << snode->chunk_size <<
     " cell_size_bytes " << snode->cell_size_bytes << " offset_bytes_in_parent_cell " << snode->offset_bytes_in_parent_cell <<
     " is_path_all_dense " << snode->is_path_all_dense << " index_offsets.size() " << snode->index_offsets.size() << 
     " num_active_indices " << snode->num_active_indices << " num_cells_per_container " << snode->num_cells_per_container <<
     " ch.size() " << snode->ch.size() << " snode_tree_id " << snode->get_snode_tree_id() << std::endl;
  for(int i=0; i < snode->index_offsets.size(); i++) {
    std::cout << "  index_offsets[" << i << "] = " << snode->index_offsets[i] << std::endl;
  }
  for(int i=0; i < snode->num_active_indices; i++) {
    std::cout << "  physical_index_position[" << i << "] = " << snode->physical_index_position[i] << std::endl;
  }
  for(int i=0; i < snode->num_active_indices; i++) {
    AxisExtractor &extractor = snode->extractors[i];
    std::cout << "  extractor[" << i << "] active " << extractor.active << " num_elements_from_root " << extractor.num_elements_from_root <<
    " shape " << extractor.shape << " acc_shape " << extractor.acc_shape << std::endl;
  }

  auto deleter = [](PyObject *capsule) {};

  int tree_id = snode->get_snode_tree_id();
  DevicePtr tree_device_ptr = program->get_snode_tree_device_ptr(tree_id);
  std::cout << "tree_device_ptr " << (void *)&tree_device_ptr << " offset " << tree_device_ptr.offset << std::endl;
  int field_in_tree_offset = program->get_field_in_tree_offset(tree_id, snode);
  std::cout << "field_in_tree_offset " << field_in_tree_offset << std::endl;

  int ndim = snode->num_active_indices;
  int64_t *shape = nullptr;
  if (ndim > 0) {
    shape = new int64_t[ndim];
    for(int i = 0; i < ndim; i++) {
      int axis_shape = snode->shape_along_axis(i);
      shape[i] = axis_shape;
      std::cout << "  shape[" << i << "] = " << shape[i] << std::endl;
    }
  }

  void *raw_ptr = nullptr;
  DLDeviceType device_type = DLDeviceType::kDLCPU;

  Arch arch = program->compile_config().arch;
  if (arch_is_cpu(arch)) {
    cpu::CpuDevice *cpu_device = static_cast<cpu::CpuDevice *>(devalloc.device);
  // cpu::CpuDevice *cpu_device = dynamic_cast<cpu::CpuDevice *>(tree_device_ptr.device);
  // if (cpu_device != nullptr) {
    std::cout << "cpu device is non null" << std::endl;
    cpu::CpuDevice::AllocInfo alloc_info = cpu_device->get_alloc_info(tree_device_ptr);
    raw_ptr = alloc_info.ptr;
    std::cout << "raw ptr " << raw_ptr << std::endl;

    int *ptr_as_int = (int *)raw_ptr;
    std::cout << "[0]" << ptr_as_int[0] << " [1] " << ptr_as_int[1] << " [2] " << ptr_as_int[2] << " [3] " << ptr_as_int[3] << std::endl;
  }


  MyData *my_data = new MyData();
  my_data->value = 42;
  pybind11::capsule capsule = pybind11::capsule(
      static_cast<void *>(my_data), "dltensor", deleter);
  return capsule;
}

pybind11::capsule ndarray_to_dlpack(Program *program,
                                    pybind11::object owner,
                                    Ndarray *ndarray) {
  auto *owner_holder = new pybind11::object(owner);

  DeviceAllocation devalloc = ndarray->get_device_allocation();

  void *raw_ptr = nullptr;
  DLDeviceType device_type = DLDeviceType::kDLCPU;

  Arch arch = program->compile_config().arch;
  if (arch_is_cpu(arch)) {
    cpu::CpuDevice *cpu_device = static_cast<cpu::CpuDevice *>(devalloc.device);
    cpu::CpuDevice::AllocInfo alloc_info = cpu_device->get_alloc_info(devalloc);
    raw_ptr = alloc_info.ptr;
  }
#if TI_WITH_CUDA
  else if (arch_is_cuda(arch)) {
    cuda::CudaDevice *cuda_device =
        static_cast<cuda::CudaDevice *>(devalloc.device);
    cuda::CudaDevice::AllocInfo alloc_info =
        cuda_device->get_alloc_info(devalloc);
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
