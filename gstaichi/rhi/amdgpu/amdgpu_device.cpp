#include "gstaichi/rhi/amdgpu/amdgpu_device.h"
#include "gstaichi/rhi/llvm/device_memory_pool.h"

#include "gstaichi/jit/jit_module.h"

namespace gstaichi {
namespace lang {

namespace amdgpu {

AmdgpuDevice::AmdgpuDevice() {
  // Initialize the device memory pool
  DeviceMemoryPool::get_instance(Arch::amdgpu, false /*merge_upon_release*/);
}

AmdgpuDevice::AllocInfo AmdgpuDevice::get_alloc_info(
    const DeviceAllocation handle) {
  validate_device_alloc(handle);
  return allocations_[handle.alloc_id];
}

RhiResult AmdgpuDevice::allocate_memory(const AllocParams &params,
                                        DeviceAllocation *out_devalloc) {
  std::cout << "AmdgpuDevice::allocate_memory called, size: " << params.size << std::endl;
  AllocInfo info;
  std::cout << "1" << std::endl;
  auto &mem_pool = DeviceMemoryPool::get_instance(Arch::amdgpu, false /*merge_upon_release*/);
  std::cout << "1" << std::endl;

  bool managed = params.host_read || params.host_write;
  std::cout << "1" << std::endl;
  void *ptr =
      mem_pool.allocate(params.size, DeviceMemoryPool::page_size, managed);
  std::cout << "1" << std::endl;
  if (ptr == nullptr) {
    std::cout << "2" << std::endl;
    return RhiResult::out_of_memory;
  }
  std::cout << "1" << std::endl;

  info.ptr = ptr;
  std::cout << "1" << std::endl;
  info.size = params.size;
  std::cout << "1" << std::endl;
  info.is_imported = false;
  std::cout << "1" << std::endl;
  info.use_cached = false;
  std::cout << "1" << std::endl;
  info.use_preallocated = false;
  std::cout << "1" << std::endl;

  if (info.ptr == nullptr) {
  std::cout << "3" << std::endl;
    return RhiResult::out_of_memory;
  }
  std::cout << "1" << std::endl;

  std::cout << "AMDGPUDevice::allocate_memory memset" << std::endl;
  AMDGPUDriver::get_instance().memset((void *)info.ptr, 0, info.size);

  std::cout << "DeviceAllocation()" << std::endl;
  *out_devalloc = DeviceAllocation{};
  std::cout << "after DeviceAllocation()" << std::endl;
  out_devalloc->alloc_id = allocations_.size();
  out_devalloc->device = this;

  allocations_.push_back(info);
  return RhiResult::success;
}

DeviceAllocation AmdgpuDevice::allocate_memory_runtime(
    const LlvmRuntimeAllocParams &params) {
  AllocInfo info;
  info.size = gstaichi::iroundup(params.size, gstaichi_page_size);
  if (params.host_read || params.host_write) {
    TI_NOT_IMPLEMENTED
  } else {
    info.ptr =
        DeviceMemoryPool::get_instance(Arch::amdgpu, false /*merge_upon_release*/).allocate_with_cache(this, params);
    TI_ASSERT(info.ptr != nullptr);

    AMDGPUDriver::get_instance().memset((void *)info.ptr, 0, info.size);
  }
  info.is_imported = false;
  info.use_cached = true;
  info.use_preallocated = true;

  DeviceAllocation alloc;
  alloc.alloc_id = allocations_.size();
  alloc.device = this;

  allocations_.push_back(info);
  return alloc;
}

uint64_t *AmdgpuDevice::allocate_llvm_runtime_memory_jit(
    const LlvmRuntimeAllocParams &params) {
  params.runtime_jit->call<void *, std::size_t, std::size_t>(
      "runtime_memory_allocate_aligned", params.runtime, params.size,
      gstaichi_page_size, params.result_buffer);
  AMDGPUDriver::get_instance().stream_synchronize(nullptr);
  uint64 *ret{nullptr};
  AMDGPUDriver::get_instance().memcpy_device_to_host(&ret, params.result_buffer,
                                                     sizeof(uint64));
  return ret;
}

void AmdgpuDevice::dealloc_memory(DeviceAllocation handle) {
  // After reset, all allocations are invalid
  if (allocations_.empty()) {
    return;
  }

  validate_device_alloc(handle);
  AllocInfo &info = allocations_[handle.alloc_id];
  if (info.ptr == nullptr) {
    TI_ERROR("the DeviceAllocation is already deallocated");
  }
  TI_ASSERT(!info.is_imported);
  if (info.use_cached) {
    DeviceMemoryPool::get_instance(Arch::amdgpu, false /*merge_upon_release*/).release(info.size, (uint64_t *)info.ptr,
                                             false);
  } else if (!info.use_preallocated) {
    DeviceMemoryPool::get_instance(Arch::amdgpu, false /*merge_upon_release*/).release(info.size, info.ptr);
  }
  info.ptr = nullptr;
}

RhiResult AmdgpuDevice::map(DeviceAllocation alloc, void **mapped_ptr) {
  AllocInfo &info = allocations_[alloc.alloc_id];
  size_t size = info.size;
  info.mapped = new char[size];
  // FIXME: there should be a better way to do this...
  AMDGPUDriver::get_instance().memcpy_device_to_host(info.mapped, info.ptr,
                                                     size);
  *mapped_ptr = info.mapped;
  return RhiResult::success;
}

void AmdgpuDevice::unmap(DeviceAllocation alloc) {
  AllocInfo &info = allocations_[alloc.alloc_id];
  AMDGPUDriver::get_instance().memcpy_host_to_device(info.ptr, info.mapped,
                                                     info.size);
  delete[] static_cast<char *>(info.mapped);
  return;
}

void AmdgpuDevice::memcpy_internal(DevicePtr dst,
                                   DevicePtr src,
                                   uint64_t size) {
  void *dst_ptr =
      static_cast<char *>(allocations_[dst.alloc_id].ptr) + dst.offset;
  void *src_ptr =
      static_cast<char *>(allocations_[src.alloc_id].ptr) + src.offset;
  AMDGPUDriver::get_instance().memcpy_device_to_device(dst_ptr, src_ptr, size);
}

DeviceAllocation AmdgpuDevice::import_memory(void *ptr, size_t size) {
  AllocInfo info;
  info.ptr = ptr;
  info.size = size;
  info.is_imported = true;

  DeviceAllocation alloc;
  alloc.alloc_id = allocations_.size();
  alloc.device = this;

  allocations_.push_back(info);
  return alloc;
}

}  // namespace amdgpu
}  // namespace lang
}  // namespace gstaichi
